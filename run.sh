#!/usr/bin/env bash
# One-command pipeline: data -> image -> secrets -> job -> logs.
#
#   ./run.sh              everything, then follow the logs
#   ./run.sh data         only rebuild data/{train,val}.jsonl
#   ./run.sh image        only docker login + build + push
#   ./run.sh secrets      only refresh the sft-env and nrp-registry Secrets
#   ./run.sh submit       only (re)submit the Job
#   ./run.sh logs         follow the current job's logs
#   ./run.sh status       job, pods and recent events
#   ./run.sh clean        delete the job (Secrets and PVC are kept)
#
# Everything is configured through .env; see .env.example.
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  echo "error: .env not found. Run: cp .env.example .env && \$EDITOR .env" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a

: "${IMAGE:?set IMAGE in .env}"
: "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
PVC_NAME="${PVC_NAME:-qwen-sft-data}"
JOB_NAME="${JOB_NAME:-qwen-sft}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NRP_REGISTRY="${IMAGE%%/*}"
# The path segment after the host is the GitLab namespace, which is also the
# registry login user for a personal access token.
NRP_REGISTRY_USER="${NRP_REGISTRY_USER:-$(echo "${IMAGE}" | cut -d/ -f2)}"

KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
step_data() {
  log "Building data/train.jsonl"
  "${PYTHON_BIN}" src/prepare_data.py --input-dir bespoke-v2 --output-dir data
}

step_image() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"
  log "docker login ${NRP_REGISTRY} as ${NRP_REGISTRY_USER}"
  printf '%s' "${NRP_REGISTRY_TOKEN}" |
    docker login "${NRP_REGISTRY}" --username "${NRP_REGISTRY_USER}" --password-stdin

  log "Building ${IMAGE}"
  docker build --platform linux/amd64 -t "${IMAGE}" .

  log "Pushing ${IMAGE}"
  docker push "${IMAGE}"
}

step_secrets() {
  : "${NRP_REGISTRY_TOKEN:?set NRP_REGISTRY_TOKEN in .env}"

  log "Refreshing Secret sft-env (from .env)"
  "${KUBECTL[@]}" delete secret sft-env --ignore-not-found
  "${KUBECTL[@]}" create secret generic sft-env --from-env-file=.env

  log "Refreshing Secret nrp-registry (image pull)"
  "${KUBECTL[@]}" delete secret nrp-registry --ignore-not-found
  "${KUBECTL[@]}" create secret docker-registry nrp-registry \
    --docker-server="${NRP_REGISTRY}" \
    --docker-username="${NRP_REGISTRY_USER}" \
    --docker-password="${NRP_REGISTRY_TOKEN}"
}

step_pvc() {
  if "${KUBECTL[@]}" get pvc "${PVC_NAME}" >/dev/null 2>&1; then
    log "PVC ${PVC_NAME} already exists"
  else
    log "Creating PVC ${PVC_NAME}"
    "${KUBECTL[@]}" apply -f k8s/pvc.yaml
  fi
}

step_submit() {
  log "Submitting Job ${JOB_NAME}"
  # A Job's pod template is immutable, so an existing Job has to go first.
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
  # The manifest carries an ${IMAGE} placeholder; k8s cannot read .env itself.
  sed "s|\${IMAGE}|${IMAGE}|g" k8s/job.yaml | "${KUBECTL[@]}" apply -f -
}

job_pod() {
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
}

step_logs() {
  log "Waiting for the ${JOB_NAME} pod to start"
  pod=""
  for _ in $(seq 1 120); do
    pod="$(job_pod)"
    if [ -n "${pod}" ]; then
      phase="$("${KUBECTL[@]}" get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      case "${phase}" in
        Running | Succeeded | Failed) break ;;
      esac
    fi
    printf '.'
    sleep 5
  done
  echo
  if [ -z "${pod}" ]; then
    echo "no pod yet; check './run.sh status'" >&2
    exit 1
  fi
  "${KUBECTL[@]}" logs -f "${pod}"
}

step_status() {
  "${KUBECTL[@]}" get job "${JOB_NAME}" || true
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" || true
  "${KUBECTL[@]}" get events --sort-by=.lastTimestamp | tail -15 || true
}

step_clean() {
  log "Deleting Job ${JOB_NAME}"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "${1:-all}" in
  data) step_data ;;
  image) step_image ;;
  secrets) step_secrets ;;
  pvc) step_pvc ;;
  submit) step_submit ;;
  logs) step_logs ;;
  status) step_status ;;
  clean) step_clean ;;
  all)
    [ -f data/train.jsonl ] || step_data
    step_image
    step_secrets
    step_pvc
    step_submit
    step_logs
    ;;
  *)
    echo "usage: $0 [all|data|image|secrets|pvc|submit|logs|status|clean]" >&2
    exit 1
    ;;
esac
