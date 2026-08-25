#!/usr/bin/env bash
# MATH-500 evaluation on NRP, driven the same way as ../run.sh.
#
#   ./evaluate/run_eval.sh              code -> job -> logs
#   ./evaluate/run_eval.sh code         only refresh the eval-code ConfigMap
#   ./evaluate/run_eval.sh submit       only (re)submit the Job
#   ./evaluate/run_eval.sh logs         follow the current job's logs
#   ./evaluate/run_eval.sh status       job, pod, recent events
#   ./evaluate/run_eval.sh fetch        copy results out of the PVC to ./eval-results
#   ./evaluate/run_eval.sh clean        delete the Job (results and PVC survive)
#
# Reads ../.env for IMAGE and K8S_NAMESPACE but never writes to it. No image rebuild
# is involved: the script is shipped as a ConfigMap into the existing training image.
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

if [ ! -f "${REPO_ROOT}/.env" ]; then
  echo "error: ${REPO_ROOT}/.env not found" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "${REPO_ROOT}/.env"
set +a

: "${IMAGE:?set IMAGE in .env}"
: "${K8S_NAMESPACE:?set K8S_NAMESPACE in .env}"
JOB_NAME="qwen-eval"
KUBECTL=(kubectl --namespace "${K8S_NAMESPACE}")

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step_code() {
  log "Refreshing ConfigMap eval-code"
  "${KUBECTL[@]}" delete configmap eval-code --ignore-not-found
  "${KUBECTL[@]}" create configmap eval-code --from-file=eval_math500.py
}

step_submit() {
  log "Submitting Job ${JOB_NAME}"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
  sed "s|\${IMAGE}|${IMAGE}|g" k8s/eval-job.yaml | "${KUBECTL[@]}" apply -f -
}

eval_pod() {
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" \
    -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
}

step_logs() {
  log "Waiting for the ${JOB_NAME} pod"
  pod=""; last=""
  for i in $(seq 1 120); do
    pod="$(eval_pod)"
    if [ -n "${pod}" ]; then
      phase="$("${KUBECTL[@]}" get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
      case "${phase}" in Running | Succeeded | Failed) break ;; esac
      reason="$("${KUBECTL[@]}" get pod "${pod}" \
        -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].message}' 2>/dev/null || true)"
      msg="${phase}: $(printf '%s' "${reason:-$phase}" | cut -c1-150)"
      if [ "${msg}" != "${last}" ]; then printf '\n  [%3ds] %s\n' "$((i * 5))" "${msg}"; last="${msg}"
      else printf '.'; fi
    fi
    sleep 5
  done
  echo
  [ -n "${pod}" ] || { echo "no pod created; try './run_eval.sh status'" >&2; exit 1; }
  "${KUBECTL[@]}" logs -f "${pod}"
}

step_status() {
  "${KUBECTL[@]}" get job "${JOB_NAME}" || true
  "${KUBECTL[@]}" get pods -l "job-name=${JOB_NAME}" || true
  "${KUBECTL[@]}" get events --sort-by=.lastTimestamp | tail -10 || true
}

step_fetch() {
  log "Copying results out of the PVC"
  "${KUBECTL[@]}" apply -f "${REPO_ROOT}/k8s/data-shell.yaml"
  until [ "$("${KUBECTL[@]}" get pod data-shell -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
    sleep 3
  done
  mkdir -p ../eval-results
  "${KUBECTL[@]}" exec data-shell -- tar cf - -C /data/eval . | tar xf - -C ../eval-results
  echo "results in ${REPO_ROOT}/eval-results"
  echo "delete the helper pod when done: kubectl delete pod data-shell"
}

step_clean() {
  log "Deleting Job ${JOB_NAME}"
  "${KUBECTL[@]}" delete job "${JOB_NAME}" --ignore-not-found
}

case "${1:-all}" in
  code) step_code ;;
  submit) step_submit ;;
  logs) step_logs ;;
  status) step_status ;;
  fetch) step_fetch ;;
  clean) step_clean ;;
  all) step_code; step_submit; step_logs ;;
  *) echo "usage: $0 [all|code|submit|logs|status|fetch|clean]" >&2; exit 1 ;;
esac
