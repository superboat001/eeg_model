#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${PROJECT_ROOT}/exp/.background"
PID_FILE="${STATE_DIR}/latest.pid"
LOG_PATH_FILE="${STATE_DIR}/latest.log.path"
MODE_FILE="${STATE_DIR}/latest.mode"
COMMAND_FILE="${STATE_DIR}/latest.command"

usage() {
    cat <<'EOF'
Usage:
  ./run_eeg_background.sh [start] [train_eeg.py options]
  ./run_eeg_background.sh sanity [train_eeg.py options]
  ./run_eeg_background.sh status
  ./run_eeg_background.sh tail [lines]
  ./run_eeg_background.sh help

Examples:
  ./run_eeg_background.sh
  ./run_eeg_background.sh start --device cuda:0 --run-name segments_v3
  ./run_eeg_background.sh sanity --device cuda:0
  ./run_eeg_background.sh status
  ./run_eeg_background.sh tail 120
EOF
}

resolve_conda() {
    local conda_bin="${CONDA_BIN:-}"
    if [[ -z "${conda_bin}" ]]; then
        conda_bin="$(command -v conda || true)"
    fi
    if [[ -z "${conda_bin}" && -x "/home/cgz/miniconda3/bin/conda" ]]; then
        conda_bin="/home/cgz/miniconda3/bin/conda"
    fi
    if [[ -z "${conda_bin}" || ! -x "${conda_bin}" ]]; then
        echo "Error: conda executable was not found. Set CONDA_BIN explicitly." >&2
        exit 1
    fi
    printf '%s\n' "${conda_bin}"
}

read_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        tr -d '[:space:]' < "${PID_FILE}"
    fi
}

is_running() {
    local pid
    pid="$(read_pid)"
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

latest_log() {
    if [[ ! -f "${LOG_PATH_FILE}" ]]; then
        echo "Error: no background EEG job has been recorded." >&2
        exit 1
    fi
    local log_file
    log_file="$(<"${LOG_PATH_FILE}")"
    if [[ ! -f "${log_file}" ]]; then
        echo "Error: recorded log file does not exist: ${log_file}" >&2
        exit 1
    fi
    printf '%s\n' "${log_file}"
}

start_job() {
    local mode="$1"
    shift
    mkdir -p "${STATE_DIR}"

    if is_running; then
        local current_pid current_mode current_log
        current_pid="$(read_pid)"
        current_mode="$(<"${MODE_FILE}")"
        current_log="$(<"${LOG_PATH_FILE}")"
        echo "Error: a background EEG job is already running." >&2
        echo "PID=${current_pid} MODE=${current_mode} LOG=${current_log}" >&2
        exit 1
    fi

    local conda_bin timestamp log_file pid
    conda_bin="$(resolve_conda)"
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    log_file="${STATE_DIR}/${timestamp}_${mode}.log"
    local -a command=(
        "${conda_bin}" run --no-capture-output -n cgz
        python -u "${PROJECT_ROOT}/train_eeg.py"
    )
    if [[ "${mode}" == "sanity" ]]; then
        command+=(--sanity-overfit)
    fi
    command+=("$@")

    (
        cd "${PROJECT_ROOT}"
        nohup "${command[@]}" > "${log_file}" 2>&1 < /dev/null &
        pid=$!
        printf '%s\n' "${pid}" > "${PID_FILE}"
        printf '%s\n' "${log_file}" > "${LOG_PATH_FILE}"
        printf '%s\n' "${mode}" > "${MODE_FILE}"
        printf '%q ' "${command[@]}" > "${COMMAND_FILE}"
        printf '\n' >> "${COMMAND_FILE}"
    )

    pid="$(read_pid)"
    sleep 1
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "Error: the background process exited during startup." >&2
        tail -n 80 "${log_file}" >&2 || true
        exit 1
    fi

    echo "Background EEG ${mode} job started successfully."
    echo "PID: ${pid}"
    echo "Launcher log: ${log_file}"
    echo "The SSH connection can now be closed safely."
    echo "Status: ${PROJECT_ROOT}/run_eeg_background.sh status"
    echo "Logs:   ${PROJECT_ROOT}/run_eeg_background.sh tail"
}

show_status() {
    mkdir -p "${STATE_DIR}"
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "No background EEG job has been recorded."
        exit 1
    fi
    local pid mode log_file
    pid="$(read_pid)"
    mode="$(<"${MODE_FILE}")"
    log_file="$(<"${LOG_PATH_FILE}")"
    if is_running; then
        echo "RUNNING: PID=${pid} MODE=${mode}"
        ps -p "${pid}" -o pid=,etime=,stat=,cmd=
    else
        echo "NOT RUNNING: last PID=${pid} MODE=${mode}"
    fi
    echo "Launcher log: ${log_file}"
    if [[ -f "${COMMAND_FILE}" ]]; then
        echo "Command: $(<"${COMMAND_FILE}")"
    fi
}

follow_log() {
    local lines="${1:-100}"
    if [[ ! "${lines}" =~ ^[0-9]+$ ]] || (( lines < 1 )); then
        echo "Error: tail line count must be a positive integer." >&2
        exit 1
    fi
    local log_file
    log_file="$(latest_log)"
    echo "Following ${log_file}; press Ctrl-C to stop viewing (training continues)."
    exec tail -n "${lines}" -f "${log_file}"
}

action="${1:-start}"
if (( $# > 0 )); then
    shift
fi

case "${action}" in
    start)
        start_job "train" "$@"
        ;;
    sanity)
        start_job "sanity" "$@"
        ;;
    status)
        show_status
        ;;
    tail)
        follow_log "${1:-100}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Error: unknown action: ${action}" >&2
        usage >&2
        exit 2
        ;;
esac
