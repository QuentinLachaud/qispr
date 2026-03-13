#!/bin/bash

set -u

APP_DIR="/Users/quentinlachaud/code/qispr"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
SCRIPT_PATH="$APP_DIR/dictate.py"
LOG_FILE="/tmp/qispr.log"
PID_FILE="/tmp/qispr.pid"

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(<"$PID_FILE")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        exit 0
    fi
    rm -f "$PID_FILE"
fi

cd "$APP_DIR" || exit 1
export PYTHONUNBUFFERED=1
/usr/bin/nohup "$PYTHON_BIN" "$SCRIPT_PATH" </dev/null >>"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"
