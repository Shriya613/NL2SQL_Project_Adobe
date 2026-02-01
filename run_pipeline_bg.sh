#!/usr/bin/env bash

# NOTE FOR MY BEAGLES: run with 1 after for generating data, and 2 after for filtering and schema change
# Ex. ./run_pipeline_bg.sh 1 # to generate data
# Ex. ./run_pipeline_bg.sh 2 # to filter and schema change
set -euo pipefail

# Absolute project root - use directory where script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/pipeline/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Parse command-line arguments
STEP="${1:-}"  # Step 1 or 2 (optional)

# Build command arguments
PYTHON_ARGS=()
if [ -n "$STEP" ]; then
    PYTHON_ARGS+=("--step" "$STEP")
fi

echo "Starting pipeline at $(date) ..."
echo "Logging to: $LOG_FILE"
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    echo "Arguments: ${PYTHON_ARGS[*]}"
fi

nohup python3 "$PROJECT_ROOT/pipeline/run.py" "${PYTHON_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "Pipeline running in background with PID: $PID"
echo "$PID" > "$LOG_DIR/last_pipeline_pid.txt"

echo ""
echo "📋📋📋📋📋 FOLLOWING LOGS (Press Ctrl+C to stop following, process will continue) 📋📋📋📋📋"
echo ""

# Wait a moment for log file to be created
sleep 1

# Tail the log file with follow mode
tail -f "$LOG_FILE"


