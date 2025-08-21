#!/bin/bash

# Script to extract relevant logs from journalctl around a specific time
# Usage: ./extract_crash_logs.sh [time] [duration_before] [duration_after]
# Example: ./extract_crash_logs.sh "23:01:49" 30 60
# This would get logs from 30 seconds before to 60 seconds after 23:01:49

TIME=${1:-"23:01:49"}
BEFORE=${2:-30}
AFTER=${3:-60}
SERVICE="prismatron.service"

echo "Extracting logs around $TIME (${BEFORE}s before, ${AFTER}s after) for service: $SERVICE"
echo "=================================================="

# Get today's date
TODAY=$(date +%Y-%m-%d)

# Calculate time range
START_TIME=$(date -d "$TODAY $TIME - $BEFORE seconds" "+%Y-%m-%d %H:%M:%S")
END_TIME=$(date -d "$TODAY $TIME + $AFTER seconds" "+%Y-%m-%d %H:%M:%S")

echo "Time range: $START_TIME to $END_TIME"
echo ""

# Extract logs with context
journalctl -u $SERVICE --since="$START_TIME" --until="$END_TIME" --no-pager | grep -E "(CRITICAL|ERROR|RENDERER|renderer|thread|Thread|crash|failed|Exception|Traceback)" -A 2 -B 2

echo ""
echo "=================================================="
echo "Full logs for the time range (last 100 lines):"
echo ""

journalctl -u $SERVICE --since="$START_TIME" --until="$END_TIME" --no-pager | tail -n 100
