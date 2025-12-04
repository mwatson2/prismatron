#!/bin/bash

# Script to extract relevant logs from journalctl around a specific time
# Usage: ./extract_crash_logs.sh [time] [duration_minutes_before] [duration_minutes_after]
# Example: ./extract_crash_logs.sh "23:01" 1 2
# This would get logs from 1 minute before to 2 minutes after 23:01

TIME=${1:-"23:01"}
BEFORE_MIN=${2:-1}
AFTER_MIN=${3:-2}
SERVICE="prismatron.service"

echo "Extracting logs around $TIME (${BEFORE_MIN} min before, ${AFTER_MIN} min after) for service: $SERVICE"
echo "=================================================="

# Use journalctl's built-in time parsing
# Get logs from the last 24 hours and grep for our time window
echo "Looking for CRITICAL/ERROR messages:"
echo ""

# First, get any CRITICAL or thread crash messages from the last hour
journalctl -u $SERVICE --since="1 hour ago" --no-pager | grep -E "(CRITICAL|RENDERER THREAD|renderer thread|Thread crashed|thread died|heartbeat)" -A 3 -B 3

echo ""
echo "=================================================="
echo "Looking for renderer/thread errors:"
echo ""

journalctl -u $SERVICE --since="1 hour ago" --no-pager | grep -E "(Error in rendering loop|render_frame_at_timestamp|Failed to render|Frame rendering failed)" -A 5 -B 2

echo ""
echo "=================================================="
echo "Last 50 lines of recent logs:"
echo ""

journalctl -u $SERVICE --since="10 minutes ago" --no-pager | tail -n 50
