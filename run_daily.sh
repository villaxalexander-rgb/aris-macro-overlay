#!/bin/bash
# A.R.I.S Macro Overlay System — Daily Auto-Run
# This script runs the signal pipeline, commits results, and pushes to GitHub.
# Designed to run via cron or launchd at 6:00 AM daily.

set -e

PROJECT_DIR="$HOME/Desktop/aris-macro-overlay"
LOG_FILE="$PROJECT_DIR/logs/daily_run.log"
DATE=$(date +%Y-%m-%d)

cd "$PROJECT_DIR"
mkdir -p logs

echo "========================================" >> "$LOG_FILE"
echo "A.R.I.S Daily Run — $DATE $(date +%H:%M)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Run the pipeline
echo "Running pipeline..." >> "$LOG_FILE"
python3 main.py >> "$LOG_FILE" 2>&1

# Git commit and push the daily signal JSON
echo "Committing to GitHub..." >> "$LOG_FILE"
git add data/signals/ logs/
git diff --cached --quiet || {
    git commit -m "Daily signal run: $DATE"
    git push origin main
    echo "Pushed to GitHub." >> "$LOG_FILE"
}

echo "Done." >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

deactivate
