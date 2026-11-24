#!/bin/bash
# Puntboom daily pipeline: collect → retrain → predict
# Runs via cron at 6am daily

set -euo pipefail

PROJECT_DIR="/Users/Dijha/Desktop/puntboom"
PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG="$PROJECT_DIR/logs/pipeline_$(date +%Y%m%d).log"

mkdir -p "$PROJECT_DIR/logs"

echo "=== Puntboom pipeline started at $(date) ===" >> "$LOG"

# Step 1: Collect data (football, NHL, euro hockey, basketball, tennis, odds)
echo "[$(date +%H:%M:%S)] Collecting data..." >> "$LOG"
$PYTHON -c "
import asyncio, logging
logging.basicConfig(level=logging.INFO)
from puntboom.tasks.collect import collect_all_data
asyncio.run(collect_all_data())
" >> "$LOG" 2>&1 || echo "WARNING: Collection had errors" >> "$LOG"

# Step 2: Retrain models with new results
echo "[$(date +%H:%M:%S)] Retraining models..." >> "$LOG"
$PYTHON -c "
import asyncio, logging
logging.basicConfig(level=logging.INFO)
from puntboom.tasks.retrain import retrain_all_models
result = asyncio.run(retrain_all_models())
print(result)
" >> "$LOG" 2>&1 || echo "WARNING: Retraining had errors" >> "$LOG"

# Step 3: Generate predictions for upcoming matches
echo "[$(date +%H:%M:%S)] Generating predictions..." >> "$LOG"
$PYTHON -c "
import asyncio, logging
logging.basicConfig(level=logging.INFO)
from puntboom.tasks.predict import generate_predictions
asyncio.run(generate_predictions())
" >> "$LOG" 2>&1 || echo "WARNING: Prediction generation had errors" >> "$LOG"

echo "=== Pipeline finished at $(date) ===" >> "$LOG"
