#!/usr/bin/env bash

set -u

if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/run_era5_batch.sh START_YEAR END_YEAR"
    echo "Example: bash scripts/run_era5_batch.sh 2006 2010"
    exit 1
fi

START_YEAR="$1"
END_YEAR="$2"

if ! [[ "$START_YEAR" =~ ^[0-9]{4}$ && "$END_YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Start year and end year must be four-digit years."
    exit 1
fi

if (( START_YEAR < 2005 || END_YEAR > 2025 || START_YEAR > END_YEAR )); then
    echo "Year range must be between 2005 and 2025."
    exit 1
fi

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

PYTHON="$PROJECT_ROOT/venv/bin/python"

DOWNLOAD_SCRIPT="$PROJECT_ROOT/src/data/download_era5_fire_season.py"
PROCESS_SCRIPT="$PROJECT_ROOT/src/data/process_era5_fire_season.py"

LOG_FOLDER="$PROJECT_ROOT/outputs/logs"
mkdir -p "$LOG_FOLDER"

LOG_FILE="$LOG_FOLDER/era5_${START_YEAR}_${END_YEAR}_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON"
    exit 1
fi

if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
    echo "Downloader not found: $DOWNLOAD_SCRIPT"
    exit 1
fi

if [[ ! -f "$PROCESS_SCRIPT" ]]; then
    echo "Processor not found: $PROCESS_SCRIPT"
    exit 1
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================"
echo "ERA5-Land multi-year pipeline"
echo "======================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Year range: ${START_YEAR}-${END_YEAR}"
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo

for year in $(seq "$START_YEAR" "$END_YEAR"); do
    echo
    echo "######################################################################"
    echo "YEAR $year"
    echo "######################################################################"

    echo
    echo "Downloading and validating ERA5-Land inputs..."

    if ! "$PYTHON" "$DOWNLOAD_SCRIPT" --year "$year"; then
        echo
        echo "FAILED: Download stage for $year"
        echo "The batch has stopped. Completed years remain valid."
        exit 1
    fi

    echo
    echo "Processing annual weather and wildfire tables..."

    if ! "$PYTHON" "$PROCESS_SCRIPT" --year "$year"; then
        echo
        echo "FAILED: Processing stage for $year"
        echo "The batch has stopped. Completed years remain valid."
        exit 1
    fi

    echo
    echo "COMPLETED YEAR: $year"
done

echo
echo "======================================================================"
echo "BATCH COMPLETED"
echo "======================================================================"
echo "Processed years: ${START_YEAR}-${END_YEAR}"
echo "Finished: $(date)"
echo "Log file: $LOG_FILE"
