#!/bin/bash

set -e

START_YEAR=${1:-2005}
END_YEAR=${2:-2025}

echo "======================================================================"
echo "MANITOBA WILDFIRE NEXT-DAY FEATURE BATCH"
echo "======================================================================"
echo "Year range: ${START_YEAR}-${END_YEAR}"
echo "Started: $(date)"
echo

for YEAR in $(seq "$START_YEAR" "$END_YEAR")
do
    echo
    echo "######################################################################"
    echo "YEAR ${YEAR}"
    echo "######################################################################"
    echo

    python src/features/build_weather_features.py "$YEAR"

    echo
    echo "COMPLETED YEAR: ${YEAR}"
done

echo
echo "======================================================================"
echo "FEATURE BATCH COMPLETED"
echo "======================================================================"
echo "Processed years: ${START_YEAR}-${END_YEAR}"
echo "Finished: $(date)"
