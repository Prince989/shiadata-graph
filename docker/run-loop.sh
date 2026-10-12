#!/bin/sh
# Restart one ETL command forever. Exit 0 means the budget finished (re-run).
# Exit 2 means the provider is down or the free-tier keys are exhausted.
# Any other status is a crash.
if [ "$#" -eq 0 ]; then
  echo "usage: run-loop.sh command..." >&2
  exit 1
fi
while true; do
  "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "run-loop: command finished (exit 0); sleeping 15s"
    sleep 15
  elif [ "$code" -eq 2 ]; then
    echo "run-loop: provider unavailable (exit 2); sleeping 900s"
    sleep 900
  else
    echo "run-loop: command failed (exit ${code}); sleeping 30s"
    sleep 30
  fi
done
