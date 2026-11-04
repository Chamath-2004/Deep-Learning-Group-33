#!/usr/bin/env bash
set -u

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
elif [ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
else
    echo "Virtual environment Python was not found in .venv" >&2
    exit 1
fi

echo "Testing word-overlap coursework target..."
"$PYTHON" "$ROOT_DIR/factchecking_main.py" --mode word_overlap
word_overlap_status=$?

echo "Testing EntailmentFactChecker.predict..."
"$PYTHON" "$ROOT_DIR/test_entailment_predict.py"
predict_status=$?

echo "Testing entailment coursework target..."
"$PYTHON" "$ROOT_DIR/factchecking_main.py" --mode entailment
entailment_status=$?

if [ "$word_overlap_status" -ne 0 ] || [ "$predict_status" -ne 0 ] || [ "$entailment_status" -ne 0 ]; then
    exit 1
fi

echo "All coursework targets passed."
