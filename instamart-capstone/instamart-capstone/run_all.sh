#!/usr/bin/env bash
# Convenience wrapper: runs every step in order (same commands as the README Reproduce section).
set -e
python generate_data.py
python src/part_a_clean.py
python src/validate.py
python src/part_a_analysis.py
python src/part_b.py
python -m src.part_c
python -m src.part_d
