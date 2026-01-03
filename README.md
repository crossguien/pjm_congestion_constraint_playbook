# PJM Congestion & Constraint Playbook (Desk-Style)

This repo builds a repeatable congestion and basis-risk playbook using public PJM LMP data.

It produces:
- Top locations by congestion intensity and tail risk
- Top basis (location-to-location) pairs and when they blow out
- Seasonality diagnostics (hour-of-day, day-of-week, month)
- Exportable tables and figures you can reference in interviews

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt
```

## Run (out-of-box)
```bash
python src/main.py --days 60 --market day_ahead --outdir outputs
```

## Variants
```bash
python src/main.py --days 60 --market real_time --outdir outputs
python src/main.py --days 90 --ref_location "PJM RTO" --market day_ahead --outdir outputs
python src/main.py --days 60 --max_locations 25 --outdir outputs
```

## Notes
- "Constraints" can mean transmission constraint IDs; public endpoints vary.
  This project focuses on observable congestion outcomes in LMP components and basis behavior.
