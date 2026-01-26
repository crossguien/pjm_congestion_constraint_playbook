# PJM Congestion & Constraint Playbook (Desk-Style)

This repo builds a repeatable congestion and basis-risk playbook using public PJM LMP data.

It produces:
- Top locations by congestion intensity and tail risk
- Top basis (location-to-location) pairs and when they blow out
- Seasonality diagnostics (hour-of-day, day-of-week, month)

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

## Demo without a PJM API key (offline sample)
```bash
python src/main.py --mode offline --days 60 --market day_ahead --outdir outputs
```

## Run with a PJM API key (online)
```bash
export PJM_API_KEY="your_key_here"
python src/main.py --mode online --days 60 --market day_ahead --outdir outputs
```

## Auto mode
If `PJM_API_KEY` is set, the script uses live PJM data. If not, it falls back to
synthetic sample data so the workflow still runs end-to-end.
```bash
python src/main.py --mode auto --days 60 --market day_ahead --outdir outputs
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
- Offline mode uses synthetic sample data for demonstration only (not actual PJM prices).
