# PJM Congestion & Constraint Playbook (Desk-Style)

Congestion risk in PJM is highly localized, non-linear, and seasonal. This playbook systematizes how desks identify where congestion concentrates, when it binds, and how basis relationships behave during stress periods using observable market outcomes.

The goal is to convert raw LMP data into repeatable congestion and basis-risk insights.

It produces:
- Top locations by congestion intensity and tail risk

  Identify nodes where congestion is both frequent and severe.

- Basis pair rankings

  Highlight location pairs that consistently blow out under stress.

- Seasonality diagnostics

  Reveal when congestion and basis risk are most likely to materialize by hour, day, and month.

For example, recurring congestion during specific peak hours suggests structural constraints rather than transient noise.

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

## Trade intuition

- Persistent congestion intensity indicates structural transmission limits.

- Basis blowouts during specific hours point to constraint-driven risk.

- Seasonal recurrence supports directional or option-like exposure rather than flat hedges.

## Example desk workflow

- Run the playbook to rank congestion-prone locations.

- Identify basis pairs with asymmetric tail behavior.

- Focus risk during historically active hours and seasons.

- Use results to inform virtuals, FTR selection, or nodal exposure.

## Assumptions and limitations

- Focuses on observable congestion outcomes, not proprietary constraint forecasts.

- Offline mode uses synthetic data for demonstration only.

- Public endpoints may vary in constraint naming and availability.

- Intended as a screening and risk-framing tool, not a standalone trading signal.

## Production extensions

- Integrate constraint-level metadata and outage information.

- Automate rolling congestion regime detection.

- Combine with DA vs RT spread models for unified risk views.

- Scale to portfolio-level basis and congestion exposure tracking.

## Notes
- "Constraints" can mean transmission constraint IDs; public endpoints vary.
  This project focuses on observable congestion outcomes in LMP components and basis behavior.
- Offline mode uses synthetic sample data for demonstration only (not actual PJM prices).
