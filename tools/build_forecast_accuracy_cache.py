#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "forecast_accuracy_all_cities.py"
OUTFILE = ROOT / "state" / "forecast_accuracy_summary.json"


def load_accuracy_module():
    spec = importlib.util.spec_from_file_location("forecast_accuracy_all_cities", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_accuracy_module()

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source": "Open-Meteo historical forecast vs IEM archived NWS CLI",
        "days_requested": 30,
        "cities": {},
    }

    print("Building forecast accuracy cache...")
    print(f"Output: {OUTFILE}")
    print()

    for city in mod.CITIES:
        key = city["key"]
        print(f"Fetching {city['name']} ({city['station']})...", flush=True)

        try:
            rows, hi, lo = mod.build_rows(city, days=30)

            latest_high = mod.yesterday_line(rows, "high")
            latest_low = mod.yesterday_line(rows, "low")

            payload["cities"][key] = {
                "key": key,
                "name": city["name"],
                "station": city["station"],
                "timezone": city["tz"],
                "cli_pil": city["cli_pil"],
                "high": {
                    "days": hi["count"],
                    "avg_abs_miss": hi["avg_abs"],
                    "bias": hi["bias"],
                    "within_1_pct": hi["within_1"],
                    "latest": latest_high,
                },
                "low": {
                    "days": lo["count"],
                    "avg_abs_miss": lo["avg_abs"],
                    "bias": lo["bias"],
                    "within_1_pct": lo["within_1"],
                    "latest": latest_low,
                },
            }

        except Exception as e:
            payload["cities"][key] = {
                "key": key,
                "name": city["name"],
                "station": city["station"],
                "error": repr(e),
            }

    OUTFILE.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print()
    print("Done.")
    print(f"Wrote {OUTFILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
