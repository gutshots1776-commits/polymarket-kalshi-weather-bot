#!/usr/bin/env python3
"""
Forecast accuracy test.

First pass:
- One city: SATX / KSAT
- Last 30 complete local days
- Historical forecast high/low from Open-Meteo historical forecast API
- Actual observed high/low from IEM ASOS archive
- Prints miss, bias, and within-1-degree hit rate

This does NOT touch the dashboard.
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


CITY = {
    "key": "satx",
    "name": "San Antonio",
    "station": "KSAT",
    "iem_station": "SAT",
    "cli_pil": "CLIEWX",
    "cli_station_match": "SAN ANTONIO",
    "lat": 29.5337,
    "lon": -98.4698,
    "tz": "America/Chicago",
}


def fetch_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "kalshi-weather-forecast-accuracy-test/1.0"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def date_range_last_complete_days(days: int, tz_name: str) -> tuple[date, date]:
    tz = ZoneInfo(tz_name)
    today_local = datetime.now(tz).date()
    end = today_local - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return start, end


def get_forecast_daily(city: dict, start: date, end: date) -> dict[str, dict]:
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": city["tz"],
        "models": "gfs_seamless",
    }

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    txt = fetch_text(url)

    import json
    data = json.loads(txt)

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []

    by_day: dict[str, list[float]] = defaultdict(list)

    for t, temp in zip(times, temps):
        if temp is None:
            continue
        day = str(t)[:10]
        try:
            by_day[day].append(float(temp))
        except Exception:
            pass

    out: dict[str, dict] = {}
    for day, vals in by_day.items():
        if vals:
            out[day] = {
                "forecast_high": round(max(vals)),
                "forecast_low": round(min(vals)),
                "forecast_high_raw": max(vals),
                "forecast_low_raw": min(vals),
            }

    return out



def split_iem_products(text: str) -> list[str]:
    """
    IEM text retrieve returns multiple raw NWS products joined together.
    Most products contain a control-A separator or a repeated WMO header.
    This splitter is intentionally loose for CLI testing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in text.split("\x01") if p.strip()]
    if len(parts) > 1:
        return parts

    # Fallback: split before WMO climate headers if multiple products are pasted together.
    chunks = re.split(r"(?=\n[A-Z]{4}\d{2}\s+K[A-Z]{3}\s+\d{6}\nCLI[A-Z0-9]{3})", "\n" + text)
    return [c.strip() for c in chunks if c.strip()]


def parse_cli_product(product: str, city: dict) -> tuple[str | None, int | None, int | None]:
    """
    Parse one NWS CLI climate report product.

    Returns:
      report_date_iso, max_temp, min_temp

    This targets the regular daily CLI format:
      ...THE SAN ANTONIO CLIMATE SUMMARY FOR JUNE 8 2026...
      MAXIMUM         92
      MINIMUM         76
    """
    upper = product.upper()
    station_match = city.get("cli_station_match", "").upper()

    if station_match and station_match not in upper:
        return None, None, None

    date_match = re.search(
        r"CLIMATE SUMMARY FOR ([A-Z]+)\s+(\d{1,2})\s+(\d{4})",
        upper,
    )
    if not date_match:
        return None, None, None

    mon_name, day_s, year_s = date_match.groups()
    months = {
        "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
        "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
        "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    }

    month = months.get(mon_name)
    if not month:
        return None, None, None

    report_day = date(int(year_s), month, int(day_s)).isoformat()

    max_temp = None
    min_temp = None

    max_match = re.search(r"\bMAXIMUM\s+(-?\d{1,3})\b", upper)
    min_match = re.search(r"\bMINIMUM\s+(-?\d{1,3})\b", upper)

    if max_match:
        max_temp = int(max_match.group(1))
    if min_match:
        min_temp = int(min_match.group(1))

    return report_day, max_temp, min_temp


def get_actual_daily_iem(city: dict, start: date, end: date) -> dict[str, dict]:
    """
    Actual high/low from official NWS CLI text products via IEM AFOS archive.
    This should be closer to Kalshi settlement than raw ASOS temp rows.
    """
    params = {
        "pil": city["cli_pil"],
        "fmt": "text",
        "limit": "9999",
        "order": "asc",
        "sdate": f"{start.isoformat()}T00:00Z",
        "edate": f"{(end + timedelta(days=2)).isoformat()}T23:59Z",
    }

    url = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?" + urllib.parse.urlencode(params)
    txt = fetch_text(url)

    out: dict[str, dict] = {}

    for product in split_iem_products(txt):
        report_day, max_temp, min_temp = parse_cli_product(product, city)
        if not report_day:
            continue

        if start.isoformat() <= report_day <= end.isoformat():
            out[report_day] = {
                "actual_high": max_temp,
                "actual_low": min_temp,
                "obs_count": "CLI",
            }

    return out

def miss_text(forecast: int | None, actual: int | None) -> str:
    if forecast is None or actual is None:
        return "—"

    miss = forecast - actual
    if miss == 0:
        return "0°"
    if miss > 0:
        return f"+{miss}° warm"
    return f"{miss}° cool"


def summarize(rows: list[dict], kind: str) -> dict:
    misses = []

    for r in rows:
        f = r.get(f"forecast_{kind}")
        a = r.get(f"actual_{kind}")
        if f is None or a is None:
            continue
        misses.append(f - a)

    if not misses:
        return {
            "count": 0,
            "avg_abs_miss": None,
            "bias": None,
            "within_1": None,
        }

    avg_abs = statistics.mean(abs(x) for x in misses)
    bias = statistics.mean(misses)
    within_1 = sum(1 for x in misses if abs(x) <= 1) / len(misses) * 100

    return {
        "count": len(misses),
        "avg_abs_miss": avg_abs,
        "bias": bias,
        "within_1": within_1,
    }


def fmt_num(x: float | None, digits: int = 1) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def main() -> int:
    city = CITY
    start, end = date_range_last_complete_days(30, city["tz"])

    print(f"City: {city['name']} ({city['station']})")
    print(f"Range: {start} through {end}")
    print()

    print("Fetching historical forecast...")
    forecast = get_forecast_daily(city, start, end)

    print("Fetching actual observations...")
    actual = get_actual_daily_iem(city, start, end)

    rows = []

    d = start
    while d <= end:
        day = d.isoformat()
        f = forecast.get(day, {})
        a = actual.get(day, {})

        row = {
            "date": day,
            "forecast_high": f.get("forecast_high"),
            "actual_high": a.get("actual_high"),
            "forecast_low": f.get("forecast_low"),
            "actual_low": a.get("actual_low"),
            "obs_count": a.get("obs_count"),
        }
        rows.append(row)
        d += timedelta(days=1)

    print()
    print("date        fc_hi act_hi hi_miss     fc_lo act_lo lo_miss     obs")
    print("----------  ----- ------ ----------  ----- ------ ----------  ---")

    for r in rows:
        print(
            f"{r['date']}  "
            f"{str(r['forecast_high'] or '—'):>5} "
            f"{str(r['actual_high'] or '—'):>6} "
            f"{miss_text(r['forecast_high'], r['actual_high']):>10}  "
            f"{str(r['forecast_low'] or '—'):>5} "
            f"{str(r['actual_low'] or '—'):>6} "
            f"{miss_text(r['forecast_low'], r['actual_low']):>10}  "
            f"{str(r['obs_count'] or '—'):>3}"
        )

    hi = summarize(rows, "high")
    lo = summarize(rows, "low")

    print()
    print("SUMMARY")
    print("-------")
    print(
        f"HIGH: days={hi['count']} | avg miss={fmt_num(hi['avg_abs_miss'])}° "
        f"| bias={fmt_num(hi['bias'])}° | within 1°={fmt_num(hi['within_1'], 0)}%"
    )
    print(
        f"LOW : days={lo['count']} | avg miss={fmt_num(lo['avg_abs_miss'])}° "
        f"| bias={fmt_num(lo['bias'])}° | within 1°={fmt_num(lo['within_1'], 0)}%"
    )

    print()
    print("Bias meaning:")
    print("+ means forecast ran warm. - means forecast ran cool.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
