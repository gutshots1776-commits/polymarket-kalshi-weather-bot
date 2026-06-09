#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


CITY = {
    "key": "satx",
    "name": "San Antonio",
    "station": "KSAT",
    "cli_pil": "CLISAT",
    "lat": 29.5337,
    "lon": -98.4698,
    "tz": "America/Chicago",
}


MONTHS = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}


def fetch_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "kalshi-weather-forecast-accuracy-cli-test/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def last_complete_days(days: int, tz_name: str) -> tuple[date, date]:
    today = datetime.now(ZoneInfo(tz_name)).date()
    end = today - timedelta(days=1)
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
    data = json.loads(fetch_text(url))

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []

    by_day: dict[str, list[float]] = defaultdict(list)

    for t, temp in zip(times, temps):
        if temp is None:
            continue
        by_day[str(t)[:10]].append(float(temp))

    out = {}
    for day, vals in by_day.items():
        if vals:
            out[day] = {
                "forecast_high": round(max(vals)),
                "forecast_low": round(min(vals)),
            }

    return out


def get_actual_cli_daily(city: dict, start: date, end: date) -> dict[str, dict]:
    params = {
        "pil": city["cli_pil"],
        "fmt": "text",
        "limit": "9999",
        "order": "asc",
        "sdate": f"{start.isoformat()}T00:00Z",
        "edate": f"{(end + timedelta(days=2)).isoformat()}T23:59Z",
    }

    url = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?" + urllib.parse.urlencode(params)
    txt = fetch_text(url).replace("\r\n", "\n").replace("\r", "\n")

    out = {}

    pattern = re.compile(
        r"THE SAN ANTONIO CLIMATE SUMMARY FOR\s+"
        r"([A-Z]+)\s+(\d{1,2})\s+(\d{4})"
        r".*?"
        r"TEMPERATURE \(F\)"
        r".*?"
        r"YESTERDAY\s+"
        r"MAXIMUM\s+(-?\d{1,3})"
        r".*?"
        r"MINIMUM\s+(-?\d{1,3})",
        re.S,
    )

    for m in pattern.finditer(txt):
        mon_name, day_s, year_s, max_s, min_s = m.groups()
        month = MONTHS.get(mon_name)
        if not month:
            continue

        report_day = date(int(year_s), month, int(day_s)).isoformat()

        if start.isoformat() <= report_day <= end.isoformat():
            out[report_day] = {
                "actual_high": int(max_s),
                "actual_low": int(min_s),
                "obs_count": "CLI",
            }

    return out


def miss_text(forecast, actual) -> str:
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
        return {"count": 0, "avg_abs": None, "bias": None, "within_1": None}

    return {
        "count": len(misses),
        "avg_abs": statistics.mean(abs(x) for x in misses),
        "bias": statistics.mean(misses),
        "within_1": sum(1 for x in misses if abs(x) <= 1) / len(misses) * 100,
    }


def fmt(x, digits=1) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def main() -> int:
    city = CITY
    start, end = last_complete_days(30, city["tz"])

    print(f"City: {city['name']} ({city['station']})")
    print(f"Range: {start} through {end}")
    print()

    print("Fetching historical forecast...")
    forecast = get_forecast_daily(city, start, end)

    print("Fetching official CLI actuals...")
    actual = get_actual_cli_daily(city, start, end)

    print(f"Forecast days found: {len(forecast)}")
    print(f"CLI actual days found: {len(actual)}")
    print()

    rows = []
    d = start
    while d <= end:
        day = d.isoformat()
        f = forecast.get(day, {})
        a = actual.get(day, {})

        rows.append(
            {
                "date": day,
                "forecast_high": f.get("forecast_high"),
                "actual_high": a.get("actual_high"),
                "forecast_low": f.get("forecast_low"),
                "actual_low": a.get("actual_low"),
                "obs_count": a.get("obs_count"),
            }
        )
        d += timedelta(days=1)

    print("date        fc_hi act_hi hi_miss     fc_lo act_lo lo_miss     src")
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
        f"HIGH: days={hi['count']} | avg miss={fmt(hi['avg_abs'])}° "
        f"| bias={fmt(hi['bias'])}° | within 1°={fmt(hi['within_1'], 0)}%"
    )
    print(
        f"LOW : days={lo['count']} | avg miss={fmt(lo['avg_abs'])}° "
        f"| bias={fmt(lo['bias'])}° | within 1°={fmt(lo['within_1'], 0)}%"
    )

    print()
    print("Bias meaning:")
    print("+ means forecast ran warm. - means forecast ran cool.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
