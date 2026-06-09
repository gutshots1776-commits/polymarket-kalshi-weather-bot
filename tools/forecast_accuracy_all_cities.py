#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


CITIES = [
    {"key":"atl", "name":"Atlanta", "station":"KATL", "cli_pil":"CLIATL", "lat":33.6407, "lon":-84.4277, "tz":"America/New_York"},
    {"key":"aus", "name":"Austin", "station":"KAUS", "cli_pil":"CLIAUS", "lat":30.1975, "lon":-97.6664, "tz":"America/Chicago"},
    {"key":"bos", "name":"Boston", "station":"KBOS", "cli_pil":"CLIBOS", "lat":42.3656, "lon":-71.0096, "tz":"America/New_York"},
    {"key":"chicago", "name":"Chicago", "station":"KMDW", "cli_pil":"CLIMDW", "lat":41.7868, "lon":-87.7522, "tz":"America/Chicago"},
    {"key":"dal", "name":"Dallas", "station":"KDAL", "cli_pil":"CLIDAL", "lat":32.8471, "lon":-96.8518, "tz":"America/Chicago"},
    {"key":"dc", "name":"Washington DC", "station":"KDCA", "cli_pil":"CLIDCA", "lat":38.8512, "lon":-77.0402, "tz":"America/New_York"},
    {"key":"denver", "name":"Denver", "station":"KDEN", "cli_pil":"CLIDEN", "lat":39.8561, "lon":-104.6737, "tz":"America/Denver"},
    {"key":"hou", "name":"Houston", "station":"KHOU", "cli_pil":"CLIHOU", "lat":29.6454, "lon":-95.2789, "tz":"America/Chicago"},
    {"key":"los_angeles", "name":"Los Angeles", "station":"KLAX", "cli_pil":"CLILAX", "lat":33.9416, "lon":-118.4085, "tz":"America/Los_Angeles"},
    {"key":"lv", "name":"Las Vegas", "station":"KLAS", "cli_pil":"CLILAS", "lat":36.0840, "lon":-115.1537, "tz":"America/Los_Angeles"},
    {"key":"miami", "name":"Miami", "station":"KMIA", "cli_pil":"CLIMIA", "lat":25.7959, "lon":-80.2870, "tz":"America/New_York"},
    {"key":"min", "name":"Minneapolis", "station":"KMSP", "cli_pil":"CLIMSP", "lat":44.8848, "lon":-93.2223, "tz":"America/Chicago"},
    {"key":"nola", "name":"New Orleans", "station":"KMSY", "cli_pil":"CLIMSY", "lat":29.9934, "lon":-90.2580, "tz":"America/Chicago"},
    {"key":"nyc", "name":"New York City", "station":"KNYC", "cli_pil":"CLINYC", "lat":40.7794, "lon":-73.9692, "tz":"America/New_York"},
    {"key":"okc", "name":"Oklahoma City", "station":"KOKC", "cli_pil":"CLIOKC", "lat":35.3931, "lon":-97.6007, "tz":"America/Chicago"},
    {"key":"phi", "name":"Philadelphia", "station":"KPHL", "cli_pil":"CLIPHL", "lat":39.8744, "lon":-75.2424, "tz":"America/New_York"},
    {"key":"phx", "name":"Phoenix", "station":"KPHX", "cli_pil":"CLIPHX", "lat":33.4342, "lon":-112.0116, "tz":"America/Phoenix"},
    {"key":"satx", "name":"San Antonio", "station":"KSAT", "cli_pil":"CLISAT", "lat":29.5337, "lon":-98.4698, "tz":"America/Chicago"},
    {"key":"sea", "name":"Seattle", "station":"KSEA", "cli_pil":"CLISEA", "lat":47.4502, "lon":-122.3088, "tz":"America/Los_Angeles"},
    {"key":"sf", "name":"San Francisco", "station":"KSFO", "cli_pil":"CLISFO", "lat":37.6213, "lon":-122.3790, "tz":"America/Los_Angeles"},
]


MONTHS = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}


def fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "kalshi-weather-forecast-accuracy-all-cities/1.0"},
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
        r"THE .*? CLIMATE SUMMARY FOR\s+"
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
            # If there are multiple CLI reports for one date, order=asc means this keeps the latest one.
            out[report_day] = {
                "actual_high": int(max_s),
                "actual_low": int(min_s),
                "source": "CLI",
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


def build_rows(city: dict, days: int = 30) -> tuple[list[dict], dict, dict]:
    start, end = last_complete_days(days, city["tz"])

    forecast = get_forecast_daily(city, start, end)
    time.sleep(0.2)
    actual = get_actual_cli_daily(city, start, end)

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
                "source": a.get("source"),
            }
        )
        d += timedelta(days=1)

    return rows, summarize(rows, "high"), summarize(rows, "low")


def yesterday_line(rows: list[dict], kind: str) -> str:
    if not rows:
        return "yesterday n/a"

    r = rows[-1]
    f = r.get(f"forecast_{kind}")
    a = r.get(f"actual_{kind}")

    if f is None or a is None:
        return "yesterday n/a"

    return f"yesterday fc {f}° / actual {a}° / {miss_text(f, a)}"


def main() -> int:
    print("Forecast Accuracy — all 20 cities")
    print("Source: Open-Meteo historical forecast vs IEM archived NWS CLI")
    print()

    results = []

    for city in CITIES:
        try:
            print(f"Fetching {city['name']} ({city['station']})...", flush=True)
            rows, hi, lo = build_rows(city, days=30)

            results.append((city, rows, hi, lo))

        except Exception as e:
            print(f"ERROR {city['name']} ({city['station']}): {e!r}", flush=True)
            results.append((city, [], {"count": 0, "avg_abs": None, "bias": None, "within_1": None}, {"count": 0, "avg_abs": None, "bias": None, "within_1": None}))

    print()
    print("SUMMARY")
    print("=======")

    for city, rows, hi, lo in results:
        print()
        print(f"{city['name']} ({city['station']})")
        print(
            f"  HIGH: days={hi['count']:>2} | avg miss={fmt(hi['avg_abs'])}° "
            f"| bias={fmt(hi['bias'])}° | within 1°={fmt(hi['within_1'], 0)}% "
            f"| {yesterday_line(rows, 'high')}"
        )
        print(
            f"  LOW : days={lo['count']:>2} | avg miss={fmt(lo['avg_abs'])}° "
            f"| bias={fmt(lo['bias'])}° | within 1°={fmt(lo['within_1'], 0)}% "
            f"| {yesterday_line(rows, 'low')}"
        )

    print()
    print("Bias meaning: + means forecast ran warm. - means forecast ran cool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
