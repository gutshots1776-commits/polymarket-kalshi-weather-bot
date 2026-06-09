"""Kalshi weather temperature market fetcher."""
import logging
import re
from datetime import date, datetime
from typing import Dict, List, Optional

from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present
from backend.data.weather_markets import WeatherMarket

logger = logging.getLogger("trading_bot")

# Kalshi series tickers for high-temperature markets by city
CITY_SERIES: Dict[str, str] = {
    "nyc": "KXHIGHNY",
    "chicago": "KXHIGHCHI",
    "miami": "KXHIGHMIA",
    "los_angeles": "KXHIGHLAX",
    "denver": "KXHIGHDEN",
}

CITY_NAMES: Dict[str, str] = {
    "nyc": "New York",
    "chicago": "Chicago",
    "miami": "Miami",
    "los_angeles": "Los Angeles",
    "denver": "Denver",
}

# Month abbreviation mapping for ticker parsing
MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_kalshi_ticker(ticker: str, city_key: str) -> Optional[dict]:
    """
    Parse a Kalshi bracket ticker into market parameters.

    Format: KXHIGHNY-26MAR01-B45.5
      - 26MAR01 = 2026-03-01
      - B45.5 = bracket boundary at 45.5°F (above)
      - T45.5 would be "at or below" (top boundary)
    """
    # Match: SERIES-YYMONDD-B/Tnn.n
    match = re.match(
        r'^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-([BT])([\d.]+)$',
        ticker,
    )
    if not match:
        return None

    yy = int(match.group(1))
    mon_str = match.group(2)
    dd = int(match.group(3))
    boundary_type = match.group(4)
    threshold = float(match.group(5))

    month = MONTH_ABBR.get(mon_str)
    if not month:
        return None

    year = 2000 + yy
    try:
        target_date = date(year, month, dd)
    except ValueError:
        return None

    # B = bottom boundary → "above" threshold; T = top boundary → "below" threshold
    direction = "above" if boundary_type == "B" else "below"

    return {
        "target_date": target_date,
        "threshold_f": threshold,
        "metric": "high",
        "direction": direction,
    }


def _kalshi_cents(value):
    """Convert Kalshi price fields to cents. Handles 66, "66", 0.66, or "0.66"."""
    try:
        if value is None:
            return None
        x = float(value)
        if x <= 1.0:
            x *= 100.0
        return x
    except Exception:
        return None


def _kalshi_price_bundle(m: dict) -> dict:
    """Return usable YES/NO bid/ask/last in cents from Kalshi market payload."""
    yes_bid = _kalshi_cents(m.get("yes_bid_dollars"))
    yes_ask = _kalshi_cents(m.get("yes_ask_dollars"))
    no_bid = _kalshi_cents(m.get("no_bid_dollars"))
    no_ask = _kalshi_cents(m.get("no_ask_dollars"))
    last_price = _kalshi_cents(m.get("last_price_dollars"))

    if yes_bid is None:
        yes_bid = _kalshi_cents(m.get("yes_bid"))
    if yes_ask is None:
        yes_ask = _kalshi_cents(m.get("yes_ask"))
    if no_bid is None:
        no_bid = _kalshi_cents(m.get("no_bid"))
    if no_ask is None:
        no_ask = _kalshi_cents(m.get("no_ask"))
    if last_price is None:
        last_price = _kalshi_cents(m.get("last_price"))

    # Derive opposite side when Kalshi only gives one side.
    if yes_ask is None and no_bid is not None:
        yes_ask = max(0.0, 100.0 - no_bid)
    if yes_bid is None and no_ask is not None:
        yes_bid = max(0.0, 100.0 - no_ask)
    if no_ask is None and yes_bid is not None:
        no_ask = max(0.0, 100.0 - yes_bid)
    if no_bid is None and yes_ask is not None:
        no_bid = max(0.0, 100.0 - yes_ask)

    # Last trade fallback.
    if yes_ask is None and last_price is not None:
        yes_ask = last_price
    if yes_bid is None and last_price is not None:
        yes_bid = last_price
    if no_ask is None and last_price is not None:
        no_ask = max(0.0, 100.0 - last_price)
    if no_bid is None and last_price is not None:
        no_bid = max(0.0, 100.0 - last_price)

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
    }


async def fetch_kalshi_weather_markets(
    city_keys: Optional[List[str]] = None,
) -> List[WeatherMarket]:
    """
    Fetch open weather temperature markets from Kalshi.

    Queries the KXHIGH{city} series for each configured city,
    handles cursor-based pagination, and returns WeatherMarket objects.
    """
    if not kalshi_credentials_present():
        return []

    client = KalshiClient()
    markets: List[WeatherMarket] = []
    today = date.today()

    cities = city_keys or list(CITY_SERIES.keys())

    for city_key in cities:
        series = CITY_SERIES.get(city_key)
        if not series:
            continue

        city_name = CITY_NAMES.get(city_key, city_key)
        cursor = None

        try:
            while True:
                params = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor

                data = await client.get_markets(params)
                raw_markets = data.get("markets", [])

                debug_raw = len(raw_markets)
                debug_parse_fail = 0
                debug_old_date = 0
                debug_price_missing = 0
                debug_bad_price = 0
                debug_added = 0
                debug_samples = []

                for m in raw_markets:
                    ticker = m.get("ticker", "")
                    parsed = _parse_kalshi_ticker(ticker, city_key)
                    if not parsed:
                        debug_parse_fail += 1
                        if len(debug_samples) < 5:
                            debug_samples.append(f"parse_fail ticker={ticker} title={m.get('title')}")
                        continue

                    if parsed["target_date"] < today:
                        debug_old_date += 1
                        if len(debug_samples) < 5:
                            debug_samples.append(f"old_date ticker={ticker} date={parsed['target_date']}")
                        continue

                    # Use real Kalshi prices only. Read both *_dollars and legacy cent fields.
                    px = _kalshi_price_bundle(m)

                    yes_cents = px.get("yes_ask") or px.get("yes_bid") or px.get("last_price")
                    no_cents = px.get("no_ask") or px.get("no_bid")

                    if yes_cents is None:
                        debug_price_missing += 1
                        if len(debug_samples) < 5:
                            debug_samples.append(f"price_missing ticker={ticker} keys={sorted([k for k in m.keys() if 'price' in k.lower() or 'bid' in k.lower() or 'ask' in k.lower() or 'dollar' in k.lower()])}")
                        continue
                    if no_cents is None:
                        no_cents = max(0.0, 100.0 - float(yes_cents))

                    yes_price = float(yes_cents) / 100.0
                    no_price = float(no_cents) / 100.0

                    # Skip only broken prices. Do not drop normal low/high bucket markets.
                    if yes_price <= 0 or yes_price >= 1 or no_price <= 0 or no_price >= 1:
                        debug_bad_price += 1
                        if len(debug_samples) < 5:
                            debug_samples.append(f"bad_price ticker={ticker} yes={yes_price} no={no_price}")
                        continue

                    volume = float(m.get("volume", 0) or 0)

                    debug_added += 1

                    markets.append(WeatherMarket(
                        slug=ticker,
                        market_id=ticker,
                        platform="kalshi",
                        title=m.get("title", ticker),
                        city_key=city_key,
                        city_name=city_name,
                        target_date=parsed["target_date"],
                        threshold_f=parsed["threshold_f"],
                        metric=parsed["metric"],
                        direction=parsed["direction"],
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=volume,
                    ))

                logger.info(
                    "Kalshi parse debug %s %s: raw=%s added=%s parse_fail=%s old_date=%s price_missing=%s bad_price=%s samples=%s",
                    city_key, series, debug_raw, debug_added, debug_parse_fail, debug_old_date,
                    debug_price_missing, debug_bad_price, debug_samples
                )

                # Handle pagination
                cursor = data.get("cursor")
                if not cursor or not raw_markets:
                    break

        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi markets for {city_key} ({series}): {e}")

    logger.info(f"Found {len(markets)} Kalshi weather markets")
    return markets
