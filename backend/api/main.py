"""FastAPI backend for BTC 5-min trading bot dashboard."""
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import asyncio
import json
import os

from backend.config import settings
from backend.models.database import (
    get_db, init_db, SessionLocal,
    Signal, Trade, BotState, AILog, ScanLog
)
from backend.core.signals import scan_for_signals, TradingSignal
from backend.data.btc_markets import fetch_active_btc_markets, BtcMarket
from backend.data.crypto import fetch_crypto_price, compute_btc_microstructure

from pydantic import BaseModel

app = FastAPI(
    title="BTC 5-Min Trading Bot",
    description="Polymarket BTC Up/Down 5-minute market trading bot",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()


# Pydantic response models
class BtcPriceResponse(BaseModel):
    price: float
    change_24h: float
    change_7d: float
    market_cap: float
    volume_24h: float
    last_updated: datetime


class BtcWindowResponse(BaseModel):
    slug: str
    market_id: str
    up_price: float
    down_price: float
    window_start: datetime
    window_end: datetime
    volume: float
    is_active: bool
    is_upcoming: bool
    time_until_end: float
    spread: float


class MicrostructureResponse(BaseModel):
    rsi: float = 50.0
    momentum_1m: float = 0.0
    momentum_5m: float = 0.0
    momentum_15m: float = 0.0
    vwap_deviation: float = 0.0
    sma_crossover: float = 0.0
    volatility: float = 0.0
    price: float = 0.0
    source: str = "unknown"


class SignalResponse(BaseModel):
    market_ticker: str
    market_title: str
    platform: str
    direction: str
    model_probability: float
    market_probability: float
    edge: float
    confidence: float
    suggested_size: float
    reasoning: str
    timestamp: datetime
    category: str = "crypto"
    event_slug: Optional[str] = None
    btc_price: float = 0.0
    btc_change_24h: float = 0.0
    window_end: Optional[datetime] = None
    actionable: bool = False


class TradeResponse(BaseModel):
    id: int
    market_ticker: str
    platform: str
    event_slug: Optional[str] = None
    direction: str
    entry_price: float
    size: float
    timestamp: datetime
    settled: bool
    result: str
    pnl: Optional[float]


class BotStats(BaseModel):
    bankroll: float
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    is_running: bool
    last_run: Optional[datetime]


class CalibrationBucket(BaseModel):
    bucket: str
    predicted_avg: float
    actual_rate: float
    count: int


class CalibrationSummary(BaseModel):
    total_signals: int
    total_with_outcome: int
    accuracy: float
    avg_predicted_edge: float
    avg_actual_edge: float
    brier_score: float


class WeatherForecastResponse(BaseModel):
    city_key: str
    city_name: str
    target_date: str
    mean_high: float
    std_high: float
    mean_low: float
    std_low: float
    num_members: int
    ensemble_agreement: float


class WeatherMarketResponse(BaseModel):
    slug: str
    market_id: str
    platform: str = "polymarket"
    title: str
    city_key: str
    city_name: str
    target_date: str
    threshold_f: float
    metric: str
    direction: str
    yes_price: float
    no_price: float
    volume: float


class WeatherSignalResponse(BaseModel):
    market_id: str
    city_key: str
    city_name: str
    target_date: str
    threshold_f: float
    metric: str
    direction: str
    model_probability: float
    market_probability: float
    edge: float
    confidence: float
    suggested_size: float
    reasoning: str
    ensemble_mean: float
    ensemble_std: float
    ensemble_members: int
    actionable: bool = False


class DashboardData(BaseModel):
    stats: BotStats
    btc_price: Optional[BtcPriceResponse]
    microstructure: Optional[MicrostructureResponse] = None
    windows: List[BtcWindowResponse]
    active_signals: List[SignalResponse]
    recent_trades: List[TradeResponse]
    equity_curve: List[dict]
    calibration: Optional[CalibrationSummary] = None
    weather_signals: List[WeatherSignalResponse] = []
    weather_forecasts: List[WeatherForecastResponse] = []


class EventResponse(BaseModel):
    timestamp: str
    type: str
    message: str
    data: dict = {}


# Startup / Shutdown
@app.on_event("startup")
async def startup():
    print("=" * 60)
    print("BTC 5-MIN TRADING BOT v3.0")
    print("=" * 60)
    print("Initializing database...")

    init_db()

    db = SessionLocal()
    try:
        state = db.query(BotState).first()
        if not state:
            state = BotState(
                bankroll=settings.INITIAL_BANKROLL,
                total_trades=0,
                winning_trades=0,
                total_pnl=0.0,
                is_running=True
            )
            db.add(state)
            db.commit()
            print(f"Created new bot state with ${settings.INITIAL_BANKROLL:,.2f} bankroll")
        else:
            state.is_running = True
            db.commit()
            print(f"Loaded bot state: Bankroll ${state.bankroll:,.2f}, P&L ${state.total_pnl:+,.2f}, {state.total_trades} trades")
    finally:
        db.close()

    print("")
    print("Configuration:")
    print(f"  - Simulation mode: {settings.SIMULATION_MODE}")
    print(f"  - Min edge threshold: {settings.MIN_EDGE_THRESHOLD:.0%}")
    print(f"  - Kelly fraction: {settings.KELLY_FRACTION:.0%}")
    print(f"  - Scan interval: {settings.SCAN_INTERVAL_SECONDS}s")
    print(f"  - Settlement interval: {settings.SETTLEMENT_INTERVAL_SECONDS}s")
    print("")

    from backend.core.scheduler import start_scheduler, log_event
    start_scheduler()
    log_event("success", "BTC 5-min trading bot initialized")

    print("Bot is now running!")
    print(f"  - BTC scan: every {settings.SCAN_INTERVAL_SECONDS}s (edge >= {settings.MIN_EDGE_THRESHOLD:.0%})")
    print(f"  - Settlement check: every {settings.SETTLEMENT_INTERVAL_SECONDS}s")
    if settings.WEATHER_ENABLED:
        print(f"  - Weather scan: every {settings.WEATHER_SCAN_INTERVAL_SECONDS}s (edge >= {settings.WEATHER_MIN_EDGE_THRESHOLD:.0%})")
        print(f"  - Weather cities: {settings.WEATHER_CITIES}")
    else:
        print("  - Weather trading: DISABLED")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    from backend.core.scheduler import stop_scheduler
    stop_scheduler()


# Core endpoints
@app.get("/")
async def root():
    return {"status": "ok", "message": "BTC 5-Min Trading Bot API v3.0", "simulation_mode": settings.SIMULATION_MODE}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/stats", response_model=BotStats)
async def get_stats(db: Session = Depends(get_db)):
    state = db.query(BotState).first()
    if not state:
        raise HTTPException(status_code=404, detail="Bot state not initialized")

    win_rate = state.winning_trades / state.total_trades if state.total_trades > 0 else 0

    return BotStats(
        bankroll=state.bankroll,
        total_trades=state.total_trades,
        winning_trades=state.winning_trades,
        win_rate=win_rate,
        total_pnl=state.total_pnl,
        is_running=state.is_running,
        last_run=state.last_run
    )


# BTC-specific endpoints
@app.get("/api/btc/price", response_model=Optional[BtcPriceResponse])
async def get_btc_price():
    """Get current BTC price and momentum data."""
    try:
        btc = await fetch_crypto_price("BTC")
        if not btc:
            return None

        return BtcPriceResponse(
            price=btc.current_price,
            change_24h=btc.change_24h,
            change_7d=btc.change_7d,
            market_cap=btc.market_cap,
            volume_24h=btc.volume_24h,
            last_updated=btc.last_updated
        )
    except Exception:
        return None


@app.get("/api/btc/windows", response_model=List[BtcWindowResponse])
async def get_btc_windows():
    """Get upcoming BTC 5-min windows with prices."""
    try:
        markets = await fetch_active_btc_markets()
        return [
            BtcWindowResponse(
                slug=m.slug,
                market_id=m.market_id,
                up_price=m.up_price,
                down_price=m.down_price,
                window_start=m.window_start,
                window_end=m.window_end,
                volume=m.volume,
                is_active=m.is_active,
                is_upcoming=m.is_upcoming,
                time_until_end=m.time_until_end,
                spread=m.spread,
            )
            for m in markets
        ]
    except Exception:
        return []


@app.get("/api/signals", response_model=List[SignalResponse])
async def get_signals():
    """Get current BTC trading signals."""
    try:
        signals = await scan_for_signals()
        return [_signal_to_response(s) for s in signals]
    except Exception:
        return []


@app.get("/api/signals/actionable", response_model=List[SignalResponse])
async def get_actionable_signals():
    """Get only signals that pass the edge threshold."""
    try:
        signals = await scan_for_signals()
        actionable = [s for s in signals if s.passes_threshold]
        return [_signal_to_response(s) for s in actionable]
    except Exception:
        return []


def _signal_to_response(s: TradingSignal, actionable: bool = False) -> SignalResponse:
    return SignalResponse(
        market_ticker=s.market.market_id,
        market_title=f"BTC 5m - {s.market.slug}",
        platform="polymarket",
        direction=s.direction,
        model_probability=s.model_probability,
        market_probability=s.market_probability,
        edge=s.edge,
        confidence=s.confidence,
        suggested_size=s.suggested_size,
        reasoning=s.reasoning,
        timestamp=s.timestamp,
        category="crypto",
        event_slug=s.market.slug,
        btc_price=s.btc_price,
        btc_change_24h=s.btc_change_24h,
        window_end=s.market.window_end,
        actionable=actionable,
    )


@app.get("/api/trades", response_model=List[TradeResponse])
async def get_trades(
    limit: int = 50,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Trade)
    if status:
        query = query.filter(Trade.result == status)
    trades = query.order_by(Trade.timestamp.desc()).limit(limit).all()

    return [
        TradeResponse(
            id=t.id,
            market_ticker=t.market_ticker,
            platform=t.platform,
            event_slug=t.event_slug,
            direction=t.direction,
            entry_price=t.entry_price,
            size=t.size,
            timestamp=t.timestamp,
            settled=t.settled,
            result=t.result,
            pnl=t.pnl
        )
        for t in trades
    ]


@app.get("/api/equity-curve")
async def get_equity_curve(db: Session = Depends(get_db)):
    trades = db.query(Trade).filter(Trade.settled == True).order_by(Trade.timestamp).all()

    curve = []
    cumulative_pnl = 0
    bankroll = settings.INITIAL_BANKROLL

    for trade in trades:
        if trade.pnl is not None:
            cumulative_pnl += trade.pnl
            curve.append({
                "timestamp": trade.timestamp.isoformat(),
                "pnl": cumulative_pnl,
                "bankroll": bankroll + cumulative_pnl,
                "trade_id": trade.id
            })

    return curve


@app.post("/api/simulate-trade")
async def simulate_trade(signal_ticker: str, db: Session = Depends(get_db)):
    from backend.core.scheduler import log_event

    signals = await scan_for_signals()
    signal = next((s for s in signals if s.market.market_id == signal_ticker), None)

    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    state = db.query(BotState).first()
    if not state:
        raise HTTPException(status_code=500, detail="Bot state not initialized")

    entry_price = signal.market.up_price if signal.direction == "up" else signal.market.down_price

    trade = Trade(
        market_ticker=signal.market.market_id,
        platform="polymarket",
        event_slug=signal.market.slug,
        direction=signal.direction,
        entry_price=entry_price,
        size=min(signal.suggested_size, state.bankroll * 0.05),
        model_probability=signal.model_probability,
        market_price_at_entry=signal.market_probability,
        edge_at_entry=signal.edge
    )

    db.add(trade)
    state.total_trades += 1
    db.commit()

    log_event("trade", f"Manual BTC trade: {signal.direction.upper()} {signal.market.slug}")
    return {"status": "ok", "trade_id": trade.id, "size": trade.size}


@app.post("/api/run-scan")
async def run_scan(db: Session = Depends(get_db)):
    from backend.core.scheduler import run_manual_scan, log_event

    state = db.query(BotState).first()
    if state:
        state.last_run = datetime.utcnow()
        db.commit()

    log_event("info", "Manual scan triggered (BTC + Weather)")
    await run_manual_scan()

    signals = await scan_for_signals()
    actionable = [s for s in signals if s.passes_threshold]

    result = {
        "status": "ok",
        "total_signals": len(signals),
        "actionable_signals": len(actionable),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Also run weather scan if enabled
    if settings.WEATHER_ENABLED:
        try:
            from backend.core.weather_signals import scan_for_weather_signals
            wx_signals = await scan_for_weather_signals()
            wx_actionable = [s for s in wx_signals if s.passes_threshold]
            result["weather_signals"] = len(wx_signals)
            result["weather_actionable"] = len(wx_actionable)
        except Exception:
            result["weather_signals"] = 0
            result["weather_actionable"] = 0

    return result


@app.post("/api/settle-trades")
async def settle_trades_endpoint(db: Session = Depends(get_db)):
    from backend.core.settlement import settle_pending_trades, update_bot_state_with_settlements
    from backend.core.scheduler import log_event

    log_event("info", "Manual settlement triggered")

    settled = await settle_pending_trades(db)
    await update_bot_state_with_settlements(db, settled)

    return {
        "status": "ok",
        "settled_count": len(settled),
        "trades": [{"id": t.id, "result": t.result, "pnl": t.pnl} for t in settled]
    }


def _compute_calibration_summary(db: Session) -> Optional[CalibrationSummary]:
    """Compute calibration summary from settled signals."""
    total_signals = db.query(Signal).count()
    settled_signals = db.query(Signal).filter(Signal.outcome_correct.isnot(None)).all()

    if not settled_signals:
        if total_signals == 0:
            return None
        return CalibrationSummary(
            total_signals=total_signals,
            total_with_outcome=0,
            accuracy=0.0,
            avg_predicted_edge=0.0,
            avg_actual_edge=0.0,
            brier_score=0.0,
        )

    total_with_outcome = len(settled_signals)
    correct = sum(1 for s in settled_signals if s.outcome_correct)
    accuracy = correct / total_with_outcome if total_with_outcome > 0 else 0.0

    avg_predicted_edge = sum(abs(s.edge) for s in settled_signals) / total_with_outcome
    # Actual edge: for correct predictions, edge was real; for incorrect, edge was negative
    avg_actual_edge = sum(
        abs(s.edge) if s.outcome_correct else -abs(s.edge)
        for s in settled_signals
    ) / total_with_outcome

    # Brier score: mean squared error of probability forecasts
    # For each signal: (predicted_prob - actual_outcome)^2
    brier_sum = 0.0
    for s in settled_signals:
        # Model probability is for UP; actual is 1.0 if UP won, 0.0 if DOWN won
        actual = s.settlement_value if s.settlement_value is not None else 0.5
        brier_sum += (s.model_probability - actual) ** 2
    brier_score = brier_sum / total_with_outcome

    return CalibrationSummary(
        total_signals=total_signals,
        total_with_outcome=total_with_outcome,
        accuracy=accuracy,
        avg_predicted_edge=avg_predicted_edge,
        avg_actual_edge=avg_actual_edge,
        brier_score=brier_score,
    )


@app.get("/api/calibration")
async def get_calibration(db: Session = Depends(get_db)):
    """Return calibration data: predicted probability vs actual win rate."""
    signals = db.query(Signal).filter(Signal.outcome_correct.isnot(None)).all()

    if not signals:
        return {"buckets": [], "summary": None}

    # Bucket signals by model_probability into 5% bins
    from collections import defaultdict
    buckets_data = defaultdict(lambda: {"predicted_sum": 0.0, "correct": 0, "total": 0})

    for s in signals:
        # Bin by 5% increments
        bin_start = int(s.model_probability * 100 // 5) * 5
        bin_end = bin_start + 5
        bucket_key = f"{bin_start}-{bin_end}%"

        buckets_data[bucket_key]["predicted_sum"] += s.model_probability
        buckets_data[bucket_key]["total"] += 1
        if s.outcome_correct:
            buckets_data[bucket_key]["correct"] += 1

    buckets = []
    for bucket_key in sorted(buckets_data.keys()):
        d = buckets_data[bucket_key]
        buckets.append(CalibrationBucket(
            bucket=bucket_key,
            predicted_avg=d["predicted_sum"] / d["total"],
            actual_rate=d["correct"] / d["total"],
            count=d["total"],
        ))

    summary = _compute_calibration_summary(db)

    return {"buckets": buckets, "summary": summary}


# Kalshi endpoints
@app.get("/api/kalshi/status")
async def get_kalshi_status():
    """Test Kalshi API authentication and return connection status."""
    from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present

    if not kalshi_credentials_present():
        return {
            "connected": False,
            "error": "Kalshi credentials not configured (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH)",
        }

    try:
        client = KalshiClient()
        balance_data = await client.get_balance()
        return {
            "connected": True,
            "balance": balance_data,
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
        }


# Weather endpoints
@app.get("/api/weather/forecasts", response_model=List[WeatherForecastResponse])
async def get_weather_forecasts():
    """Get ensemble forecasts for configured cities."""
    if not settings.WEATHER_ENABLED:
        return []

    try:
        from backend.data.weather import fetch_ensemble_forecast, CITY_CONFIG
        from datetime import date

        city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]
        forecasts = []

        for city_key in city_keys:
            if city_key not in CITY_CONFIG:
                continue
            forecast = await fetch_ensemble_forecast(city_key)
            if forecast:
                forecasts.append(WeatherForecastResponse(
                    city_key=forecast.city_key,
                    city_name=forecast.city_name,
                    target_date=forecast.target_date.isoformat(),
                    mean_high=forecast.mean_high,
                    std_high=forecast.std_high,
                    mean_low=forecast.mean_low,
                    std_low=forecast.std_low,
                    num_members=forecast.num_members,
                    ensemble_agreement=forecast.ensemble_agreement,
                ))

        return forecasts
    except Exception:
        return []


@app.get("/api/weather/markets", response_model=List[WeatherMarketResponse])
async def get_weather_markets():
    """Get active weather temperature markets."""
    if not settings.WEATHER_ENABLED:
        return []

    try:
        from backend.data.weather_markets import fetch_polymarket_weather_markets

        city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]
        markets = await fetch_polymarket_weather_markets(city_keys)

        # Also fetch Kalshi markets if enabled
        if settings.KALSHI_ENABLED:
            try:
                from backend.data.kalshi_client import kalshi_credentials_present
                from backend.data.kalshi_markets import fetch_kalshi_weather_markets
                if kalshi_credentials_present():
                    kalshi_markets = await fetch_kalshi_weather_markets(city_keys)
                    markets.extend(kalshi_markets)
            except Exception:
                pass

        return [
            WeatherMarketResponse(
                slug=m.slug,
                market_id=m.market_id,
                platform=m.platform,
                title=m.title,
                city_key=m.city_key,
                city_name=m.city_name,
                target_date=m.target_date.isoformat(),
                threshold_f=m.threshold_f,
                metric=m.metric,
                direction=m.direction,
                yes_price=m.yes_price,
                no_price=m.no_price,
                volume=m.volume,
            )
            for m in markets
        ]
    except Exception:
        return []


@app.get("/api/weather/signals", response_model=List[WeatherSignalResponse])
async def get_weather_signals():
    """Get current weather trading signals."""
    if not settings.WEATHER_ENABLED:
        return []

    try:
        from backend.core.weather_signals import scan_for_weather_signals

        signals = await scan_for_weather_signals()
        return [_weather_signal_to_response(s) for s in signals]
    except Exception:
        return []


def _weather_signal_to_response(s) -> WeatherSignalResponse:
    return WeatherSignalResponse(
        market_id=s.market.market_id,
        city_key=s.market.city_key,
        city_name=s.market.city_name,
        target_date=s.market.target_date.isoformat(),
        threshold_f=s.market.threshold_f,
        metric=s.market.metric,
        direction=s.direction,
        model_probability=s.model_probability,
        market_probability=s.market_probability,
        edge=s.edge,
        confidence=s.confidence,
        suggested_size=s.suggested_size,
        reasoning=s.reasoning,
        ensemble_mean=s.ensemble_mean,
        ensemble_std=s.ensemble_std,
        ensemble_members=s.ensemble_members,
        actionable=s.passes_threshold,
    )


@app.get("/api/events", response_model=List[EventResponse])
async def get_events(limit: int = 50):
    from backend.core.scheduler import get_recent_events
    events = get_recent_events(limit)
    return [
        EventResponse(
            timestamp=e["timestamp"],
            type=e["type"],
            message=e["message"],
            data=e.get("data", {})
        )
        for e in events
    ]


# Bot control
@app.post("/api/bot/start")
async def start_bot(db: Session = Depends(get_db)):
    from backend.core.scheduler import start_scheduler, log_event, is_scheduler_running

    state = db.query(BotState).first()
    if state:
        state.is_running = True
        db.commit()

    if not is_scheduler_running():
        start_scheduler()

    log_event("success", "Trading bot started")
    return {"status": "started", "is_running": True}


@app.post("/api/bot/stop")
async def stop_bot(db: Session = Depends(get_db)):
    from backend.core.scheduler import log_event

    state = db.query(BotState).first()
    if state:
        state.is_running = False
        db.commit()

    log_event("info", "Trading bot paused")
    return {"status": "stopped", "is_running": False}


@app.post("/api/bot/reset")
async def reset_bot(db: Session = Depends(get_db)):
    from backend.core.scheduler import log_event

    try:
        trades_deleted = db.query(Trade).delete()
        state = db.query(BotState).first()
        if state:
            state.bankroll = settings.INITIAL_BANKROLL
            state.total_trades = 0
            state.winning_trades = 0
            state.total_pnl = 0.0
            state.is_running = True

        ai_logs_deleted = db.query(AILog).delete()
        db.commit()

        log_event("success", f"Bot reset: {trades_deleted} trades deleted. Fresh start with ${settings.INITIAL_BANKROLL:,.2f}")

        return {
            "status": "reset",
            "trades_deleted": trades_deleted,
            "ai_logs_deleted": ai_logs_deleted,
            "new_bankroll": settings.INITIAL_BANKROLL
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.get("/api/dashboard", response_model=DashboardData)
async def get_dashboard(db: Session = Depends(get_db)):
    """Get all dashboard data in one call."""
    stats = await get_stats(db)

    # Fetch BTC price from microstructure first, fallback to CoinGecko
    btc_price_data = None
    micro_data = None
    try:
        micro = await compute_btc_microstructure()
        if micro:
            micro_data = MicrostructureResponse(
                rsi=micro.rsi,
                momentum_1m=micro.momentum_1m,
                momentum_5m=micro.momentum_5m,
                momentum_15m=micro.momentum_15m,
                vwap_deviation=micro.vwap_deviation,
                sma_crossover=micro.sma_crossover,
                volatility=micro.volatility,
                price=micro.price,
                source=micro.source,
            )
            btc_price_data = BtcPriceResponse(
                price=micro.price,
                change_24h=micro.momentum_15m * 96,  # rough extrapolation
                change_7d=0,
                market_cap=0,
                volume_24h=0,
                last_updated=datetime.utcnow(),
            )
    except Exception:
        pass
    if not btc_price_data:
        try:
            btc = await fetch_crypto_price("BTC")
            if btc:
                btc_price_data = BtcPriceResponse(
                    price=btc.current_price,
                    change_24h=btc.change_24h,
                    change_7d=btc.change_7d,
                    market_cap=btc.market_cap,
                    volume_24h=btc.volume_24h,
                    last_updated=btc.last_updated
                )
        except Exception:
            pass

    # Fetch windows
    windows = []
    try:
        markets = await fetch_active_btc_markets()
        windows = [
            BtcWindowResponse(
                slug=m.slug,
                market_id=m.market_id,
                up_price=m.up_price,
                down_price=m.down_price,
                window_start=m.window_start,
                window_end=m.window_end,
                volume=m.volume,
                is_active=m.is_active,
                is_upcoming=m.is_upcoming,
                time_until_end=m.time_until_end,
                spread=m.spread,
            )
            for m in markets
        ]
    except Exception:
        pass

    # Signals — return ALL signals, mark which are actionable
    signals = []
    try:
        raw_signals = await scan_for_signals()
        signals = [_signal_to_response(s, actionable=s.passes_threshold) for s in raw_signals]
    except Exception:
        pass

    # Recent trades
    trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(50).all()
    recent_trades = [
        TradeResponse(
            id=t.id,
            market_ticker=t.market_ticker,
            platform=t.platform,
            event_slug=t.event_slug,
            direction=t.direction,
            entry_price=t.entry_price,
            size=t.size,
            timestamp=t.timestamp,
            settled=t.settled,
            result=t.result,
            pnl=t.pnl
        )
        for t in trades
    ]

    # Equity curve
    equity_trades = db.query(Trade).filter(Trade.settled == True).order_by(Trade.timestamp).all()
    equity_curve = []
    cumulative_pnl = 0
    for trade in equity_trades:
        if trade.pnl is not None:
            cumulative_pnl += trade.pnl
            equity_curve.append({
                "timestamp": trade.timestamp.isoformat(),
                "pnl": cumulative_pnl,
                "bankroll": settings.INITIAL_BANKROLL + cumulative_pnl
            })

    # Calibration summary
    calibration = _compute_calibration_summary(db)

    # Weather data (if enabled)
    weather_signals_data = []
    weather_forecasts_data = []
    if settings.WEATHER_ENABLED:
        try:
            from backend.core.weather_signals import scan_for_weather_signals
            from backend.data.weather import fetch_ensemble_forecast, CITY_CONFIG

            wx_signals = await scan_for_weather_signals()
            weather_signals_data = [_weather_signal_to_response(s) for s in wx_signals]

            city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]
            for city_key in city_keys:
                if city_key not in CITY_CONFIG:
                    continue
                forecast = await fetch_ensemble_forecast(city_key)
                if forecast:
                    weather_forecasts_data.append(WeatherForecastResponse(
                        city_key=forecast.city_key,
                        city_name=forecast.city_name,
                        target_date=forecast.target_date.isoformat(),
                        mean_high=forecast.mean_high,
                        std_high=forecast.std_high,
                        mean_low=forecast.mean_low,
                        std_low=forecast.std_low,
                        num_members=forecast.num_members,
                        ensemble_agreement=forecast.ensemble_agreement,
                    ))
        except Exception:
            pass

    return DashboardData(
        stats=stats,
        btc_price=btc_price_data,
        microstructure=micro_data,
        windows=windows,
        active_signals=signals,
        recent_trades=recent_trades,
        equity_curve=equity_curve,
        calibration=calibration,
        weather_signals=weather_signals_data,
        weather_forecasts=weather_forecasts_data,
    )


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)

    try:
        await websocket.send_json({
            "timestamp": datetime.utcnow().isoformat(),
            "type": "success",
            "message": "Connected to BTC trading bot"
        })

        from backend.core.scheduler import get_recent_events
        for event in get_recent_events(20):
            await websocket.send_json(event)

        last_event_count = len(get_recent_events(200))
        while True:
            await asyncio.sleep(2)

            current_events = get_recent_events(200)
            if len(current_events) > last_event_count:
                new_events = current_events[last_event_count - len(current_events):]
                for event in new_events:
                    await websocket.send_json(event)
                last_event_count = len(current_events)

            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

CURRENT_TEMPS_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Kalshi WX Current Temps</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #111827;
      --card2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --border: #334155;
    }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 16px 10px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(8px);
      z-index: 10;
    }
    h1 {
      margin: 0;
      font-size: 22px;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
      line-height: 1.35;
    }
    .controls {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    button, select {
      background: var(--card2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
    }
    main {
      padding: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 10px 20px rgba(0,0,0,.18);
    }
    .code {
      font-size: 13px;
      color: var(--muted);
      letter-spacing: .06em;
      font-weight: 700;
    }
    .name {
      font-size: 15px;
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .temp {
      font-size: 34px;
      font-weight: 800;
      margin-top: 10px;
    }
    .time {
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
      min-height: 16px;
    }
    .error {
      color: var(--bad);
      font-size: 12px;
      margin-top: 8px;
    }
    a {
      color: #93c5fd;
      font-size: 12px;
      text-decoration: none;
      display: inline-block;
      margin-top: 10px;
    }
    .footer {
      color: var(--muted);
      font-size: 12px;
      margin-top: 14px;
      line-height: 1.4;
    }
  </style>
</head>
<body>
<header>
  <h1>Kalshi WX Current Temps</h1>
  <div class="sub">
    Current Open-Meteo grid temperatures for the 20-city Kalshi watchlist.
    This is not official METAR/DSM/CLI settlement data.
  </div>
  <div class="controls">
    <button onclick="loadTemps(true)">Refresh now</button>
    <select id="sortMode" onchange="renderCards()">
      <option value="default">Sort: Default</option>
      <option value="temp_desc">Sort: Hottest first</option>
      <option value="temp_asc">Sort: Coldest first</option>
      <option value="code">Sort: City code</option>
    </select>
    <span class="sub" id="status">Loading...</span>
  </div>
</header>

<main>
  <div class="grid" id="grid"></div>
  <div class="footer">
    Auto-refreshes every 60 minutes while open. Opening or refreshing the page fetches fresh data.
  </div>
</main>

<script>
const CITIES = [
  {code:"DEN", name:"Denver", lat:39.8561, lon:-104.6737},
  {code:"NYC", name:"New York City", lat:40.7828, lon:-73.9653},
  {code:"PHI", name:"Philadelphia", lat:39.8719, lon:-75.2411},
  {code:"CHI", name:"Chicago", lat:41.7868, lon:-87.7522},
  {code:"LA", name:"Los Angeles", lat:33.9425, lon:-118.4081},
  {code:"MIA", name:"Miami", lat:25.7959, lon:-80.2870},
  {code:"SF", name:"San Francisco", lat:37.6213, lon:-122.3790},
  {code:"SEA", name:"Seattle", lat:47.4502, lon:-122.3088},
  {code:"ATL", name:"Atlanta", lat:33.6407, lon:-84.4277},
  {code:"AUS", name:"Austin", lat:30.1975, lon:-97.6664},
  {code:"BOS", name:"Boston", lat:42.3656, lon:-71.0096},
  {code:"DAL", name:"Dallas", lat:32.8471, lon:-96.8518},
  {code:"DC", name:"Washington DC", lat:38.8512, lon:-77.0402},
  {code:"HOU", name:"Houston", lat:29.6454, lon:-95.2789},
  {code:"LV", name:"Las Vegas", lat:36.0801, lon:-115.1522},
  {code:"MIN", name:"Minneapolis", lat:44.8848, lon:-93.2223},
  {code:"NOLA", name:"New Orleans", lat:29.9934, lon:-90.2580},
  {code:"OKC", name:"Oklahoma City", lat:35.3931, lon:-97.6007},
  {code:"PHX", name:"Phoenix", lat:33.4342, lon:-112.0116},
  {code:"SATX", name:"San Antonio", lat:29.5337, lon:-98.4698}
];

let rows = CITIES.map(c => ({...c, temp: null, time: "", error: ""}));

function urlFor(c) {
  return `https://api.open-meteo.com/v1/forecast?latitude=${c.lat}&longitude=${c.lon}&current=temperature_2m&temperature_unit=fahrenheit&timezone=auto`;
}

function fmtTemp(t) {
  return (t === null || t === undefined || Number.isNaN(t)) ? "—" : `${Math.round(t)}°`;
}

function renderCards() {
  const mode = document.getElementById("sortMode").value;
  let sorted = [...rows];

  if (mode === "temp_desc") sorted.sort((a,b) => (b.temp ?? -999) - (a.temp ?? -999));
  if (mode === "temp_asc") sorted.sort((a,b) => (a.temp ?? 999) - (b.temp ?? 999));
  if (mode === "code") sorted.sort((a,b) => a.code.localeCompare(b.code));

  document.getElementById("grid").innerHTML = sorted.map(r => `
    <div class="card">
      <div class="code">${r.code}</div>
      <div class="name">${r.name}</div>
      <div class="temp">${fmtTemp(r.temp)}</div>
      <div class="time">${r.time ? "Obs/API time: " + r.time : ""}</div>
      ${r.error ? `<div class="error">${r.error}</div>` : ""}
      <a href="${urlFor(r)}" target="_blank" rel="noopener">Open raw Open-Meteo data</a>
    </div>
  `).join("");
}

async function fetchCity(c) {
  const res = await fetch(urlFor(c), {cache: "no-store"});
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return {
    ...c,
    temp: data?.current?.temperature_2m ?? null,
    time: data?.current?.time ?? "",
    error: ""
  };
}

async function loadTemps(manual=false) {
  const status = document.getElementById("status");
  status.textContent = manual ? "Refreshing..." : "Loading...";
  renderCards();

  const results = await Promise.all(CITIES.map(async c => {
    try {
      return await fetchCity(c);
    } catch (e) {
      return {...c, temp: null, time: "", error: String(e.message || e)};
    }
  }));

  rows = results;
  renderCards();

  const now = new Date();
  status.textContent = `Last refresh: ${now.toLocaleTimeString()} · Next auto-refresh in 60 min`;
}

loadTemps();
setInterval(() => loadTemps(false), 60 * 60 * 1000);
</script>
</body>
</html>
"""

@app.get("/current-temps", response_class=HTMLResponse)
async def current_temps_dashboard():
    return HTMLResponse(CURRENT_TEMPS_DASHBOARD_HTML)

ENSEMBLE_TEMPS_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Kalshi WX Ensemble + Observed Temps</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #111827;
      --card2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --border: #334155;
      --bad: #ef4444;
      --good: #22c55e;
      --warn: #f59e0b;
    }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 16px 10px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      background: rgba(15, 23, 42, 0.96);
      backdrop-filter: blur(8px);
      z-index: 10;
    }
    h1 { margin: 0; font-size: 22px; }
    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
      line-height: 1.35;
    }
    .controls {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    button, select {
      background: var(--card2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 14px;
    }
    button { cursor: pointer; }
    main { padding: 14px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 10px 20px rgba(0,0,0,.18);
    }
    .code {
      font-size: 13px;
      color: var(--muted);
      letter-spacing: .06em;
      font-weight: 700;
    }
    .name { font-size: 15px; margin-top: 2px; }
    .temp {
      font-size: 34px;
      font-weight: 800;
      margin-top: 10px;
    }
    .label {
      font-size: 12px;
      color: var(--muted);
      margin-top: -4px;
    }
    .compare {
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 9px 0;
      margin-top: 10px;
      display: grid;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
    }
    .compare-row {
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }
    .compare-row strong {
      color: var(--text);
    }
    .delta-hot { color: var(--warn); }
    .delta-cold { color: #60a5fa; }
    .model-badge {
      display: inline-block;
      width: fit-content;
      margin-top: 4px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .02em;
      border: 1px solid var(--border);
    }
    .badge-track {
      color: #86efac;
      background: rgba(34, 197, 94, 0.12);
      border-color: rgba(34, 197, 94, 0.35);
    }
    .badge-hot {
      color: #fcd34d;
      background: rgba(245, 158, 11, 0.12);
      border-color: rgba(245, 158, 11, 0.35);
    }
    .badge-cold {
      color: #93c5fd;
      background: rgba(96, 165, 250, 0.12);
      border-color: rgba(96, 165, 250, 0.35);
    }
    .badge-unknown {
      color: var(--muted);
      background: rgba(148, 163, 184, 0.10);
      border-color: rgba(148, 163, 184, 0.25);
    }
    .forecast-block {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }
    .forecast-row {
      border-top: 1px solid var(--border);
      padding-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .forecast-row strong {
      color: var(--text);
      font-size: 13px;
      display: block;
      margin-top: 2px;
    }
    .spread {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }
    .time {
      font-size: 12px;
      color: var(--muted);
      margin-top: 10px;
      min-height: 16px;
    }
    .error {
      color: var(--bad);
      font-size: 12px;
      margin-top: 8px;
    }
    a {
      color: #93c5fd;
      font-size: 12px;
      text-decoration: none;
      display: inline-block;
      margin-top: 10px;
    }
    .footer {
      color: var(--muted);
      font-size: 12px;
      margin-top: 14px;
      line-height: 1.4;
    }
  </style>
</head>
<body>
<header>
  <h1>Kalshi WX Ensemble + Observed Temps</h1>
  <div class="sub">
    Open-Meteo GFS/GEFS ensemble model temps plus latest public NWS station observations.
    Observed high/low is calculated from public station observations returned so far today using each city's local timezone, including daylight saving time where applicable.
  </div>
  <div class="controls">
    <button onclick="loadTemps(true)">Refresh now</button>
    <select id="sortMode" onchange="renderCards()">
      <option value="default">Sort: Default</option>
      <option value="delta_desc">Sort: Running hottest vs model</option>
      <option value="delta_asc">Sort: Running coldest vs model</option>
      <option value="obs_desc">Sort: Warmest observed</option>
      <option value="obs_asc">Sort: Coldest observed</option>
      <option value="today_high_desc">Sort: Highest forecast today</option>
      <option value="code">Sort: City code</option>
    </select>
    <span class="sub" id="status">Loading...</span>
  </div>
</header>

<main>
  <div class="grid" id="grid"></div>
  <div class="footer">
    Auto-refreshes every 60 minutes while open. Local observed times use each city's timezone and automatically account for daylight saving time. This is not hidden ASOS 1-minute data and not final CLI settlement data.
  </div>
</main>

<script>
const CITIES = [
  {code:"DEN", name:"Denver", lat:39.8561, lon:-104.6737, station:"KDEN", tz:"America/Denver"},
  {code:"NYC", name:"New York City", lat:40.7828, lon:-73.9653, station:"KNYC", tz:"America/New_York"},
  {code:"PHI", name:"Philadelphia", lat:39.8719, lon:-75.2411, station:"KPHL", tz:"America/New_York"},
  {code:"CHI", name:"Chicago", lat:41.7868, lon:-87.7522, station:"KMDW", tz:"America/Chicago"},
  {code:"LA", name:"Los Angeles", lat:33.9425, lon:-118.4081, station:"KLAX", tz:"America/Los_Angeles"},
  {code:"MIA", name:"Miami", lat:25.7959, lon:-80.2870, station:"KMIA", tz:"America/New_York"},
  {code:"SF", name:"San Francisco", lat:37.6213, lon:-122.3790, station:"KSFO", tz:"America/Los_Angeles"},
  {code:"SEA", name:"Seattle", lat:47.4502, lon:-122.3088, station:"KSEA", tz:"America/Los_Angeles"},
  {code:"ATL", name:"Atlanta", lat:33.6407, lon:-84.4277, station:"KATL", tz:"America/New_York"},
  {code:"AUS", name:"Austin", lat:30.1975, lon:-97.6664, station:"KAUS", tz:"America/Chicago"},
  {code:"BOS", name:"Boston", lat:42.3656, lon:-71.0096, station:"KBOS", tz:"America/New_York"},
  {code:"DAL", name:"Dallas", lat:32.8471, lon:-96.8518, station:"KDAL", tz:"America/Chicago"},
  {code:"DC", name:"Washington DC", lat:38.8512, lon:-77.0402, station:"KDCA", tz:"America/New_York"},
  {code:"HOU", name:"Houston", lat:29.6454, lon:-95.2789, station:"KHOU", tz:"America/Chicago"},
  {code:"LV", name:"Las Vegas", lat:36.0801, lon:-115.1522, station:"KLAS", tz:"America/Los_Angeles"},
  {code:"MIN", name:"Minneapolis", lat:44.8848, lon:-93.2223, station:"KMSP", tz:"America/Chicago"},
  {code:"NOLA", name:"New Orleans", lat:29.9934, lon:-90.2580, station:"KMSY", tz:"America/Chicago"},
  {code:"OKC", name:"Oklahoma City", lat:35.3931, lon:-97.6007, station:"KOKC", tz:"America/Chicago"},
  {code:"PHX", name:"Phoenix", lat:33.4342, lon:-112.0116, station:"KPHX", tz:"America/Phoenix"},
  {code:"SATX", name:"San Antonio", lat:29.5337, lon:-98.4698, station:"KSAT", tz:"America/Chicago"}
];

let rows = CITIES.map(c => ({
  ...c,
  modelNow: null,
  modelTime: "",
  obsTemp: null,
  obsHigh: null,
  obsLow: null,
  obsHighTime: "",
  obsLowTime: "",
  obsTime: "",
  days: [],
  n: 0,
  error: ""
}));

function ensembleUrl(c) {
  return `https://ensemble-api.open-meteo.com/v1/ensemble?latitude=${c.lat}&longitude=${c.lon}&hourly=temperature_2m&models=gfs_seamless&forecast_days=3&temperature_unit=fahrenheit&timezone=UTC`;
}

function obsUrl(c) {
  const end = new Date();
  const start = new Date(end.getTime() - 36 * 60 * 60 * 1000);
  return `https://api.weather.gov/stations/${c.station}/observations?start=${start.toISOString()}&end=${end.toISOString()}`;
}

function fmtTemp(t) {
  return (t === null || t === undefined || Number.isNaN(t)) ? "—" : `${Math.round(t)}°`;
}

function fmtOne(t) {
  return (t === null || t === undefined || Number.isNaN(t)) ? "—" : `${t.toFixed(1)}°`;
}

function cToF(c) {
  return c * 9 / 5 + 32;
}

function mean(vals) {
  if (!vals.length) return null;
  return vals.reduce((a,b) => a+b, 0) / vals.length;
}

function percentile(vals, p) {
  if (!vals.length) return null;
  const sorted = [...vals].sort((a,b) => a-b);
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function localYmd(isoTime, tz) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(new Date(isoTime));

  const y = parts.find(p => p.type === "year")?.value;
  const m = parts.find(p => p.type === "month")?.value;
  const d = parts.find(p => p.type === "day")?.value;
  return `${y}-${m}-${d}`;
}

function localTime(isoTime, tz) {
  if (!isoTime) return "";
  try {
    return new Date(isoTime).toLocaleString(undefined, {
      timeZone: tz,
      month: "numeric",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit"
    });
  } catch {
    return isoTime;
  }
}

function dayLabel(isoDate, idx) {
  if (idx === 0) return "Today";
  if (idx === 1) return "Tomorrow";
  try {
    return new Date(isoDate + "T12:00:00").toLocaleDateString(undefined, {
      weekday: "short",
      month: "numeric",
      day: "numeric"
    });
  } catch {
    return `Day ${idx + 1}`;
  }
}

function tempKeys(hourly) {
  return Object.keys(hourly || {}).filter(k => k === "temperature_2m" || k.startsWith("temperature_2m_member"));
}

function nearestIndex(times) {
  if (!times || !times.length) return 0;
  const now = Date.now();
  let best = 0;
  let bestDiff = Infinity;
  times.forEach((t, i) => {
    const diff = Math.abs(new Date(t).getTime() - now);
    if (diff < bestDiff) {
      best = i;
      bestDiff = diff;
    }
  });
  return best;
}

function parseEnsemble(c, data) {
  const hourly = data.hourly || {};
  const times = hourly.time || [];
  const keys = marketTempKeys(hourly);

  if (!times.length || !keys.length) {
    throw new Error("No ensemble hourly temps returned");
  }

  const ni = nearestIndex(times);
  const nowVals = keys.map(k => hourly[k]?.[ni]).filter(v => v !== null && v !== undefined).map(Number);
  const modelNow = mean(nowVals);

  const grouped = {};
  times.forEach((t, i) => {
    const d = String(t).slice(0, 10);
    if (!grouped[d]) grouped[d] = {};
    keys.forEach(k => {
      const v = hourly[k]?.[i];
      if (v === null || v === undefined) return;
      if (!grouped[d][k]) grouped[d][k] = [];
      grouped[d][k].push(Number(v));
    });
  });

  const days = Object.keys(grouped).slice(0, 3).map((date, idx) => {
    const highs = [];
    const lows = [];

    for (const k of keys) {
      const vals = grouped[date][k] || [];
      if (!vals.length) continue;
      highs.push(Math.max(...vals));
      lows.push(Math.min(...vals));
    }

    return {
      date,
      label: dayLabel(date, idx),
      highMean: mean(highs),
      lowMean: mean(lows),
      highP10: percentile(highs, 0.10),
      highP50: percentile(highs, 0.50),
      highP90: percentile(highs, 0.90),
      lowP10: percentile(lows, 0.10),
      lowP50: percentile(lows, 0.50),
      lowP90: percentile(lows, 0.90),
      n: highs.length
    };
  });

  return {
    ...c,
    modelNow,
    modelTime: times[ni] || "",
    days,
    n: keys.length,
    error: ""
  };
}

function parseObserved(c, data) {
  const features = data?.features || [];
  const todayKey = localYmd(new Date().toISOString(), c.tz);
  const obs = [];

  for (const f of features) {
    const p = f.properties || {};
    const iso = p.timestamp;
    const tempC = p.temperature?.value;
    if (!iso || tempC === null || tempC === undefined) continue;

    obs.push({
      time: iso,
      temp: cToF(Number(tempC)),
      day: localYmd(iso, c.tz)
    });
  }

  obs.sort((a, b) => new Date(b.time) - new Date(a.time));

  const latest = obs[0] || null;
  const todayObs = obs.filter(o => o.day === todayKey);

  let highObs = null;
  let lowObs = null;

  for (const o of todayObs) {
    if (!highObs || o.temp > highObs.temp) highObs = o;
    if (!lowObs || o.temp < lowObs.temp) lowObs = o;
  }

  return {
    obsTemp: latest ? latest.temp : null,
    obsTime: latest ? latest.time : "",
    obsHigh: highObs ? highObs.temp : null,
    obsLow: lowObs ? lowObs.temp : null,
    obsHighTime: highObs ? highObs.time : "",
    obsLowTime: lowObs ? lowObs.time : ""
  };
}

function deltaText(r) {
  if (r.modelNow === null || r.obsTemp === null) return "Delta: —";
  const d = r.obsTemp - r.modelNow;
  const sign = d >= 0 ? "+" : "";
  return `Observed vs model: ${sign}${d.toFixed(1)}°`;
}

function deltaClass(r) {
  if (r.modelNow === null || r.obsTemp === null) return "";
  const d = r.obsTemp - r.modelNow;
  if (d >= 1.5) return "delta-hot";
  if (d <= -1.5) return "delta-cold";
  return "";
}

function modelStatusText(r) {
  if (r.modelNow === null || r.obsTemp === null) return "Waiting on observed/model data";
  const d = r.obsTemp - r.modelNow;
  if (d >= 1.5) return "Running hot vs model";
  if (d <= -1.5) return "Running cold vs model";
  return "On track with model";
}

function modelStatusClass(r) {
  if (r.modelNow === null || r.obsTemp === null) return "badge-unknown";
  const d = r.obsTemp - r.modelNow;
  if (d >= 1.5) return "badge-hot";
  if (d <= -1.5) return "badge-cold";
  return "badge-track";
}

function renderCards() {
  const mode = document.getElementById("sortMode").value;
  let sorted = [...rows];

  const delta = r => (r.obsTemp ?? 0) - (r.modelNow ?? 0);
  if (mode === "delta_desc") sorted.sort((a,b) => delta(b) - delta(a));
  if (mode === "delta_asc") sorted.sort((a,b) => delta(a) - delta(b));
  if (mode === "obs_desc") sorted.sort((a,b) => (b.obsTemp ?? -999) - (a.obsTemp ?? -999));
  if (mode === "obs_asc") sorted.sort((a,b) => (a.obsTemp ?? 999) - (b.obsTemp ?? 999));
  if (mode === "today_high_desc") sorted.sort((a,b) => ((b.days?.[0]?.highMean) ?? -999) - ((a.days?.[0]?.highMean) ?? -999));
  if (mode === "code") sorted.sort((a,b) => a.code.localeCompare(b.code));

  document.getElementById("grid").innerHTML = sorted.map(r => `
    <div class="card">
      <div class="code">${r.code}</div>
      <div class="name">${r.name} · ${r.station}</div>

      <div class="temp">${fmtTemp(r.modelNow)}</div>
      <div class="label">Forecasted now · N=${r.n || "—"}</div>

      <div class="compare">
        <div class="compare-row">
          <span>Last observed</span>
          <strong>${fmtTemp(r.obsTemp)}</strong>
        </div>
        <div class="spread">${r.obsTime ? "Observed at: " + localTime(r.obsTime, r.tz) : ""}</div>

        <div class="compare-row">
          <span>Observed high today</span>
          <strong>${fmtTemp(r.obsHigh)}</strong>
        </div>
        <div class="spread">${r.obsHighTime ? "High time: " + localTime(r.obsHighTime, r.tz) : ""}</div>

        <div class="compare-row">
          <span>Observed low today</span>
          <strong>${fmtTemp(r.obsLow)}</strong>
        </div>
        <div class="spread">${r.obsLowTime ? "Low time: " + localTime(r.obsLowTime, r.tz) : ""}</div>

        <div class="compare-row ${deltaClass(r)}">
          <span>${deltaText(r)}</span>
          <strong></strong>
        </div>
        <div class="model-badge ${modelStatusClass(r)}">${modelStatusText(r)}</div>
      </div>

      <div class="forecast-block">
        ${(r.days || []).map(d => `
          <div class="forecast-row">
            ${d.label}
            <strong>Ens H ${fmtTemp(d.highMean)} / L ${fmtTemp(d.lowMean)}</strong>
            <div class="spread">
              High P10/P50/P90: ${fmtTemp(d.highP10)} / ${fmtTemp(d.highP50)} / ${fmtTemp(d.highP90)}
            </div>
            <div class="spread">
              Low P10/P50/P90: ${fmtTemp(d.lowP10)} / ${fmtTemp(d.lowP50)} / ${fmtTemp(d.lowP90)}
            </div>
          </div>
        `).join("")}
      </div>

      <div class="time">${r.modelTime ? "Nearest model hour: " + localTime(r.modelTime, r.tz) : ""}</div>
      <div class="time">${r.obsTime ? "Last observed time: " + localTime(r.obsTime, r.tz) : ""}</div>
      ${r.error ? `<div class="error">${r.error}</div>` : ""}
      <a href="${ensembleUrl(r)}" target="_blank" rel="noopener">Raw ensemble</a>
      <br>
      <a href="${obsUrl(r)}" target="_blank" rel="noopener">Raw NWS observations</a>
    </div>
  `).join("");
}

async function fetchCity(c) {
  const [ensRes, obsRes] = await Promise.all([
    fetch(ensembleUrl(c), {cache: "no-store"}),
    fetch(obsUrl(c), {cache: "no-store"})
  ]);

  if (!ensRes.ok) throw new Error(`Ensemble HTTP ${ensRes.status}`);

  const ensData = await ensRes.json();
  const row = parseEnsemble(c, ensData);

  if (obsRes.ok) {
    const obsData = await obsRes.json();
    return {...row, ...parseObserved(c, obsData)};
  }

  return {...row, error: `Obs HTTP ${obsRes.status}`};
}

async function loadTemps(manual=false) {
  const status = document.getElementById("status");
  status.textContent = manual ? "Refreshing..." : "Loading...";
  renderCards();

  const results = await Promise.all(CITIES.map(async c => {
    try {
      return await fetchCity(c);
    } catch (e) {
      return {
        ...c,
        modelNow: null,
        modelTime: "",
        obsTemp: null,
        obsHigh: null,
        obsLow: null,
        obsHighTime: "",
        obsLowTime: "",
        obsTime: "",
        days: [],
        n: 0,
        error: String(e.message || e)
      };
    }
  }));

  rows = results;
  renderCards();

  const now = new Date();
  status.textContent = `Last refresh: ${now.toLocaleTimeString()} · Next auto-refresh in 60 min`;
}

loadTemps();
setInterval(() => loadTemps(false), 60 * 60 * 1000);
</script>
</body>
</html>
"""

@app.get("/ensemble-temps", response_class=HTMLResponse)
async def ensemble_temps_dashboard():
    return HTMLResponse(ENSEMBLE_TEMPS_DASHBOARD_HTML)

KALSHI_TEMP_SERIES_CANDIDATES = {
    "atl": {
        "name": "Atlanta", "station": "KATL",
        "high": ["KXHIGHTATL", "KXHIGHATL"],
        "low": ["KXLOWTATL", "KXLOWATL"],
    },
    "aus": {
        "name": "Austin", "station": "KAUS",
        "high": ["KXHIGHAUS", "KXHIGHTAUS"],
        "low": ["KXLOWAUS", "KXLOWTAUS"],
    },
    "bos": {
        "name": "Boston", "station": "KBOS",
        "high": ["KXHIGHTBOS", "KXHIGHBOS"],
        "low": ["KXLOWTBOS", "KXLOWBOS"],
    },
    "chicago": {
        "name": "Chicago", "station": "KMDW",
        "high": ["KXHIGHCHI", "KXHIGHTCHI"],
        "low": ["KXLOWCHI", "KXLOWTCHI"],
    },
    "dal": {
        "name": "Dallas", "station": "KDAL",
        "high": ["KXHIGHTDAL", "KXHIGHDAL"],
        "low": ["KXLOWTDAL", "KXLOWDAL"],
    },
    "dc": {
        "name": "Washington DC", "station": "KDCA",
        "high": ["KXHIGHTDC", "KXHIGHDC"],
        "low": ["KXLOWTDC", "KXLOWDC"],
    },
    "denver": {
        "name": "Denver", "station": "KDEN",
        "high": ["KXHIGHDEN", "KXHIGHTEMPDEN", "KXHIGHTDEN"],
        "low": ["KXLOWDEN", "KXLOWTDEN"],
    },
    "hou": {
        "name": "Houston", "station": "KHOU",
        "high": ["KXHIGHHOU", "KXHIGHTHOU"],
        "low": ["KXLOWTHOU", "KXLOWHOU"],
    },
    "los_angeles": {
        "name": "Los Angeles", "station": "KLAX",
        "high": ["KXHIGHLAX", "KXHIGHLA", "KXHIGHTLA"],
        "low": ["KXLOWLAX", "KXLOWTLAX", "KXLOWLA"],
    },
    "lv": {
        "name": "Las Vegas", "station": "KLAS",
        "high": ["KXHIGHTLV", "KXHIGHLV"],
        "low": ["KXLOWTLV", "KXLOWLV"],
    },
    "miami": {
        "name": "Miami", "station": "KMIA",
        "high": ["KXHIGHMIA", "KXHIGHTMIA"],
        "low": ["KXLOWMIA", "KXLOWTMIA"],
    },
    "min": {
        "name": "Minneapolis", "station": "KMSP",
        "high": ["KXHIGHTMIN", "KXHIGHMIN"],
        "low": ["KXLOWTMIN", "KXLOWMIN"],
    },
    "nola": {
        "name": "New Orleans", "station": "KMSY",
        "high": ["KXHIGHTNOLA", "KXHIGHNOLA"],
        "low": ["KXLOWTNOLA", "KXLOWNOLA"],
    },
    "nyc": {
        "name": "New York City", "station": "KNYC",
        "high": ["KXHIGHNY", "KXHIGHNYC", "KXHIGHNY0", "KXHIGHNYD", "KXHIGHTNY", "KXHIGHTNYC"],
        "low": ["KXLOWNY", "KXLOWNYC", "KXLOWTNY", "KXLOWTNYC"],
    },
    "okc": {
        "name": "Oklahoma City", "station": "KOKC",
        "high": ["KXHIGHTOKC", "KXHIGHOKC"],
        "low": ["KXLOWTOKC", "KXLOWOKC"],
    },
    "phi": {
        "name": "Philadelphia", "station": "KPHL",
        "high": ["KXHIGHPHI", "KXHIGHPHIL", "KXHIGHTPHI", "KXHIGHTPHIL"],
        "low": ["KXLOWPHI", "KXLOWPHIL", "KXLOWTPHI", "KXLOWTPHIL"],
    },
    "phx": {
        "name": "Phoenix", "station": "KPHX",
        "high": ["KXHIGHTPHX", "KXHIGHPHX"],
        "low": ["KXLOWTPHX", "KXLOWPHX"],
    },
    "satx": {
        "name": "San Antonio", "station": "KSAT",
        "high": ["KXHIGHTSATX", "KXHIGHSATX"],
        "low": ["KXLOWTSATX", "KXLOWSATX"],
    },
    "sea": {
        "name": "Seattle", "station": "KSEA",
        "high": ["KXHIGHSEA", "KXHIGHTSEA"],
        "low": ["KXLOWSEA", "KXLOWTSEA"],
    },
    "sf": {
        "name": "San Francisco", "station": "KSFO",
        "high": ["KXHIGHSF", "KXHIGHTSFO", "KXHIGHSFO"],
        "low": ["KXLOWSF", "KXLOWSFO", "KXLOWTSF", "KXLOWTSFO"],
    },
}


@app.get("/api/kalshi/series-discovery")
async def kalshi_series_discovery():
    """
    Test possible Kalshi temperature series tickers and report which ones currently return open markets.
    This uses Render's Kalshi credentials, not the browser's.
    """
    from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present

    if not kalshi_credentials_present():
        return {
            "ok": False,
            "error": "Kalshi credentials are not configured on this service.",
            "cities": {},
        }

    client = KalshiClient()
    output = {
        "ok": True,
        "note": "Series with open_count > 0 are the active candidates to use for the market board.",
        "cities": {},
    }

    for city_key, cfg in KALSHI_TEMP_SERIES_CANDIDATES.items():
        city_result = {
            "name": cfg["name"],
            "station": cfg["station"],
            "high": [],
            "low": [],
        }

        for market_type in ("high", "low"):
            for series in cfg[market_type]:
                item = {
                    "series": series,
                    "open_count": 0,
                    "sample_tickers": [],
                    "sample_titles": [],
                    "error": None,
                }

                try:
                    data = await client.get_markets({
                        "series_ticker": series,
                        "status": "open",
                        "limit": 10,
                    })
                    markets = data.get("markets", []) or []
                    item["open_count"] = len(markets)
                    item["sample_tickers"] = [m.get("ticker") for m in markets[:5]]
                    item["sample_titles"] = [m.get("title") for m in markets[:3]]
                except Exception as e:
                    item["error"] = str(e)

                city_result[market_type].append(item)

        output["cities"][city_key] = city_result

    return output

ACTIVE_KALSHI_TEMP_SERIES = {
    "atl": {"name": "Atlanta", "station": "KATL", "tz": "America/New_York", "high": "KXHIGHTATL", "low": "KXLOWTATL"},
    "aus": {"name": "Austin", "station": "KAUS", "tz": "America/Chicago", "high": "KXHIGHAUS", "low": "KXLOWTAUS"},
    "bos": {"name": "Boston", "station": "KBOS", "tz": "America/New_York", "high": "KXHIGHTBOS", "low": "KXLOWTBOS"},
    "chicago": {"name": "Chicago", "station": "KMDW", "tz": "America/Chicago", "high": "KXHIGHCHI", "low": "KXLOWTCHI"},
    "dal": {"name": "Dallas", "station": "KDAL", "tz": "America/Chicago", "high": "KXHIGHTDAL", "low": "KXLOWTDAL"},
    "dc": {"name": "Washington DC", "station": "KDCA", "tz": "America/New_York", "high": "KXHIGHTDC", "low": "KXLOWTDC"},
    "denver": {"name": "Denver", "station": "KDEN", "tz": "America/Denver", "high": "KXHIGHDEN", "low": "KXLOWTDEN"},
    "hou": {"name": "Houston", "station": "KHOU", "tz": "America/Chicago", "high": "KXHIGHTHOU", "low": "KXLOWTHOU"},
    "los_angeles": {"name": "Los Angeles", "station": "KLAX", "tz": "America/Los_Angeles", "high": "KXHIGHLAX", "low": "KXLOWTLAX"},
    "lv": {"name": "Las Vegas", "station": "KLAS", "tz": "America/Los_Angeles", "high": "KXHIGHTLV", "low": "KXLOWTLV"},
    "miami": {"name": "Miami", "station": "KMIA", "tz": "America/New_York", "high": "KXHIGHMIA", "low": "KXLOWTMIA"},
    "min": {"name": "Minneapolis", "station": "KMSP", "tz": "America/Chicago", "high": "KXHIGHTMIN", "low": "KXLOWTMIN"},
    "nola": {"name": "New Orleans", "station": "KMSY", "tz": "America/Chicago", "high": "KXHIGHTNOLA", "low": "KXLOWTNOLA"},
    "nyc": {"name": "New York City", "station": "KNYC", "tz": "America/New_York", "high": "KXHIGHNY", "low": "KXLOWTNYC"},
    "okc": {"name": "Oklahoma City", "station": "KOKC", "tz": "America/Chicago", "high": "KXHIGHTOKC", "low": "KXLOWTOKC"},
    "phi": {"name": "Philadelphia", "station": "KPHL", "tz": "America/New_York", "high": "KXHIGHPHIL", "low": "KXLOWTPHIL"},
    "phx": {"name": "Phoenix", "station": "KPHX", "tz": "America/Phoenix", "high": "KXHIGHTPHX", "low": "KXLOWTPHX"},
    "satx": {"name": "San Antonio", "station": "KSAT", "tz": "America/Chicago", "high": "KXHIGHTSATX", "low": "KXLOWTSATX"},
    "sea": {"name": "Seattle", "station": "KSEA", "tz": "America/Los_Angeles", "high": "KXHIGHTSEA", "low": "KXLOWTSEA"},
    "sf": {"name": "San Francisco", "station": "KSFO", "tz": "America/Los_Angeles", "high": "KXHIGHTSFO", "low": "KXLOWTSFO"},
}


def _best_yes_bid_ask_from_orderbook(orderbook_payload: dict) -> dict:
    """
    Extract best YES bid/ask in cents from Kalshi orderbook response.
    Handles both common shapes:
      {"orderbook": {"yes": [[price, qty]], "no": [[price, qty]]}}
      {"yes": [[price, qty]], "no": [[price, qty]]}
    """
    ob = orderbook_payload.get("orderbook") or orderbook_payload
    yes_levels = ob.get("yes") or []
    no_levels = ob.get("no") or []

    def clean_price(x):
        try:
            if isinstance(x, (list, tuple)) and x:
                return float(x[0])
            return float(x)
        except Exception:
            return None

    yes_prices = [clean_price(x) for x in yes_levels]
    no_prices = [clean_price(x) for x in no_levels]
    yes_prices = [x for x in yes_prices if x is not None and x > 0]
    no_prices = [x for x in no_prices if x is not None and x > 0]

    # YES bid is highest YES bid.
    yes_bid = max(yes_prices) if yes_prices else None

    # YES ask can be derived from highest NO bid: buying YES takes the opposite side of NO.
    # If NO bid is 20c, implied YES ask is about 80c.
    yes_ask = None
    if no_prices:
        yes_ask = 100.0 - max(no_prices)

    return {
        "yes_bid_effective": yes_bid,
        "yes_ask_effective": yes_ask,
    }


def _market_price_score(m: dict) -> float:
    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    last_price = m.get("last_price")

    if yes_bid is not None and yes_ask is not None and yes_bid > 0 and yes_ask > 0:
        return float(yes_bid + yes_ask) / 2.0
    if yes_bid is not None and yes_bid > 0:
        return float(yes_bid)
    if last_price is not None and last_price > 0:
        return float(last_price)
    if yes_ask is not None and yes_ask > 0:
        return float(yes_ask)
    return 0.0


def _parse_temp_bucket(title: str, ticker: str) -> dict:
    import re

    clean_title = (title or ticker or "").replace("**", "")

    range_match = re.search(r"be\s+(\d+)\s*-\s*(\d+)", clean_title)
    if range_match:
        lo = int(range_match.group(1))
        hi = int(range_match.group(2))
        return {
            "bucket_type": "range",
            "label": f"{lo}-{hi}°",
            "low": lo,
            "high": hi,
            "threshold": None,
            "center": (lo + hi) / 2.0,
        }

    above_match = re.search(r"be\s*>\s*(\d+)", clean_title)
    if above_match:
        threshold = int(above_match.group(1))
        return {
            "bucket_type": "above",
            "label": f"More than {threshold}°",
            "low": threshold + 1,
            "high": None,
            "threshold": threshold,
            "center": threshold + 1.0,
        }

    below_match = re.search(r"be\s*<\s*(\d+)", clean_title)
    if below_match:
        threshold = int(below_match.group(1))
        return {
            "bucket_type": "below",
            "label": f"Less than {threshold}°",
            "low": None,
            "high": threshold - 1,
            "threshold": threshold,
            "center": threshold - 1.0,
        }

    # Fallback for Bxx.5 bracket tickers. Example B82.5 = 82-83.
    ticker_match = re.search(r"-B(\d+)\.5$", ticker or "")
    if ticker_match:
        lo = int(ticker_match.group(1))
        hi = lo + 1
        return {
            "bucket_type": "range",
            "label": f"{lo}-{hi}°",
            "low": lo,
            "high": hi,
            "threshold": None,
            "center": (lo + hi) / 2.0,
        }

    return {
        "bucket_type": "unknown",
        "label": clean_title or ticker,
        "low": None,
        "high": None,
        "threshold": None,
        "center": None,
    }


async def _fetch_kalshi_series_markets(client, series: str) -> list:
    markets = []
    cursor = None

    while True:
        params = {
            "series_ticker": series,
            "status": "open",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        data = await client.get_markets(params)
        raw_markets = data.get("markets", []) or []
        markets.extend(raw_markets)

        cursor = data.get("cursor")
        if not cursor or not raw_markets:
            break

    return markets




def _wx_cents(value):
    """Convert Kalshi dollar price fields like 0.66 into cents like 66."""
    try:
        if value is None:
            return None
        v = float(value)
        if v <= 0:
            return None
        # Kalshi *_dollars fields are decimal dollars/probability: 0.66 = 66c.
        if v <= 1:
            return round(v * 100, 2)
        # Older/int fields may already be cents.
        return round(v, 2)
    except Exception:
        return None


def _wx_num(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _wx_market_prices(m: dict) -> dict:
    yes_bid = _wx_cents(m.get("yes_bid_dollars"))
    yes_ask = _wx_cents(m.get("yes_ask_dollars"))
    no_bid = _wx_cents(m.get("no_bid_dollars"))
    no_ask = _wx_cents(m.get("no_ask_dollars"))
    last_price = _wx_cents(m.get("last_price_dollars"))

    # Backward compatibility with older fields.
    if yes_bid is None:
        yes_bid = _wx_cents(m.get("yes_bid"))
    if yes_ask is None:
        yes_ask = _wx_cents(m.get("yes_ask"))
    if no_bid is None:
        no_bid = _wx_cents(m.get("no_bid"))
    if no_ask is None:
        no_ask = _wx_cents(m.get("no_ask"))
    if last_price is None:
        last_price = _wx_cents(m.get("last_price"))

    # Derive YES side from NO side if needed.
    if yes_bid is None and no_ask is not None:
        yes_bid = max(0, 100 - no_ask)
    if yes_ask is None and no_bid is not None:
        yes_ask = max(0, 100 - no_bid)

    if yes_bid is None and last_price is not None:
        yes_bid = last_price
    if yes_ask is None and last_price is not None:
        yes_ask = last_price

    if yes_bid is not None and yes_ask is not None:
        midpoint = (yes_bid + yes_ask) / 2
        spread = abs(yes_ask - yes_bid)
    elif yes_bid is not None:
        midpoint = yes_bid
        spread = None
    elif yes_ask is not None:
        midpoint = yes_ask
        spread = None
    else:
        midpoint = 0
        spread = None

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
        "midpoint": midpoint,
        "spread": spread,
        "volume": _wx_num(m.get("volume_fp")) or _wx_num(m.get("volume")) or 0,
        "volume_24h": _wx_num(m.get("volume_24h_fp")) or 0,
        "open_interest": _wx_num(m.get("open_interest_fp")) or _wx_num(m.get("open_interest")) or 0,
        "liquidity": _wx_num(m.get("liquidity_dollars")) or _wx_num(m.get("liquidity")) or 0,
    }


def _market_price_score(m: dict) -> float:
    return float(_wx_market_prices(m).get("midpoint") or 0)


def _wx_ticker_date_code(ticker: str):
    """Extract Kalshi date code from ticker like KXLOWTATL-26JUN09-B70.5."""
    try:
        import re
        m = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", ticker or "")
        return m.group(1) if m else None
    except Exception:
        return None


def _wx_select_single_market_date(buckets: list) -> list:
    """
    Kalshi can return more than one date in the same series.
    Keep only one date so a city card does not show duplicate buckets.
    """
    grouped = {}
    for b in buckets or []:
        d = _wx_ticker_date_code(b.get("ticker"))
        grouped.setdefault(d or "UNKNOWN", []).append(b)

    if not grouped:
        return buckets or []

    best_date = sorted(grouped.keys(), key=lambda d: (-len(grouped[d]), d))[0]
    return grouped[best_date]



_WX_MARKET_FORECAST_CACHE = {}

_WX_MARKET_COORDS = {
    "atl": {"lat": 33.6407, "lon": -84.4277},
    "aus": {"lat": 30.1975, "lon": -97.6664},
    "bos": {"lat": 42.3656, "lon": -71.0096},
    "chicago": {"lat": 41.7868, "lon": -87.7522},
    "dal": {"lat": 32.8471, "lon": -96.8518},
    "dc": {"lat": 38.8512, "lon": -77.0402},
    "denver": {"lat": 39.8561, "lon": -104.6737},
    "hou": {"lat": 29.6454, "lon": -95.2789},
    "los_angeles": {"lat": 33.9425, "lon": -118.4081},
    "lv": {"lat": 36.0801, "lon": -115.1522},
    "miami": {"lat": 25.7959, "lon": -80.2870},
    "min": {"lat": 44.8848, "lon": -93.2223},
    "nola": {"lat": 29.9934, "lon": -90.2580},
    "nyc": {"lat": 40.7828, "lon": -73.9653},
    "okc": {"lat": 35.3931, "lon": -97.6007},
    "phi": {"lat": 39.8719, "lon": -75.2411},
    "phx": {"lat": 33.4342, "lon": -112.0116},
    "satx": {"lat": 29.5337, "lon": -98.4698},
    "sea": {"lat": 47.4502, "lon": -122.3088},
    "sf": {"lat": 37.6213, "lon": -122.3790},
}


def _wx_mean(vals):
    nums = []
    for v in vals:
        try:
            if v is not None:
                nums.append(float(v))
        except Exception:
            pass
    if not nums:
        return None
    return sum(nums) / len(nums)


async def _wx_fetch_market_forecast(city_key: str, tz_name: str) -> dict:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    import httpx

    coords = _WX_MARKET_COORDS.get(city_key)
    if not coords:
        return {}

    cache_key = f"{city_key}:{tz_name}"
    now_ts = datetime.now(timezone.utc).timestamp()

    cached = _WX_MARKET_FORECAST_CACHE.get(cache_key)
    if cached and now_ts - cached.get("ts", 0) < 600:
        return cached.get("data", {})

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")

    today_local = datetime.now(tz).date()

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m",
        "models": "gfs_seamless",
        "forecast_days": 2,
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get("https://ensemble-api.open-meteo.com/v1/ensemble", params=params)
            r.raise_for_status()
            data = r.json()

        hourly = data.get("hourly", {}) or {}
        times = hourly.get("time", []) or []
        keys = [
            k for k in hourly.keys()
            if k == "temperature_2m" or k.startswith("temperature_2m_member")
        ]

        best_high = None
        best_low = None

        for i, t in enumerate(times):
            try:
                dt_utc = datetime.fromisoformat(str(t).replace("Z", "")).replace(tzinfo=timezone.utc)
                dt_local = dt_utc.astimezone(tz)
            except Exception:
                continue

            if dt_local.date() != today_local:
                continue

            avg = _wx_mean([hourly.get(k, [None] * len(times))[i] for k in keys])
            if avg is None:
                continue

            if best_high is None or avg > best_high["temp"]:
                best_high = {"temp": avg, "time": dt_utc.isoformat()}
            if best_low is None or avg < best_low["temp"]:
                best_low = {"temp": avg, "time": dt_utc.isoformat()}

        out = {
            "high": best_high["temp"] if best_high else None,
            "high_time": best_high["time"] if best_high else "",
            "low": best_low["temp"] if best_low else None,
            "low_time": best_low["time"] if best_low else "",
            "members": len(keys),
        }

        _WX_MARKET_FORECAST_CACHE[cache_key] = {"ts": now_ts, "data": out}
        return out

    except Exception as e:
        return {
            "high": None,
            "high_time": "",
            "low": None,
            "low_time": "",
            "members": 0,
            "error": str(e),
        }



@app.get("/api/forecast-accuracy")
def api_forecast_accuracy():
    """
    Return cached 30-day forecast accuracy summary.

    Cache is built by:
      python tools/build_forecast_accuracy_cache.py
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    cache_path = Path("state/forecast_accuracy_summary.json")

    if not cache_path.exists():
        return {
            "ok": False,
            "error": "forecast accuracy cache not found",
            "hint": "Run: python tools/build_forecast_accuracy_cache.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cities": {},
        }

    try:
        data = json.loads(cache_path.read_text())
    except Exception as e:
        return {
            "ok": False,
            "error": f"could not read forecast accuracy cache: {e}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cities": {},
        }

    if isinstance(data, dict):
        data["ok"] = True
        return data

    return {
        "ok": False,
        "error": "forecast accuracy cache had unexpected format",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cities": {},
    }

@app.get("/api/kalshi/market-board")
async def kalshi_market_board():
    from datetime import datetime, timezone
    from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present

    if not kalshi_credentials_present():
        return {
            "ok": False,
            "error": "Kalshi credentials are not configured on this service.",
            "cities": [],
        }

    client = KalshiClient()
    cities_out = []

    for city_key, cfg in ACTIVE_KALSHI_TEMP_SERIES.items():
        city_out = {
            "city_key": city_key,
            "name": cfg["name"],
            "station": cfg["station"],
            "tz": cfg["tz"],
            "markets": {
                "high": {"series": cfg["high"], "buckets": [], "error": None},
                "low": {"series": cfg["low"], "buckets": [], "error": None},
            },
        }

        coords_for_city = _WX_MARKET_COORDS.get(city_key, {})
        city_out["lat"] = coords_for_city.get("lat")
        city_out["lon"] = coords_for_city.get("lon")

        # Live forecasts are fetched by the browser/phone IP to avoid Render/Open-Meteo 429s.
        city_out["forecast"] = {
            "high": None,
            "high_time": "",
            "low": None,
            "low_time": "",
            "members": 0,
            "source": "browser",
        }

        for market_type in ("high", "low"):
            series = cfg[market_type]
            try:
                raw = await _fetch_kalshi_series_markets(client, series)
                buckets = []

                for m in raw:
                    ticker = m.get("ticker") or ""
                    title = m.get("title") or ticker
                    bucket = _parse_temp_bucket(title, ticker)
                    px = _wx_market_prices(m)
                    score = px.get("midpoint") or _market_price_score(m)

                    # Keep this endpoint fast. Use the market summary fields first.
                    # Orderbook-per-bucket was too slow and made the dashboard spin.
                    yes_bid_eff = m.get("yes_bid")
                    yes_ask_eff = m.get("yes_ask")
                    no_bid = m.get("no_bid")
                    no_ask = m.get("no_ask")
                    last_price = m.get("last_price")

                    # If YES prices are missing, derive from NO side.
                    try:
                        if (yes_bid_eff is None or yes_bid_eff == 0) and no_ask:
                            yes_bid_eff = max(0, 100 - float(no_ask))
                        if (yes_ask_eff is None or yes_ask_eff == 0) and no_bid:
                            yes_ask_eff = max(0, 100 - float(no_bid))
                    except Exception:
                        pass

                    # Last fallback from last trade price.
                    if (yes_bid_eff is None or yes_bid_eff == 0) and last_price:
                        yes_bid_eff = last_price
                    if (yes_ask_eff is None or yes_ask_eff == 0) and last_price:
                        yes_ask_eff = last_price

                    if yes_bid_eff and yes_ask_eff:
                        score = (float(yes_bid_eff) + float(yes_ask_eff)) / 2.0
                    elif yes_bid_eff:
                        score = float(yes_bid_eff)
                    elif yes_ask_eff:
                        score = float(yes_ask_eff)
                    else:
                        score = _market_price_score(m)

                    buckets.append({
                        "ticker": ticker,
                        "title": title,
                        "label": bucket["label"],
                        "bucket_type": bucket["bucket_type"],
                        "low": bucket["low"],
                        "high": bucket["high"],
                        "threshold": bucket["threshold"],
                        "center": bucket["center"],
                        "yes_bid": px.get("yes_bid"),
                        "yes_ask": px.get("yes_ask"),
                        "yes_bid_effective": px.get("yes_bid"),
                        "yes_ask_effective": px.get("yes_ask"),
                        "no_bid": px.get("no_bid"),
                        "no_ask": px.get("no_ask"),
                        "last_price": px.get("last_price"),
                        "volume": px.get("volume") or 0,
                        "open_interest": px.get("open_interest") or 0,
                        "score": px.get("midpoint") or score,
                    })

                buckets = _wx_select_single_market_date(buckets)
                buckets.sort(key=lambda x: x["score"], reverse=True)
                city_out["markets"][market_type]["buckets"] = buckets

            except Exception as e:
                city_out["markets"][market_type]["error"] = str(e)

        cities_out.append(city_out)

    return {
        "ok": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cities": cities_out,
    }


MARKET_BOARD_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>WX Markets Prediction Tables</title>
  <style>
    :root {
      --bg:#0b0f14;
      --card:#111827;
      --text:#e5e7eb;
      --muted:#7b827f;
      --accent:#38bdf8;
      --amber:#facc15;
      --blue:#60a5fa;
      --bad:#ef4444;
      --border:#1f2b24;
      --cardBorder:rgba(148,163,184,.24);
    }

    * { box-sizing:border-box; }

    html, body {
      width:100%;
      max-width:100%;
      overflow-x:hidden;
    }

    body {
      margin:0;
      font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:var(--bg);
      color:var(--text);
    }

    header {
      padding:16px 12px 12px;
      border-bottom:1px solid var(--border);
      position:sticky;
      top:0;
      z-index:20;
      background:rgba(5,11,8,.97);
    }

    h1 {
      margin:0;
      font-size:21px;
    }

    .sub {
      color:var(--muted);
      font-size:13px;
      margin-top:5px;
      line-height:1.35;
    }

    .controls {
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:9px;
      margin-top:13px;
    }

    .controls button, .controls select {
      background:#0d1110;
      color:var(--text);
      border:1px solid #2a302d;
      border-radius:13px;
      padding:11px 9px;
      font-size:13px;
      font-weight:800;
      min-width:0;
    }

    .controls button.active {
      border-color:rgba(250,204,21,.75);
      color:#fde68a;
    }

    .statusline {
      margin-top:10px;
      color:var(--muted);
      font-size:12px;
    }

    main {
      padding:10px;
      width:100%;
      max-width:100%;
      overflow-x:hidden;
    }

    .section-title {
      color:var(--muted);
      font-size:13px;
      font-weight:900;
      letter-spacing:.12em;
      text-transform:uppercase;
      margin:10px 4px 14px;
    }

    .grid {
      display:grid;
      grid-template-columns:1fr;
      gap:14px;
      width:100%;
      max-width:100%;
    }

    .card {
      background:linear-gradient(180deg, rgba(17,24,39,.98), rgba(15,23,42,.98));
      border:1px solid var(--cardBorder);
      border-radius:18px;
      padding:14px;
      box-shadow:0 12px 24px rgba(0,0,0,.28);
      width:100%;
      max-width:100%;
      overflow:hidden;
    }

    .topline {
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:10px;
    }

    .city {
      font-size:20px;
      font-weight:900;
    }

    .market-meta {
      color:var(--muted);
      font-size:13px;
      margin-top:5px;
      letter-spacing:.04em;
      text-transform:uppercase;
    }

    .badge {
      display:inline-flex;
      align-items:center;
      gap:5px;
      padding:7px 9px;
      border-radius:10px;
      font-size:11px;
      font-weight:900;
      white-space:nowrap;
      border:1px solid rgba(56,189,248,.30);
      background:rgba(56,189,248,.10);
      color:var(--accent);
      flex:0 0 auto;
    }

    .badge.warn {
      border-color:rgba(250,204,21,.35);
      background:rgba(250,204,21,.10);
      color:#fde047;
    }

    .badge.cold {
      border-color:rgba(96,165,250,.35);
      background:rgba(96,165,250,.10);
      color:#93c5fd;
    }

    .badge.unknown {
      border-color:rgba(148,163,184,.25);
      background:rgba(148,163,184,.08);
      color:#94a3b8;
    }

    .big-temp {
      color:var(--accent);
      font-size:48px;
      line-height:1;
      font-weight:950;
      margin-top:18px;
      letter-spacing:-.04em;
    }

    .obs-code {
      color:rgba(148,163,184,.45);
      margin-top:8px;
      font-size:13px;
      letter-spacing:.08em;
    }

    .airline {
      color:#77a7ff;
      font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;
      font-size:14px;
      margin:16px 0 14px;
      line-height:1.35;
      white-space:normal;
      overflow-wrap:anywhere;
    }

    .bucket-list {
      display:grid;
      gap:8px;
      width:100%;
      max-width:100%;
    }

    .bucket-row {
      display:grid;
      grid-template-columns:minmax(0, 1fr) auto;
      gap:8px;
      align-items:center;
      min-height:46px;
      background:rgba(0,0,0,.62);
      border:1px solid rgba(148,163,184,.12);
      border-radius:12px;
      padding:10px;
      color:rgba(229,231,235,.45);
      width:100%;
      max-width:100%;
      overflow:hidden;
    }

    .bucket-row.leader {
      border-color:rgba(56,189,248,.45);
      color:var(--text);
    }

    .bucket-row.contender {
      color:var(--text);
    }

    .bucket-left {
      display:flex;
      align-items:center;
      gap:7px;
      min-width:0;
      overflow:hidden;
    }

    .bucket-role {
      font-size:10px;
      font-weight:950;
      color:var(--muted);
      white-space:nowrap;
      flex:0 0 auto;
    }

    .leader-label { color:var(--accent); }
    .contender-label { color:#fde047; }

    .bucket-label {
      font-size:15px;
      font-weight:950;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
      min-width:0;
    }

    .prices {
      display:grid;
      justify-items:end;
      gap:2px;
      font-size:13px;
      white-space:nowrap;
      min-width:60px;
      flex:0 0 auto;
    }

    .yes-price {
      color:var(--accent);
      font-weight:950;
    }

    .ask-price {
      color:rgba(229,231,235,.55);
      font-weight:800;
    }

    .price-caption {
      color:var(--muted);
      font-size:9px;
      letter-spacing:.05em;
      text-transform:uppercase;
    }

    .confirm {
      display:flex;
      align-items:flex-start;
      gap:8px;
      color:var(--accent);
      font-size:14px;
      margin-top:16px;
      line-height:1.35;
    }

    .divider {
      height:1px;
      background:rgba(148,163,184,.16);
      margin:14px 0;
    }

    .small-data {
      display:grid;
      gap:5px;
      color:var(--muted);
      font-size:12px;
    }

    .small-row {
      display:flex;
      justify-content:space-between;
      gap:10px;
    }

    .small-row strong {
      color:rgba(229,231,235,.82);
    }

    .kalshi-link {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      margin-top:16px;
      padding:12px 16px;
      border-radius:10px;
      color:#031018;
      background:var(--green);
      text-decoration:none;
      font-weight:950;
      width:fit-content;
      max-width:100%;
    }

    .error {
      color:var(--bad);
      font-size:12px;
      margin-top:10px;
    }

    @media (min-width: 760px) {
      main { padding:14px; }
      .grid { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
      .controls { display:flex; }
      .controls button, .controls select { min-width:150px; }
      .card { padding:16px; }
      .big-temp { font-size:52px; }
      .bucket-role { font-size:11px; }
      .bucket-label { font-size:16px; }
      .prices { font-size:14px; min-width:66px; }
    }
  </style>
</head>
<body>
<header>
  <h1>WX Markets Prediction Tables</h1>
  <div class="sub">
    Leader and contender are based on current Kalshi market favorability using YES bid/ask midpoint.
    Observed bucket is based on public NWS station observations.
  </div>
  <div class="controls">
    <button id="lowBtn" class="active" onclick="setType('low')">LOW Markets</button>
    <button id="highBtn" onclick="setType('high')">HIGH Markets</button>
    <button onclick="loadBoard(true)">Refresh</button>
    <select id="sortMode" onchange="render()">
      <option value="default">Sort: Default</option>
      <option value="leader">Highest favorite first</option>
      <option value="agree">Market/obs match first</option>
      <option value="city">City A-Z</option>
    </select>
  </div>
  <div class="statusline" id="status">Loading...</div>
</header>

<main>
  <div class="section-title" id="sectionTitle">LOW Markets</div>
  <div class="grid" id="grid"></div>
</main>

<script>
let board = [];
let marketType = "low";

function fmtTemp(v) {
  return (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${Math.round(v)}°F`;
}
function cents(v) {
  return (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${Math.round(v)}¢`;
}
function pricePair(b) {
  if (!b) return "— / —";

  const bid =
    (b.yes_bid_effective !== null && b.yes_bid_effective !== undefined && Number(b.yes_bid_effective) > 0) ? b.yes_bid_effective :
    (b.yes_bid !== null && b.yes_bid !== undefined && Number(b.yes_bid) > 0) ? b.yes_bid :
    null;

  const ask =
    (b.yes_ask_effective !== null && b.yes_ask_effective !== undefined && Number(b.yes_ask_effective) > 0) ? b.yes_ask_effective :
    (b.yes_ask !== null && b.yes_ask !== undefined && Number(b.yes_ask) > 0) ? b.yes_ask :
    null;

  return `${cents(bid)} / ${cents(ask)}`;
}
function fmtNum(v) {
  return (v === null || v === undefined) ? "—" : Number(v).toLocaleString();
}
function kalshiMarketUrl(ticker) {
  if (!ticker) return "https://kalshi.com/markets";
  return `https://kalshi.com/markets/${ticker}`;
}
  const WX_PRODUCT_CODES = {
    atl:{cli:"ATL", dsm:"FFC"},
    aus:{cli:"AUS", dsm:"EWX"},
    bos:{cli:"BOS", dsm:"BOX"},
    chicago:{cli:"MDW", dsm:"LOT"},
    dal:{cli:"DAL", dsm:"FWD"},
    dc:{cli:"DCA", dsm:"LWX"},
    denver:{cli:"DEN", dsm:"BOU"},
    hou:{cli:"HOU", dsm:"HGX"},
    los_angeles:{cli:"LAX", dsm:"LOX"},
    lv:{cli:"LAS", dsm:"VEF"},
    miami:{cli:"MIA", dsm:"MFL"},
    min:{cli:"MSP", dsm:"MPX"},
    nola:{cli:"MSY", dsm:"LIX"},
    nyc:{cli:"NYC", dsm:"OKX"},
    okc:{cli:"OKC", dsm:"OUN"},
    phi:{cli:"PHL", dsm:"PHI"},
    phx:{cli:"PHX", dsm:"PSR"},
    satx:{cli:"SAT", dsm:"EWX"},
    sea:{cli:"SEA", dsm:"SEW"},
    sf:{cli:"SFO", dsm:"MTR"}
  };

  function metarUrl(c) {
    const station = encodeURIComponent(c.station || "");
    return `https://portal-dev.accuweather.com/metar?station=${station}&hours=48&metartype=all`;
  }

  function cliUrl(c) {
    const code = WX_PRODUCT_CODES[c.city_key]?.cli || (c.station || "").replace(/^K/, "");
    return `https://forecast.weather.gov/product.php?site=NWS&product=CLI&issuedby=${encodeURIComponent(code)}`;
  }

  function dsmUrl(c) {
    const code = WX_PRODUCT_CODES[c.city_key]?.dsm || (c.station || "").replace(/^K/, "");
    return `https://forecast.weather.gov/product.php?site=NWS&product=DSM&issuedby=${encodeURIComponent(code)}`;
  }
  function metarUrl(c) {
    const station = encodeURIComponent(c.station || "");
    return `https://portal-dev.accuweather.com/metar?station=${station}&hours=48&metartype=all`;
  }
function obsUrl(c) {
  const end = new Date();
  const start = new Date(end.getTime() - 36 * 60 * 60 * 1000);
  return `https://api.weather.gov/stations/${c.station}/observations?start=${start.toISOString()}&end=${end.toISOString()}`;
}
function cToF(c) { return c * 9 / 5 + 32; }
function localYmd(iso, tz) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone:tz, year:"numeric", month:"2-digit", day:"2-digit"
  }).formatToParts(new Date(iso));
  const y = parts.find(p => p.type === "year")?.value;
  const m = parts.find(p => p.type === "month")?.value;
  const d = parts.find(p => p.type === "day")?.value;
  return `${y}-${m}-${d}`;
}
function localTime(iso, tz) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, {
    timeZone:tz, hour:"numeric", minute:"2-digit"
  });
}
function parseObs(c, data) {
  const features = data?.features || [];
  const today = localYmd(new Date().toISOString(), c.tz);
  const obs = [];

  for (const f of features) {
    const p = f.properties || {};
    const iso = p.timestamp;
    const tc = p.temperature?.value;
    if (!iso || tc === null || tc === undefined) continue;
    obs.push({time: iso, temp: cToF(Number(tc)), day: localYmd(iso, c.tz)});
  }

  obs.sort((a,b) => new Date(b.time) - new Date(a.time));
  const todayObs = obs.filter(o => o.day === today);

  let high = null;
  let low = null;

  for (const o of todayObs) {
    if (!high || o.temp > high.temp) high = o;
    if (!low || o.temp < low.temp) low = o;
  }

  return {
    obsTemp: obs[0]?.temp ?? null,
    obsTime: obs[0]?.time ?? "",
    obsHigh: high?.temp ?? null,
    obsHighTime: high?.time ?? "",
    obsLow: low?.temp ?? null,
    obsLowTime: low?.time ?? "",
  };
}
function bucketMatchesValue(b, value) {
  if (value === null || value === undefined) return false;
  const rounded = Math.round(value);
  if (b.bucket_type === "range") return rounded >= b.low && rounded <= b.high;
  if (b.bucket_type === "above") return rounded > b.threshold;
  if (b.bucket_type === "below") return rounded < b.threshold;
  return false;
}
function bucketCenter(b) {
  if (!b) return null;
  if (b.center !== null && b.center !== undefined) return b.center;
  if (b.low !== null && b.high !== null) return (b.low + b.high) / 2;
  if (b.low !== null) return b.low;
  if (b.high !== null) return b.high;
  return null;
}

function displayBucketLabel(b) {
  if (!b) return "—";
  if (b.bucket_type === "range" && b.low !== null && b.high !== null) {
    return `${b.low}-${b.high}°`;
  }
  if (b.bucket_type === "below" && b.threshold !== null && b.threshold !== undefined) {
    return `${Number(b.threshold) - 1}° or below`;
  }
  if (b.bucket_type === "above" && b.threshold !== null && b.threshold !== undefined) {
    return `${Number(b.threshold) + 1}° or above`;
  }

  let txt = String(b.label || b.title || b.ticker || "—");
  txt = txt.replace(/^>\s*(\d+)/, (_, n) => `${Number(n) + 1}° or above`);
  txt = txt.replace(/^<\s*(\d+)/, (_, n) => `${Number(n) - 1}° or below`);
  txt = txt.replace(/>\s*(\d+)/, (_, n) => `${Number(n) + 1}° or above`);
  txt = txt.replace(/<\s*(\d+)/, (_, n) => `${Number(n) - 1}° or below`);
  return txt;
}
function marketLeader(buckets) {
  return (buckets || [])[0] || null;
}
function contender(buckets) {
  return (buckets || [])[1] || null;
}
function observedLeader(c, buckets) {
  const value = marketType === "high" ? c.obs?.obsHigh : c.obs?.obsLow;
  return (buckets || []).find(b => bucketMatchesValue(b, value)) || null;
}
function statusFor(leader, obsLead) {
  if (!leader || !obsLead) return {text:"WAITING", cls:"unknown", message:"Waiting for public obs to match a bucket."};
  if (leader.ticker === obsLead.ticker) {
    return {text:"MARKET/OBS MATCH", cls:"", message:"Public observed temp currently matches the market leader bucket."};
  }

  const lc = bucketCenter(leader);
  const oc = bucketCenter(obsLead);

  if (lc !== null && oc !== null) {
    if (lc > oc) return {text:"MARKET HOTTER", cls:"warn", message:"Market favorite is warmer than the current observed bucket."};
    if (lc < oc) return {text:"MARKET COLDER", cls:"cold", message:"Market favorite is colder than the current observed bucket."};
  }

  return {text:"MISMATCH", cls:"unknown", message:"Market favorite and observed bucket do not currently match."};
}
function setType(t) {
  marketType = t;
  document.getElementById("lowBtn").classList.toggle("active", t === "low");
  document.getElementById("highBtn").classList.toggle("active", t === "high");
  document.getElementById("sectionTitle").textContent = t === "low" ? "LOW Markets" : "HIGH Markets";
  render();
}
function bucketRow(b, idx, obsLead) {
  const isLeader = idx === 0;
  const isContender = idx === 1;
  const role = isLeader ? "👑 LEADER" : isContender ? "⚔️ CONTENDER" : "";
  const roleClass = isLeader ? "leader-label" : isContender ? "contender-label" : "";
  const rowClass = isLeader ? "leader" : isContender ? "contender" : "";
  const obsMark = obsLead && obsLead.ticker === b.ticker ? " ✅" : "";

  return `
    <div class="bucket-row ${rowClass}">
      <div class="bucket-left">
        ${role ? `<span class="bucket-role ${roleClass}">${role}</span>` : ""}
        <span class="bucket-label">${displayBucketLabel(b)}${obsMark}</span>
      </div>
      <div class="prices">
        <div>
          <span class="yes-price">${pricePair(b).split(" / ")[0]}</span>
          <span class="ask-price"> / ${pricePair(b).split(" / ")[1]}</span>
        </div>
        <div class="price-caption">YES bid / ask</div>
      </div>
    </div>
  `;
}


function marketEnsembleUrl(c) {
  return `https://ensemble-api.open-meteo.com/v1/ensemble?latitude=${c.lat}&longitude=${c.lon}&hourly=temperature_2m&models=gfs_seamless&forecast_days=3&temperature_unit=fahrenheit&timezone=UTC`;
}

function marketTempKeys(hourly) {
  return Object.keys(hourly || {}).filter(k => k === "temperature_2m" || k.startsWith("temperature_2m_member"));
}

function marketMean(vals) {
  if (!vals.length) return null;
  return vals.reduce((a,b) => a+b, 0) / vals.length;
}

function marketLocalYmd(isoTime, tz) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(new Date(isoTime));

  const y = parts.find(p => p.type === "year")?.value;
  const m = parts.find(p => p.type === "month")?.value;
  const d = parts.find(p => p.type === "day")?.value;
  return `${y}-${m}-${d}`;
}

function emptyMarketForecast(reason="") {
  return {
    high: null,
    high_time: "",
    low: null,
    low_time: "",
    members: 0,
    error: reason
  };
}

function parseMarketForecast(c, data) {
  const hourly = data?.hourly || {};
  const times = hourly.time || [];
  const keys = marketTempKeys(hourly);

  if (!times.length || !keys.length) {
    return emptyMarketForecast("No forecast temps returned");
  }

  const todayYmd = marketLocalYmd(new Date().toISOString(), c.tz);
  let high = null;
  let low = null;
  let highTime = "";
  let lowTime = "";
  let maxMembers = 0;

  times.forEach((t, i) => {
    if (marketLocalYmd(t, c.tz) !== todayYmd) return;

    const vals = keys
      .map(k => hourly[k]?.[i])
      .filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v)))
      .map(Number);

    if (!vals.length) return;

    maxMembers = Math.max(maxMembers, vals.length);
    const avg = marketMean(vals);

    if (avg === null || Number.isNaN(avg)) return;

    if (high === null || avg > high) {
      high = avg;
      highTime = t;
    }

    if (low === null || avg < low) {
      low = avg;
      lowTime = t;
    }
  });

  return {
    high: high === null ? null : Math.round(high),
    high_time: highTime,
    low: low === null ? null : Math.round(low),
    low_time: lowTime,
    members: maxMembers,
    source: "browser"
  };
}

async function fetchMarketForecast(c) {
  if (!c.lat || !c.lon) {
    return emptyMarketForecast("Missing city coordinates");
  }

  const res = await fetch(marketEnsembleUrl(c), {cache:"no-store"});

  if (!res.ok) {
    return emptyMarketForecast(`Ensemble HTTP ${res.status}`);
  }

  const data = await res.json();
  return parseMarketForecast(c, data);
}


function render() {
  const mode = document.getElementById("sortMode").value;
  let rows = [...board];

  if (mode === "city") rows.sort((a,b) => a.name.localeCompare(b.name));

  if (mode === "leader") {
    rows.sort((a,b) => {
      const bl = marketLeader(b.markets[marketType].buckets);
      const al = marketLeader(a.markets[marketType].buckets);
      return (bl?.score ?? 0) - (al?.score ?? 0);
    });
  }

  if (mode === "agree") {
    rows.sort((a,b) => {
      const ab = a.markets[marketType].buckets || [];
      const bb = b.markets[marketType].buckets || [];
      const as = statusFor(marketLeader(ab), observedLeader(a, ab)).text === "MARKET/OBS MATCH" ? 0 : 1;
      const bs = statusFor(marketLeader(bb), observedLeader(b, bb)).text === "MARKET/OBS MATCH" ? 0 : 1;
      return as - bs;
    });
  }

  document.getElementById("grid").innerHTML = rows.map(c => {
    const m = c.markets[marketType] || {};
    const buckets = m.buckets || [];
    const lead = marketLeader(buckets);
    const cont = contender(buckets);
    const obsLead = observedLeader(c, buckets);
    const st = statusFor(lead, obsLead);

    const obsValue = marketType === "high" ? c.obs?.obsHigh : c.obs?.obsLow;
    const obsTime = marketType === "high" ? c.obs?.obsHighTime : c.obs?.obsLowTime;
    const observedLabel = marketType === "high" ? "Observed HIGH" : "Observed LOW";
    const forecastValue = marketType === "high" ? c.forecast?.high : c.forecast?.low;
    const forecastTime = marketType === "high" ? c.forecast?.high_time : c.forecast?.low_time;
    const marketLabel = marketType.toUpperCase();
    const rest = buckets.slice(2, 6);

    return `
      <div class="card">
        <div class="topline">
          <div>
            <div class="city">${c.name}</div>
            <div class="market-meta">${marketLabel} · ${c.station}</div>
          </div>
          <div class="badge ${st.cls}">${st.text}</div>
        </div>

        <div class="big-temp">${fmtTemp(obsValue)}</div>
        <div class="obs-code">Current observed: ${fmtTemp(c.obs?.obsTemp)}${c.obs?.obsTime ? " at " + localTime(c.obs.obsTime, c.tz) : ""}</div>

        <div class="airline">📊 ${observedLabel}: ${fmtTemp(obsValue)}${obsTime ? " at " + localTime(obsTime, c.tz) : ""} ✅</div>
        <div class="airline">🔮 Forecasted ${observedLabel.replace("Observed ", "")}: ${fmtTemp(forecastValue)}${forecastTime ? " around " + localTime(forecastTime, c.tz) : ""}${c.forecast?.members ? " · N=" + c.forecast.members : ""}${c.forecast?.error ? " · " + c.forecast.error : ""}</div>

        <div class="bucket-list">
          ${lead ? bucketRow(lead, 0, obsLead) : ""}
          ${cont ? bucketRow(cont, 1, obsLead) : ""}
          ${rest.map((b, i) => bucketRow(b, i + 2, obsLead)).join("")}
        </div>

        <div class="confirm">✅ <span>${st.message}</span></div>

        <div class="divider"></div>

        <div class="small-data">
          <div class="small-row"><span>Observed bucket</span><strong>${obsLead ? displayBucketLabel(obsLead) : "—"}</strong></div>
          <div class="small-row"><span>Observed time</span><strong>${obsTime ? localTime(obsTime, c.tz) : "—"}</strong></div>
          <div class="small-row"><span>Forecasted ${marketLabel}</span><strong>${fmtTemp(forecastValue)}${forecastTime ? " around " + localTime(forecastTime, c.tz) : ""}</strong></div>
          <div class="small-row"><span>Leader YES bid / ask</span><strong>${lead ? pricePair(lead) : "—"}</strong></div>
          <div class="small-row"><span>Contender YES bid / ask</span><strong>${cont ? pricePair(cont) : "—"}</strong></div>
          <div class="small-row"><span>Current observed temp</span><strong>${fmtTemp(c.obs?.obsTemp)}${c.obs?.obsTime ? " at " + localTime(c.obs.obsTime, c.tz) : ""}</strong></div>
            <div class="small-row"><span>Forecast Accuracy</span><strong>collecting data</strong></div>
            <div class="small-row"><span>Yesterday forecast</span><strong>waiting for saved snapshot</strong></div>
            <div class="small-row"><span>7-day avg miss</span><strong>collecting data</strong></div>
        </div>

        ${m.error ? `<div class="error">${m.error}</div>` : ""}

        <a class="kalshi-link" href="${kalshiMarketUrl(lead?.ticker)}" target="_blank" rel="noopener">🔗 Trade on Kalshi</a>
        <a class="kalshi-link" href="${cliUrl(c)}" target="_blank" rel="noopener">🌡️ NWS CLI</a>
        <a class="kalshi-link" href="${dsmUrl(c)}" target="_blank" rel="noopener">📈 NWS DSM</a>
        <a class="kalshi-link" href="${metarUrl(c)}" target="_blank" rel="noopener">🧾 Hourly METAR</a>
      </div>
    `;
  }).join("");
}
async function loadBoard(manual=false) {
  const status = document.getElementById("status");
  status.textContent = manual ? "Refreshing..." : "Loading...";

  const res = await fetch("/api/kalshi/market-board", {cache:"no-store"});
  const data = await res.json();

  if (!data.ok) {
    status.textContent = data.error || "Failed to load market board";
    return;
  }

  const rows = [];

  for (const c of (data.cities || [])) {
    let obs = {};
    let forecast = c.forecast || emptyMarketForecast();

    try {
      const obsRes = await fetch(obsUrl(c), {cache:"no-store"});
      if (obsRes.ok) {
        const obsData = await obsRes.json();
        obs = parseObs(c, obsData);
      }
    } catch {}

    try {
      forecast = await fetchMarketForecast(c);
    } catch (e) {
      forecast = emptyMarketForecast(String(e.message || e));
    }

    rows.push({...c, obs, forecast});

    // Avoid hammering Open-Meteo from the browser with 20 simultaneous requests.
    await new Promise(resolve => setTimeout(resolve, 175));
  }

  board = rows;
  render();
  status.textContent = `Last refresh: ${new Date().toLocaleTimeString()} · ${rows.length} cities`;
}
setType("low");
loadBoard();
setInterval(() => loadBoard(false), 5 * 60 * 1000);
</script>
</body>
</html>
"""

@app.get("/market-board", response_class=HTMLResponse)
async def market_board_dashboard():
    return HTMLResponse(MARKET_BOARD_DASHBOARD_HTML)

@app.get("/api/debug/kalshi-series/{series_ticker}")
async def debug_kalshi_series(series_ticker: str):
    """
    Debug endpoint to inspect raw Kalshi market fields for one series.
    Use this to see the exact bid/ask field names returned by Kalshi.
    """
    from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present

    if not kalshi_credentials_present():
        return {
            "ok": False,
            "error": "Kalshi credentials are not configured on this service.",
        }

    client = KalshiClient()

    data = await client.get_markets({
        "series_ticker": series_ticker,
        "status": "open",
        "limit": 10,
    })

    markets = data.get("markets", []) or []

    samples = []
    for m in markets[:6]:
        samples.append({
            "ticker": m.get("ticker"),
            "title": m.get("title"),
            "subtitle": m.get("subtitle"),
            "yes_bid": px.get("yes_bid"),
            "yes_ask": px.get("yes_ask"),
            "no_bid": px.get("no_bid"),
            "no_ask": px.get("no_ask"),
            "last_price": px.get("last_price"),
            "open_interest": m.get("open_interest"),
            "volume": m.get("volume"),
            "liquidity": m.get("liquidity"),
            "price_keys_present": sorted([
                k for k in m.keys()
                if "price" in k.lower()
                or "bid" in k.lower()
                or "ask" in k.lower()
                or "volume" in k.lower()
                or "liquidity" in k.lower()
            ]),
            "all_keys": sorted(list(m.keys())),
        })

    return {
        "ok": True,
        "series_ticker": series_ticker,
        "count": len(markets),
        "samples": samples,
    }

