import math
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Crypto Trade Decision Engine", layout="wide")

st.title("Crypto Trade Decision Engine")
st.caption("Conservative, trend-aware, risk-aware decision helper for liquid crypto majors.")


# -----------------------------
# Helpers
# -----------------------------
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def round_to_step(value, step):
    if step <= 0:
        return value
    return math.floor(value / step) * step


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def compute_confidence(
    trend_ok,
    rsi_ok,
    pullback_ok,
    vol_ok,
    data_fresh,
    spread_ok,
    confluence_count
):
    score = 0.0

    if trend_ok:
        score += 0.30
    if rsi_ok:
        score += 0.18
    if pullback_ok:
        score += 0.18
    if vol_ok:
        score += 0.14
    if data_fresh:
        score += 0.10
    if spread_ok:
        score += 0.10

    if confluence_count >= 3:
        score += 0.05

    return clamp(score, 0.0, 1.0)


def build_skip_reason(
    data_fresh,
    trend_clear,
    confluence_count,
    vol_ok,
    spread_ok,
    qty_ok,
    max_positions_ok,
):
    reasons = []

    if not data_fresh:
        reasons.append("data stale")
    if not trend_clear:
        reasons.append("mid-term trend unclear")
    if confluence_count < 2:
        reasons.append("not enough confluence")
    if not vol_ok:
        reasons.append("realized volatility too high")
    if not spread_ok:
        reasons.append("spread/slippage too large")
    if not qty_ok:
        reasons.append("computed quantity below min_trade_qty")
    if not max_positions_ok:
        reasons.append("max_positions reached")

    return ", ".join(reasons) if reasons else "no valid setup"


def calculate_trade_decision(
    symbol,
    side_allowed,
    nav,
    equity,
    buying_power,
    cap_factor,
    max_positions,
    current_open_positions,
    mark_price,
    ema20,
    ema100,
    rsi,
    atr,
    realized_vol_pct,
    spread_pct,
    support,
    resistance,
    min_trade_qty,
    max_order_qty,
    quantity_step,
    estimated_fee_rate,
    rr_min,
    interval_minutes,
    candle_age_minutes,
):
    now_utc = datetime.now(timezone.utc).isoformat()

    # ---- Guards ----
    data_fresh = candle_age_minutes <= (2 * interval_minutes)
    max_positions_ok = current_open_positions < max_positions

    trend_up = ema20 > ema100
    trend_down = ema20 < ema100
    trend_clear = trend_up or trend_down

    # User prompt prefers liquid majors and trend aligned.
    # Default action bias: only take long if uptrend, short if downtrend and allowed.
    if trend_up:
        trend_direction = "long"
    elif trend_down:
        trend_direction = "short"
    else:
        trend_direction = "flat"

    rsi_long_ok = 40 <= rsi <= 62
    rsi_short_ok = 38 <= rsi <= 60

    # Pullback heuristics
    near_ema20 = abs(mark_price - ema20) / mark_price <= 0.0075  # 0.75%
    near_support = support > 0 and abs(mark_price - support) / mark_price <= 0.01
    near_resistance = resistance > 0 and abs(mark_price - resistance) / mark_price <= 0.01

    pullback_long_ok = mark_price >= ema20 * 0.995 and (near_ema20 or near_support)
    pullback_short_ok = mark_price <= ema20 * 1.005 and (near_ema20 or near_resistance)

    vol_ok = realized_vol_pct <= 4.0
    spread_ok = spread_pct <= 0.20

    # Confluence count
    if trend_direction == "long":
        confluence_count = sum([trend_up, rsi_long_ok, pullback_long_ok, vol_ok])
        rsi_ok = rsi_long_ok
        pullback_ok = pullback_long_ok
    elif trend_direction == "short":
        confluence_count = sum([trend_down, rsi_short_ok, pullback_short_ok, vol_ok])
        rsi_ok = rsi_short_ok
        pullback_ok = pullback_short_ok
    else:
        confluence_count = 0
        rsi_ok = False
        pullback_ok = False

    confidence = compute_confidence(
        trend_ok=trend_clear,
        rsi_ok=rsi_ok,
        pullback_ok=pullback_ok,
        vol_ok=vol_ok,
        data_fresh=data_fresh,
        spread_ok=spread_ok,
        confluence_count=confluence_count
    )

    # Reduce size when confidence low
    confidence_size_factor = 1.0 if confidence >= 0.5 else max(0.25, confidence)

    # Reduce size when vol elevated but not invalid
    vol_size_factor = 1.0
    if 3.0 < realized_vol_pct <= 4.0:
        vol_size_factor = 0.65

    raw_notional = min(cap_factor * equity, buying_power)
    sized_notional = raw_notional * confidence_size_factor * vol_size_factor

    # ---- Stops / targets ----
    if trend_direction == "long":
        stop_price = min(
            support if support > 0 else mark_price - (1.5 * atr),
            mark_price - (1.2 * atr)
        )
        risk_per_unit = mark_price - stop_price
        target_price = mark_price + (risk_per_unit * rr_min)
        entry_type = "tight_limit_pullback" if pullback_ok else "skip"

    elif trend_direction == "short":
        stop_price = max(
            resistance if resistance > 0 else mark_price + (1.5 * atr),
            mark_price + (1.2 * atr)
        )
        risk_per_unit = stop_price - mark_price
        target_price = mark_price - (risk_per_unit * rr_min)
        entry_type = "tight_limit_pullback" if pullback_ok else "skip"
    else:
        stop_price = 0.0
        target_price = 0.0
        risk_per_unit = 0.0
        entry_type = "skip"

    # Additional guard
    if risk_per_unit <= 0:
        qty = 0.0
    else:
        qty = sized_notional / mark_price

    qty = round_to_step(qty, quantity_step)
    qty = clamp(qty, 0.0, max_order_qty)
    qty_ok = qty >= min_trade_qty

    estimated_notional = qty * mark_price
    estimated_fee = estimated_notional * estimated_fee_rate

    allowed_short = side_allowed in ["both", "short_only"]
    allowed_long = side_allowed in ["both", "long_only"]

    valid_setup = (
        data_fresh
        and trend_clear
        and confluence_count >= 2
        and vol_ok
        and spread_ok
        and qty_ok
        and max_positions_ok
        and (
            (trend_direction == "long" and allowed_long)
            or (trend_direction == "short" and allowed_short)
        )
    )

    if not valid_setup:
        action = "skip"
        rationale = build_skip_reason(
            data_fresh=data_fresh,
            trend_clear=trend_clear,
            confluence_count=confluence_count,
            vol_ok=vol_ok,
            spread_ok=spread_ok,
            qty_ok=qty_ok,
            max_positions_ok=max_positions_ok,
        )
    else:
        action = "buy" if trend_direction == "long" else "sell"
        rationale = (
            f"Trend aligned ({'EMA20>EMA100' if trend_direction == 'long' else 'EMA20<EMA100'}), "
            f"RSI supportive, pullback near EMA/support-resistance, "
            f"RR >= 1:{rr_min:.1f}, conservative size applied."
        )

    # Meta
    result = {
        "compose_id": f"{symbol}-{int(datetime.now().timestamp())}",
        "strategy_id": "crypto_conservative_pullback_v1",
        "timestamp": now_utc,
        "symbol": symbol,
        "action": action,
        "trend_direction": trend_direction,
        "entry_type": entry_type if action != "skip" else "skip",
        "mark_price": round(mark_price, 4),
        "entry_price": round(mark_price, 4) if action != "skip" else None,
        "stop_price": round(stop_price, 4) if action != "skip" else None,
        "target_price": round(target_price, 4) if action != "skip" else None,
        "quantity": round(qty, 8) if action != "skip" else 0.0,
        "estimated_notional": round(estimated_notional, 4),
        "estimated_fee": round(estimated_fee, 4),
        "confidence_score": round(confidence, 4),
        "confluence_count": int(confluence_count),
        "rationale": rationale,
        "signals": {
            "trend_clear": trend_clear,
            "trend_up": trend_up,
            "trend_down": trend_down,
            "rsi_ok": rsi_ok,
            "pullback_ok": pullback_ok,
            "vol_ok": vol_ok,
            "spread_ok": spread_ok,
            "data_fresh": data_fresh,
        },
        "trade_meta": {
            "planned_rr": rr_min,
            "nav": nav,
            "equity": equity,
            "buying_power": buying_power,
            "cap_factor": cap_factor,
            "max_positions": max_positions,
            "current_open_positions": current_open_positions,
            "estimated_fee_rate": estimated_fee_rate,
        }
    }

    return result


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Config")

symbol = st.sidebar.selectbox("Symbol", ["BTC-USD", "ETH-USD"])
side_allowed = st.sidebar.selectbox("Allowed Side", ["both", "long_only", "short_only"], index=0)

nav = st.sidebar.number_input("NAV", min_value=0.0, value=10000.0, step=100.0)
equity = st.sidebar.number_input("Equity", min_value=0.0, value=10000.0, step=100.0)
buying_power = st.sidebar.number_input("Available Buying Power", min_value=0.0, value=5000.0, step=100.0)

cap_factor = st.sidebar.slider("cap_factor", min_value=0.001, max_value=0.05, value=0.02, step=0.001)
max_positions = st.sidebar.number_input("max_positions", min_value=1, value=2, step=1)
current_open_positions = st.sidebar.number_input("current_open_positions", min_value=0, value=0, step=1)

rr_min = st.sidebar.slider("Min Risk:Reward", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
estimated_fee_rate = st.sidebar.number_input("Estimated Fee Rate", min_value=0.0, value=0.001, step=0.0001, format="%.4f")

st.sidebar.header("Market Data")

mark_price = st.sidebar.number_input("Mark Price", min_value=0.0, value=85000.0, step=10.0)
ema20 = st.sidebar.number_input("EMA 20", min_value=0.0, value=84800.0, step=10.0)
ema100 = st.sidebar.number_input("EMA 100", min_value=0.0, value=84000.0, step=10.0)
rsi = st.sidebar.slider("RSI", min_value=0.0, max_value=100.0, value=48.0, step=0.1)
atr = st.sidebar.number_input("ATR", min_value=0.0, value=600.0, step=10.0)

realized_vol_pct = st.sidebar.slider("Realized Volatility %", min_value=0.0, max_value=10.0, value=2.2, step=0.1)
spread_pct = st.sidebar.slider("Spread %", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

support = st.sidebar.number_input("Support", min_value=0.0, value=84400.0, step=10.0)
resistance = st.sidebar.number_input("Resistance", min_value=0.0, value=85800.0, step=10.0)

st.sidebar.header("Venue Constraints")

min_trade_qty = st.sidebar.number_input("min_trade_qty", min_value=0.0, value=0.001, step=0.001, format="%.6f")
max_order_qty = st.sidebar.number_input("max_order_qty", min_value=0.0, value=2.0, step=0.01, format="%.6f")
quantity_step = st.sidebar.number_input("quantity_step", min_value=0.000001, value=0.001, step=0.000001, format="%.6f")

interval_minutes = st.sidebar.number_input("Interval Minutes", min_value=1, value=15, step=1)
candle_age_minutes = st.sidebar.number_input("Last Candle Age Minutes", min_value=0, value=10, step=1)


# -----------------------------
# Run Calculation
# -----------------------------
if st.button("Calculate Decision", type="primary"):
    result = calculate_trade_decision(
        symbol=symbol,
        side_allowed=side_allowed,
        nav=nav,
        equity=equity,
        buying_power=buying_power,
        cap_factor=cap_factor,
        max_positions=max_positions,
        current_open_positions=current_open_positions,
        mark_price=mark_price,
        ema20=ema20,
        ema100=ema100,
        rsi=rsi,
        atr=atr,
        realized_vol_pct=realized_vol_pct,
        spread_pct=spread_pct,
        support=support,
        resistance=resistance,
        min_trade_qty=min_trade_qty,
        max_order_qty=max_order_qty,
        quantity_step=quantity_step,
        estimated_fee_rate=estimated_fee_rate,
        rr_min=rr_min,
        interval_minutes=interval_minutes,
        candle_age_minutes=candle_age_minutes,
    )

    action = result["action"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Action", action.upper())
    col2.metric("Confidence", f'{result["confidence_score"]:.2f}')
    col3.metric("Quantity", result["quantity"])
    col4.metric("Estimated Notional", result["estimated_notional"])

    st.subheader("Rationale")
    st.write(result["rationale"])

    st.subheader("Trade Plan")
    trade_df = pd.DataFrame([{
        "Symbol": result["symbol"],
        "Action": result["action"],
        "Entry Type": result["entry_type"],
        "Entry": result["entry_price"],
        "Stop": result["stop_price"],
        "Target": result["target_price"],
        "Quantity": result["quantity"],
        "Estimated Notional": result["estimated_notional"],
        "Estimated Fee": result["estimated_fee"],
        "Confidence": result["confidence_score"],
        "Timestamp": result["timestamp"],
    }])
    st.dataframe(trade_df, use_container_width=True)

    st.subheader("Signals")
    signals_df = pd.DataFrame([result["signals"]])
    st.dataframe(signals_df, use_container_width=True)

    st.subheader("Raw JSON")
    st.json(result)

else:
    st.info("Enter your inputs from the sidebar, then click 'Calculate Decision'.")