"""
SentinelFi AI — Telegram Trading Agent
Bot: @SentinelFiAIBot

Features
- Bitget public market data (spot tickers + candles)
- Technical analysis: EMA20 / EMA50 / RSI(14)
- Trade setup: Entry, Take-Profit, Stop-Loss, Risk/Reward
- Confidence scoring + optional LLM commentary (OpenAI-compatible)
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ---------- Config ----------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "").strip()
BITGET_SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "").strip()
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "meta-llama/llama-3.1-8b-instruct:free").strip()

BITGET_BASE = "https://api.bitget.com"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("sentinel")


# ---------- Bitget Client ----------
class BitgetClient:
    """Lightweight async client for Bitget public spot endpoints."""

    def __init__(self) -> None:
        self.http = httpx.AsyncClient(
            base_url=BITGET_BASE,
            timeout=15.0,
            headers={"User-Agent": "SentinelFi-AI/1.0"},
        )

    async def close(self) -> None:
        await self.http.aclose()

    @staticmethod
    def normalize(symbol: str) -> str:
        """Normalize 'BTC' or 'btc-usdt' to Bitget spot v2 format 'BTCUSDT'."""
        s = symbol.upper().replace("-", "").replace("/", "").replace("_", "")
        if not s.endswith("USDT"):
            s = f"{s}USDT"
        return s

    async def ticker(self, symbol: str) -> dict:
        sym = self.normalize(symbol)
        r = await self.http.get("/api/v2/spot/market/tickers", params={"symbol": sym})
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            raise ValueError(f"No ticker data for {sym}")
        return data[0]

    async def candles(self, symbol: str, granularity: str = "1h", limit: int = 200) -> list[list]:
        sym = self.normalize(symbol)
        r = await self.http.get(
            "/api/v2/spot/market/candles",
            params={"symbol": sym, "granularity": granularity, "limit": str(limit)},
        )
        r.raise_for_status()
        return r.json().get("data") or []

    async def all_tickers(self) -> list[dict]:
        r = await self.http.get("/api/v2/spot/market/tickers")
        r.raise_for_status()
        return r.json().get("data") or []


# ---------- Technical Analysis ----------
def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.array([])
    k = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) < period + 1:
        return float("nan")
    diffs = np.diff(values)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(diffs)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def atr_proxy(closes: np.ndarray, period: int = 14) -> float:
    """ATR-style volatility proxy from close-to-close moves (no H/L parsing variance)."""
    if len(closes) < period + 1:
        return float(np.std(closes) or closes[-1] * 0.01)
    diffs = np.abs(np.diff(closes[-(period + 1):]))
    return float(diffs.mean())


def analyze(closes: np.ndarray, highs: Optional[np.ndarray] = None, lows: Optional[np.ndarray] = None) -> dict:
    """Run EMA20/EMA50/RSI analysis and produce a trade setup with reasoning."""
    price = float(closes[-1])
    ema20_arr = ema(closes, 20)
    ema50_arr = ema(closes, 50)
    e20 = float(ema20_arr[-1]) if ema20_arr.size else float("nan")
    e50 = float(ema50_arr[-1]) if ema50_arr.size else float("nan")
    rsi14 = rsi(closes, 14)
    vol = atr_proxy(closes, 14)

    # 24-bar (1D on 1H tf) momentum
    lookback = min(24, len(closes) - 1)
    chg_24 = (price / float(closes[-lookback - 1]) - 1.0) * 100 if lookback > 0 else 0.0

    # EMA slope (last 5 bars on EMA20) for momentum confirmation
    if ema20_arr.size >= 6:
        ema_slope = (ema20_arr[-1] - ema20_arr[-6]) / ema20_arr[-6] * 100
    else:
        ema_slope = 0.0

    # Trend logic
    bullish = e20 > e50 and price > e20
    bearish = e20 < e50 and price < e20
    direction = "LONG" if bullish else "SHORT" if bearish else "NEUTRAL"

    # ---- Bullish / bearish factor accounting ----
    bull_factors: list[str] = []
    bear_factors: list[str] = []

    if e20 > e50:
        bull_factors.append(f"EMA20 ({fmt_price(e20)}) above EMA50 ({fmt_price(e50)}) — uptrend structure")
    else:
        bear_factors.append(f"EMA20 ({fmt_price(e20)}) below EMA50 ({fmt_price(e50)}) — downtrend structure")

    if price > e20:
        bull_factors.append("Price trading above EMA20 — short-term momentum positive")
    else:
        bear_factors.append("Price trading below EMA20 — short-term momentum negative")

    if rsi14 >= 70:
        bear_factors.append(f"RSI {rsi14:.1f} in overbought zone — pullback risk elevated")
    elif rsi14 >= 55:
        bull_factors.append(f"RSI {rsi14:.1f} in bullish zone (55–70)")
    elif rsi14 <= 30:
        bull_factors.append(f"RSI {rsi14:.1f} oversold — potential mean-reversion bounce")
    elif rsi14 <= 45:
        bear_factors.append(f"RSI {rsi14:.1f} in bearish zone (30–45)")
    else:
        (bull_factors if direction == "LONG" else bear_factors if direction == "SHORT" else bull_factors).append(
            f"RSI {rsi14:.1f} neutral — no extreme reading"
        )

    if ema_slope > 0.3:
        bull_factors.append(f"EMA20 sloping up ({ema_slope:+.2f}% over 5h) — momentum building")
    elif ema_slope < -0.3:
        bear_factors.append(f"EMA20 sloping down ({ema_slope:+.2f}% over 5h) — momentum fading")

    if chg_24 > 3:
        bull_factors.append(f"24h change {chg_24:+.2f}% — strong buying pressure")
    elif chg_24 < -3:
        bear_factors.append(f"24h change {chg_24:+.2f}% — strong selling pressure")

    # ---- Market structure (higher highs / lower lows) ----
    structure = "Sideways / consolidating"
    if highs is not None and lows is not None and len(highs) >= 40:
        # Compare last 20 bars vs prior 20 bars
        recent_hi, prior_hi = float(highs[-20:].max()), float(highs[-40:-20].max())
        recent_lo, prior_lo = float(lows[-20:].min()), float(lows[-40:-20].min())
        hh = recent_hi > prior_hi
        hl = recent_lo > prior_lo
        lh = recent_hi < prior_hi
        ll = recent_lo < prior_lo
        if hh and hl:
            structure = "Higher highs + higher lows — confirmed uptrend"
        elif lh and ll:
            structure = "Lower highs + lower lows — confirmed downtrend"
        elif hh and ll:
            structure = "Expanding range — volatility regime"
        elif lh and hl:
            structure = "Contracting range — compression / breakout pending"

    # ---- Confidence scoring (0–100) ----
    score = 50.0
    if bullish:
        score += 20
        if 45 <= rsi14 <= 70:
            score += 15
        if rsi14 > 70:
            score -= 10
    elif bearish:
        score += 20
        if 30 <= rsi14 <= 55:
            score += 15
        if rsi14 < 30:
            score -= 10
    # Penalize distance from EMA20 (mean-reversion risk)
    dist = abs(price - e20) / e20 if e20 else 0
    score -= min(15, dist * 100)
    # Reward EMA slope agreement
    if (direction == "LONG" and ema_slope > 0) or (direction == "SHORT" and ema_slope < 0):
        score += 5
    score = max(0.0, min(100.0, score))

    # ---- Risk level ----
    vol_pct = vol / price * 100 if price else 0
    risk_score = 0
    if vol_pct > 2.0:
        risk_score += 2
    elif vol_pct > 1.0:
        risk_score += 1
    if rsi14 >= 75 or rsi14 <= 25:
        risk_score += 2
    elif rsi14 >= 70 or rsi14 <= 30:
        risk_score += 1
    if dist * 100 > 2.0:
        risk_score += 1
    if direction == "NEUTRAL":
        risk_score += 1
    risk_level = "Low" if risk_score <= 1 else "Medium" if risk_score <= 3 else "High"

    # ---- Reasoning sentence ----
    if direction == "LONG":
        reasoning = (
            f"Trend filter is bullish (EMA20>EMA50 and price>EMA20) with RSI at {rsi14:.1f}. "
            f"Volatility ≈ {vol_pct:.2f}% supports a {risk_level.lower()}-risk long entry."
        )
    elif direction == "SHORT":
        reasoning = (
            f"Trend filter is bearish (EMA20<EMA50 and price<EMA20) with RSI at {rsi14:.1f}. "
            f"Volatility ≈ {vol_pct:.2f}% supports a {risk_level.lower()}-risk short entry."
        )
    else:
        reasoning = (
            f"Trend filter is mixed — price is on the wrong side of EMA20 relative to EMA50. "
            f"RSI {rsi14:.1f}, volatility {vol_pct:.2f}%. Wait for confirmation before committing."
        )

    # ---- Trade setup (volatility-scaled) ----
    if direction == "LONG":
        entry = price
        sl = entry - 1.5 * vol
        tp = entry + 3.0 * vol
    elif direction == "SHORT":
        entry = price
        sl = entry + 1.5 * vol
        tp = entry - 3.0 * vol
    else:
        entry = price
        sl = entry - 1.5 * vol
        tp = entry + 1.5 * vol

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = round(reward / risk, 2) if risk else 0.0

    return {
        "price": price,
        "ema20": e20,
        "ema50": e50,
        "rsi": rsi14,
        "ema_slope": ema_slope,
        "chg_24": chg_24,
        "vol_pct": vol_pct,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "confidence": round(score, 1),
        "risk_level": risk_level,
        "bull_factors": bull_factors,
        "bear_factors": bear_factors,
        "structure": structure,
        "reasoning": reasoning,
    }


# ---------- Optional LLM Commentary ----------
async def llm_commentary(symbol: str, a: dict) -> Optional[str]:
    """Return short market commentary if an OpenAI-compatible key is configured."""
    if not OPENAI_API_KEY:
        return None
    prompt = (
        f"You are a crypto trading analyst. In ≤3 short sentences, comment on {symbol} "
        f"given price={a['price']:.4f}, EMA20={a['ema20']:.4f}, EMA50={a['ema50']:.4f}, "
        f"RSI={a['rsi']:.1f}, bias={a['direction']}, confidence={a['confidence']}. "
        "Be neutral, not financial advice."
    )
    try:
        async with httpx.AsyncClient(timeout=20) as cli:
            r = await cli.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 160,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("LLM commentary failed: %s", e)
        return None


# ---------- Formatting ----------
def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"${p:,.2f}"
    if p >= 1:
        return f"${p:,.4f}"
    return f"${p:.6f}"


def fmt_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.2f}%"


# ---------- Command Handlers ----------
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "*SentinelFi AI* — Crypto Trading Agent\n"
        "Live Bitget market data + EMA/RSI signal engine.\n\n"
        "Try:\n"
        "`/price BTC`  ·  `/signal ETH`  ·  `/market SOL`  ·  `/trending`\n\n"
        "Use /help for the full command list."
    )
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "*Commands*\n"
        "`/start` — Welcome message\n"
        "`/help` — This help\n"
        "`/price <SYMBOL>` — Live price (e.g. `/price BTC`)\n"
        "`/signal <SYMBOL>` — Full trade setup with reasoning, factors, risk\n"
        "`/market <SYMBOL>` — Detailed market snapshot\n"
        "`/news <SYMBOL>` — Latest crypto headlines (e.g. `/news BTC`)\n"
        "`/sentiment <SYMBOL>` — Composite tech + Fear & Greed sentiment\n"
        "`/trending` — Top movers (24h) on Bitget spot\n\n"
        "_Data: Bitget Spot V2 · TA: EMA20, EMA50, RSI(14) · 1h candles._\n"
        "_News: CoinTelegraph · Sentiment: alternative.me F&G_\n"
        "_Not financial advice._"
    )
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_price(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: `/price BTC`", parse_mode=constants.ParseMode.MARKDOWN)
        return
    sym = ctx.args[0]
    client: BitgetClient = ctx.application.bot_data["bitget"]
    try:
        t = await client.ticker(sym)
        last = float(t["lastPr"])
        change = float(t.get("change24h", 0)) * 100
        high = float(t.get("high24h", 0))
        low = float(t.get("low24h", 0))
        vol = float(t.get("quoteVolume", 0))
        msg = (
            f"*{client.normalize(sym)}* — Bitget Spot\n"
            f"Last:  *{fmt_price(last)}*  ({fmt_pct(change)})\n"
            f"24h H: {fmt_price(high)}\n"
            f"24h L: {fmt_price(low)}\n"
            f"24h Quote Vol: ${vol:,.0f}"
        )
        await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception as e:
        log.exception("price error")
        await update.message.reply_text(f"Error fetching price: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: `/market BTC`", parse_mode=constants.ParseMode.MARKDOWN)
        return
    sym = ctx.args[0]
    client: BitgetClient = ctx.application.bot_data["bitget"]
    try:
        t, kl = await asyncio.gather(client.ticker(sym), client.candles(sym, "1h", 200))
        closes = np.array([float(k[4]) for k in kl])  # Bitget returns oldest-first
        highs = np.array([float(k[2]) for k in kl])
        lows = np.array([float(k[3]) for k in kl])
        a = analyze(closes, highs, lows)
        change = float(t.get("change24h", 0)) * 100
        msg = (
            f"*{client.normalize(sym)}* — Market Snapshot\n"
            f"Price:   *{fmt_price(a['price'])}*  ({fmt_pct(change)})\n"
            f"EMA20:   {fmt_price(a['ema20'])}\n"
            f"EMA50:   {fmt_price(a['ema50'])}\n"
            f"RSI(14): {a['rsi']:.1f}\n"
            f"Bias:    *{a['direction']}*\n"
            f"24h Vol: ${float(t.get('quoteVolume',0)):,.0f}"
        )
        await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception as e:
        log.exception("market error")
        await update.message.reply_text(f"Error: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: `/signal BTC`", parse_mode=constants.ParseMode.MARKDOWN)
        return
    sym = ctx.args[0]
    client: BitgetClient = ctx.application.bot_data["bitget"]
    try:
        kl = await client.candles(sym, "1h", 200)
        closes = np.array([float(k[4]) for k in kl])
        highs = np.array([float(k[2]) for k in kl])
        lows = np.array([float(k[3]) for k in kl])
        if closes.size < 60:
            await update.message.reply_text("Not enough candle history to analyze.")
            return
        a = analyze(closes, highs, lows)

        emoji = "🟢" if a["direction"] == "LONG" else "🔴" if a["direction"] == "SHORT" else "⚪"
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[a["risk_level"]]

        bull_lines = "\n".join(f"  • {f}" for f in a["bull_factors"]) or "  • _none_"
        bear_lines = "\n".join(f"  • {f}" for f in a["bear_factors"]) or "  • _none_"

        msg = (
            f"*📡 SentinelFi Signal — {client.normalize(sym)}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} *Bias:* {a['direction']}    "
            f"🔥 *Confidence:* {a['confidence']}%    "
            f"{risk_emoji} *Risk:* {a['risk_level']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Price:  {fmt_price(a['price'])}   ({fmt_pct(a['chg_24'])} 24h)\n"
            f"📈 EMA20:  {fmt_price(a['ema20'])}\n"
            f"📉 EMA50:  {fmt_price(a['ema50'])}\n"
            f"📊 RSI(14): {a['rsi']:.1f}\n"
            f"〰️ Vol:    {a['vol_pct']:.2f}% (ATR-proxy)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*🧭 Market Structure*\n_{a['structure']}_\n\n"
            f"*💡 Why this signal*\n_{a['reasoning']}_\n\n"
            f"*✅ Bullish Factors*\n{bull_lines}\n\n"
            f"*⚠️ Bearish Factors*\n{bear_lines}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*📋 Trade Setup*\n"
            f"🎯 Entry:       {fmt_price(a['entry'])}\n"
            f"✅ Take Profit: {fmt_price(a['tp'])}\n"
            f"🛑 Stop Loss:   {fmt_price(a['sl'])}\n"
            f"⚖️ R/R Ratio:   *{a['rr']}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Timeframe: 1H · Source: Bitget · Not financial advice._"
        )

        commentary = await llm_commentary(client.normalize(sym), a)
        if commentary:
            msg += f"\n\n🧠 *AI Commentary*\n_{commentary}_"

        await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception as e:
        log.exception("signal error")
        await update.message.reply_text(f"Error: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


async def cmd_trending(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    client: BitgetClient = ctx.application.bot_data["bitget"]
    try:
        tickers = await client.all_tickers()
        rows = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            try:
                chg = float(t.get("change24h", 0)) * 100
                vol = float(t.get("quoteVolume", 0))
                if vol < 5_000_000:  # liquidity filter
                    continue
                rows.append((sym, chg, vol, float(t["lastPr"])))
            except Exception:
                continue
        rows.sort(key=lambda r: r[1], reverse=True)
        gainers = rows[:5]
        losers = sorted(rows, key=lambda r: r[1])[:5]

        def fmt_row(r):
            return f"`{r[0]:<12}` {fmt_pct(r[1]):>8}  {fmt_price(r[3])}"

        msg = "*🔥 Trending — Bitget Spot (24h)*\n\n*Top Gainers*\n"
        msg += "\n".join(fmt_row(r) for r in gainers)
        msg += "\n\n*Top Losers*\n"
        msg += "\n".join(fmt_row(r) for r in losers)
        await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception as e:
        log.exception("trending error")
        await update.message.reply_text(f"Error: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


# ---------- /news ----------
COINTELEGRAPH_TAG = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "bnb-chain",
    "XRP": "xrp", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche",
    "LINK": "chainlink", "DOT": "polkadot", "MATIC": "polygon", "TON": "ton",
    "SUI": "sui", "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
}


def _escape_md(text: str) -> str:
    """Escape Telegram Markdown (legacy) reserved chars in free-form strings."""
    if not text:
        return ""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: `/news BTC`", parse_mode=constants.ParseMode.MARKDOWN)
        return
    raw = ctx.args[0].upper().replace("USDT", "")
    tag = COINTELEGRAPH_TAG.get(raw)
    url = f"https://cointelegraph.com/rss/tag/{tag}" if tag else "https://cointelegraph.com/rss"
    label = raw if tag else "Crypto"
    try:
        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 SentinelFi"}) as cli:
            r = await cli.get(url)
            r.raise_for_status()
        root = ET.fromstring(r.text)
        items = root.findall(".//item")[:5]
        if not items:
            await update.message.reply_text(f"No recent news found for *{label}*.", parse_mode=constants.ParseMode.MARKDOWN)
            return

        lines = [f"*📰 Latest {label} News*\n━━━━━━━━━━━━━━━━━━━━"]
        now = datetime.now(timezone.utc)
        for it in items:
            title = _escape_md((it.findtext("title") or "").strip()[:140])
            link = (it.findtext("link") or "").strip()
            pub = it.findtext("pubDate") or ""
            age = ""
            try:
                dt = parsedate_to_datetime(pub)
                mins = int((now - dt).total_seconds() / 60)
                age = f"{mins}m ago" if mins < 120 else f"{mins // 60}h ago" if mins < 2880 else f"{mins // 1440}d ago"
            except Exception:
                pass
            lines.append(f"• [{title}]({link})\n  _CoinTelegraph · {age}_")
        lines.append("\n_Source: CoinTelegraph RSS_")
        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=constants.ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception as e:
        log.exception("news error")
        await update.message.reply_text(f"Error fetching news: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


# ---------- /sentiment ----------
async def _fear_greed() -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=10) as cli:
            r = await cli.get("https://api.alternative.me/fng/?limit=1")
            r.raise_for_status()
            d = (r.json().get("data") or [])
            return d[0] if d else None
    except Exception as e:
        log.warning("fng fetch failed: %s", e)
        return None


async def cmd_sentiment(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        await update.message.reply_text("Usage: `/sentiment BTC`", parse_mode=constants.ParseMode.MARKDOWN)
        return
    sym = ctx.args[0]
    client: BitgetClient = ctx.application.bot_data["bitget"]
    try:
        t_task = client.ticker(sym)
        k_task = client.candles(sym, "1h", 200)
        fng_task = _fear_greed()
        t, kl, fng = await asyncio.gather(t_task, k_task, fng_task)
        closes = np.array([float(k[4]) for k in kl])
        highs = np.array([float(k[2]) for k in kl])
        lows = np.array([float(k[3]) for k in kl])
        a = analyze(closes, highs, lows)
        chg24 = float(t.get("change24h", 0)) * 100

        # ---- Composite sentiment score (0–100) ----
        # Technical component: bias + RSI position
        tech = 50.0
        if a["direction"] == "LONG":
            tech += 15
        elif a["direction"] == "SHORT":
            tech -= 15
        tech += (a["rsi"] - 50) * 0.4  # RSI tilt
        tech += max(min(chg24, 10), -10) * 1.0  # 24h momentum
        tech = max(0, min(100, tech))

        # Market component: Fear & Greed
        fng_val = int(fng["value"]) if fng else 50
        fng_label = fng["value_classification"] if fng else "Unknown"

        composite = round(tech * 0.6 + fng_val * 0.4, 1)
        if composite >= 75:
            label, emoji = "Extreme Greed", "🤑"
        elif composite >= 60:
            label, emoji = "Greedy / Bullish", "🟢"
        elif composite >= 45:
            label, emoji = "Neutral", "⚪"
        elif composite >= 25:
            label, emoji = "Fearful / Bearish", "🔴"
        else:
            label, emoji = "Extreme Fear", "😱"

        # Sentiment bar
        filled = int(composite / 5)
        bar = "█" * filled + "░" * (20 - filled)

        msg = (
            f"*🧠 Sentiment — {client.normalize(sym)}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} *Overall:* {label}  ({composite}/100)\n"
            f"`{bar}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*🔧 Technical Sentiment*\n"
            f"  • Bias: {a['direction']}\n"
            f"  • RSI(14): {a['rsi']:.1f}\n"
            f"  • 24h Change: {fmt_pct(chg24)}\n"
            f"  • Score: {tech:.1f}/100\n\n"
            f"*🌐 Market Sentiment (Fear & Greed)*\n"
            f"  • Index: *{fng_val}* — _{fng_label}_\n"
            f"  • Source: alternative.me\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_Composite = 60% technical + 40% F&G index._"
        )
        await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)
    except Exception as e:
        log.exception("sentiment error")
        await update.message.reply_text(f"Error: `{e}`", parse_mode=constants.ParseMode.MARKDOWN)


# ---------- App Bootstrap ----------
async def on_startup(app: Application) -> None:
    app.bot_data["bitget"] = BitgetClient()
    log.info("SentinelFi bot started.")


async def on_shutdown(app: Application) -> None:
    client: BitgetClient = app.bot_data.get("bitget")
    if client:
        await client.close()
    log.info("SentinelFi bot stopped.")


def build_app() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN in environment / .env")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(on_startup)
        .post_shutdown(on_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("trending", cmd_trending))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("sentiment", cmd_sentiment))
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
