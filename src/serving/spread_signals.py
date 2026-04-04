"""Spread signal interpretation for trading context.

Translates raw spread values into actionable signals with
trend direction, moving average deviation, and business interpretation.
"""

from __future__ import annotations


def interpret_crush_spread(
    current: float,
    ma30: float,
    trend: str,
) -> dict:
    """Interpret crush spread for soybean processors.

    Args:
        current: Current crush spread value.
        ma30: 30-day moving average of crush spread.
        trend: Direction over last 5 days ("up", "down", "flat").

    Returns:
        Dict with interpretation text and signal color.
    """
    if ma30 == 0:
        return {"interpretation": "Insufficient data for signal", "signal": "neutral"}

    deviation_pct = (current - ma30) / abs(ma30) * 100

    if deviation_pct > 10:
        return {
            "interpretation": f"Crush margin {deviation_pct:.0f}% above 30d avg — bullish for processor demand",
            "signal": "bullish",
        }
    elif deviation_pct < -10:
        return {
            "interpretation": "Crush margin compressed — processors reducing throughput",
            "signal": "bearish",
        }
    elif trend == "up":
        return {
            "interpretation": "Crush margin recovering toward average — processor margins improving",
            "signal": "neutral",
        }
    elif trend == "down":
        return {
            "interpretation": "Crush margin weakening — watch for processor slowdown",
            "signal": "neutral",
        }
    else:
        return {
            "interpretation": "Crush margin near historical average",
            "signal": "neutral",
        }


def interpret_oil_palm_spread(
    current: float,
    ma30: float,
    trend: str,
) -> dict:
    """Interpret soybean oil vs palm oil spread.

    Args:
        current: Current oil/palm spread value.
        ma30: 30-day moving average.
        trend: Direction over last 5 days ("up", "down", "flat").

    Returns:
        Dict with interpretation text and signal color.
    """
    if ma30 == 0:
        return {"interpretation": "Insufficient data for signal", "signal": "neutral"}

    deviation_pct = (current - ma30) / abs(ma30) * 100

    if trend == "up" and deviation_pct > 5:
        return {
            "interpretation": "Soy oil premium to palm expanding — substitution risk rising",
            "signal": "bullish",
        }
    elif trend == "down" and deviation_pct < -5:
        return {
            "interpretation": "Spread narrowing — soy oil gaining competitive advantage",
            "signal": "bearish",
        }
    elif deviation_pct > 10:
        return {
            "interpretation": f"Spread {deviation_pct:.0f}% above avg — buyers may shift to palm",
            "signal": "bullish",
        }
    elif deviation_pct < -10:
        return {
            "interpretation": "Spread below avg — soy oil attractively priced vs palm",
            "signal": "bearish",
        }
    else:
        return {
            "interpretation": "Spread stable — no clear substitution signal",
            "signal": "neutral",
        }


def compute_trend(values: list[float], lookback: int = 5) -> str:
    """Determine trend direction from recent values.

    Args:
        values: Time-ordered list of spread values (oldest first).
        lookback: Number of periods to compare.

    Returns:
        "up", "down", or "flat".
    """
    if len(values) < lookback + 1:
        return "flat"

    recent = values[-1]
    past = values[-(lookback + 1)]

    if past == 0:
        return "flat"

    change_pct = (recent - past) / abs(past) * 100

    if change_pct > 2:
        return "up"
    elif change_pct < -2:
        return "down"
    return "flat"
