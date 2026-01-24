import streamlit as st
import pandas as pd

from stats_app.helpers.ui_components import st_df


def _find_col(df, candidates):
    """Return first matching column name from candidates (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def render_tab_options_chain(df):
    st.subheader("📋 Options Chain")

    if df is None or df.empty:
        st.warning("No chain data loaded.")
        return

    # -----------------------------
    # Try to auto-detect price + MAs
    # -----------------------------
    price_col = _find_col(df, ["price", "last", "close", "spot", "underlying"])
    ma20_col = _find_col(df, ["ma20", "sma20", "ema20"])
    ma50_col = _find_col(df, ["ma50", "sma50", "ema50"])
    ma200_col = _find_col(df, ["ma200", "sma200", "ema200"])

    explanation_lines = []

    if price_col:
        try:
            price = float(df[price_col].iloc[0])
        except Exception:
            price = None
    else:
        price = None

    def check_ma(ma_col, label):
        if ma_col and price is not None:
            try:
                ma_val = float(df[ma_col].iloc[0])
                if price > ma_val:
                    return f"• Price is **above {label}** → short-term structure is stronger."
                elif price < ma_val:
                    return f"• Price is **below {label}** → short-term structure is weaker."
                else:
                    return f"• Price is **at {label}** → market is balanced here."
            except Exception:
                return None
        return None

    # Build explanation
    l20 = check_ma(ma20_col, "MA20")
    l50 = check_ma(ma50_col, "MA50")
    l200 = check_ma(ma200_col, "MA200")

    if l20: explanation_lines.append(l20)
    if l50: explanation_lines.append(l50)
    if l200: explanation_lines.append(l200)

    # Overall regime guess (educational)
    regime = None
    if price is not None and ma20_col and ma50_col and ma200_col:
        try:
            ma20 = float(df[ma20_col].iloc[0])
            ma50 = float(df[ma50_col].iloc[0])
            ma200 = float(df[ma200_col].iloc[0])

            if price > ma20 and price > ma50 and price > ma200:
                regime = "📈 Trend context: **Uptrend environment** (price above major averages)."
            elif price < ma20 and price < ma50 and price < ma200:
                regime = "📉 Trend context: **Downtrend environment** (price below major averages)."
            else:
                regime = "📦 Trend context: **Range / transition environment** (price mixed vs averages)."
        except Exception:
            pass

    # -----------------------------
    # Display explanation panel
    # -----------------------------
    if explanation_lines or regime:
        st.info(
            "**📊 Trend Context (from Moving Averages — Educational)**\n\n"
            + ("\n".join(explanation_lines) if explanation_lines else "• Moving averages not fully available.\n")
            + ("\n\n" + regime if regime else "")
            + "\n\n⚠️ This describes the **current environment**, not a prediction or instruction."
        )
    else:
        st.info(
            "📊 **Trend Context:** Moving average columns were not detected in this table.\n\n"
            "If you add columns like `MA20`, `MA50`, `MA200`, this panel will explain the trend context automatically."
        )

    # -----------------------------
    # Show the actual chain table
    # -----------------------------
    st_df(df, height=520)
