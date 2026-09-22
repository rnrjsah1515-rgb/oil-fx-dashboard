"""탐색 분석: 롤링 상관과 국면별 관계."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import REGIMES

OIL, FX = "brent", "usdkrw"


def rolling_corr(ret: pd.DataFrame, window: int) -> pd.Series:
    """일별 로그수익률의 이동 상관. window 는 거래일 수."""
    return ret[OIL].rolling(window, min_periods=int(window * 0.8)).corr(ret[FX])


def regime_slices(index: pd.DatetimeIndex):
    for name, (start, end) in REGIMES.items():
        mask = (index >= start) & (index <= (end or index.max()))
        yield name, start, end, mask


def _corr_stats(r: pd.DataFrame) -> tuple[float, float, float]:
    """상관계수, p값, 베타(환율 수익률을 유가 수익률에 회귀한 기울기)."""
    c, p = stats.pearsonr(r[OIL], r[FX])
    beta = c * r[FX].std() / r[OIL].std()
    return c, p, beta


def regime_table(daily_ret: pd.DataFrame, monthly_ret: pd.DataFrame, daily_px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = list(regime_slices(daily_ret.index)) + [("전체", None, None, np.ones(len(daily_ret), bool))]
    for name, *_ , mask in periods:
        d = daily_ret.loc[mask, [OIL, FX]].dropna()
        idx = d.index
        m = monthly_ret.loc[(monthly_ret.index >= idx.min().to_period("M").start_time)
                            & (monthly_ret.index <= idx.max()), [OIL, FX]].dropna()
        c, p, beta = _corr_stats(d)
        cm, pm, _ = _corr_stats(m) if len(m) > 3 else (np.nan, np.nan, np.nan)
        px = daily_px.loc[idx.min():idx.max()]
        rows.append({
            "국면": name,
            "기간": f"{idx.min():%Y-%m} ~ {idx.max():%Y-%m}",
            "일수": len(d),
            "일별 상관": c, "일별 p값": p,
            "베타(유가 1%→환율 %)": beta,
            "월별 상관": cm, "월별 p값": pm, "개월": len(m),
            "Brent 변화": px[OIL].iloc[-1] / px[OIL].iloc[0] - 1,
            "원/달러 변화": px[FX].iloc[-1] / px[FX].iloc[0] - 1,
        })
    return pd.DataFrame(rows).set_index("국면")
