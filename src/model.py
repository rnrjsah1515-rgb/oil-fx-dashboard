"""4단계 모형: 유가 수익률 → 환율 수익률 민감도(β)와 시나리오 밴드.

모형:  r_fx,t = β_t · r_oil,t + ε_t,   ε_t ~ N(0, σ_t²)
- β_t, σ_t 는 t-1 일까지의 데이터로만 추정한다 (미래 정보 차단).
- 절편은 두지 않는다. 일별 환율 추세(드리프트)는 잡음 대비 너무 작아 추정 오차만 키운다.
- 3단계 결론(동시점 관계, 선행 예측력 없음)에 따라 '조건부' 예측: 유가 경로를 주면 환율 분포를 낸다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .explore import FX, OIL

BREAK_DATE = "2025-04-29"      # 3단계 QLR(2020~ 표본)이 찾은 변화 시점
EWMA_HALFLIFE = 120            # 거래일. 사전에 정한 값 (반년), 민감도는 halflife_sensitivity 로 확인
MIN_OBS = 120

METHODS = {
    "full": "전 기간 (누적)",
    "rolling250": "롤링 250일",
    "ewma": f"EWMA (반감기 {EWMA_HALFLIFE}일)",
    "postbreak": "변화 시점 이후 (참고: 사후 선택)",
}


def _beta_sigma(x: pd.Series, y: pd.Series, method: str, halflife: int = EWMA_HALFLIFE) -> pd.DataFrame:
    """각 날짜 t 에 't 까지' 데이터로 추정한 β, 잔차 σ. (예측에 쓸 땐 shift(1))"""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    if method == "postbreak":
        df = df.loc[BREAK_DATE:]
    # 원점 통과 회귀: β = E[xy]/E[x²], σ² = E[y²] - β·E[xy]
    m = pd.DataFrame({"xy": df.x * df.y, "xx": df.x ** 2, "yy": df.y ** 2})
    if method in ("full", "postbreak"):
        mr = m.expanding(min_periods=MIN_OBS).mean()
    elif method == "rolling250":
        mr = m.rolling(250, min_periods=MIN_OBS).mean()
    elif method == "ewma":
        mr = m.ewm(halflife=halflife, min_periods=MIN_OBS).mean()
    else:
        raise ValueError(method)
    beta = mr.xy / mr.xx
    sigma = np.sqrt((mr.yy - beta * mr.xy).clip(lower=0))
    return pd.DataFrame({"beta": beta, "sigma": sigma}).reindex(x.index)


def estimates(ret: pd.DataFrame, methods=METHODS, halflife: int = EWMA_HALFLIFE) -> dict[str, pd.DataFrame]:
    d = ret[[OIL, FX]].dropna()
    return {m: _beta_sigma(d[OIL], d[FX], m, halflife) for m in methods}


def walk_forward(ret: pd.DataFrame, est: dict[str, pd.DataFrame], start: str, end: str | None = None,
                 horizon: int = 1) -> pd.DataFrame:
    """β_{t-1}·(실제 유가 변화) 로 환율 변화를 맞히는 표본 밖 검증.

    horizon>1 이면 겹치지 않는 h일 누적수익률로 평가 (β, σ 는 구간 시작 전날 값, σ 는 √h 배).
    지표
      OOS R²      : 1 - MSE(모형)/MSE(0 예측). 0 예측 = '유가는 환율에 아무 영향 없다'
      방향 적중률 : 유가가 크게(상위 20%) 움직인 구간에서 환율 방향을 맞힌 비율
      80%/95% 커버리지 : 실제값이 예측구간에 들어간 비율 (80%, 95%에 가까울수록 구간이 정직함)
    """
    d = ret[[OIL, FX]].dropna().loc[:end]
    cum = d.loc[start:].iloc[: (len(d.loc[start:]) // horizon) * horizon]
    blocks = np.arange(len(cum)) // horizon
    agg = cum.groupby(blocks).sum()
    first_day = cum.index[::horizon]
    rows = []
    for name, e in est.items():
        prev = e.shift(1).reindex(first_day)                   # 구간 시작 전날까지의 추정치
        ok = prev.beta.notna().values
        if ok.sum() < 30:
            continue
        x, y = agg[OIL].values[ok], agg[FX].values[ok]
        b, s = prev.beta.values[ok], prev.sigma.values[ok] * np.sqrt(horizon)
        pred = b * x
        big = np.abs(x) >= np.quantile(np.abs(x), 0.8)
        z80, z95 = stats.norm.ppf(0.9), stats.norm.ppf(0.975)
        rows.append({
            "방식": METHODS.get(name, name), "key": name, "표본": int(ok.sum()),
            "OOS R²": 1 - np.mean((y - pred) ** 2) / np.mean(y ** 2),
            "방향 적중(유가 큰 변동)": np.mean(np.sign(pred[big]) == np.sign(y[big])),
            "80% 커버리지": np.mean(np.abs(y - pred) <= z80 * s),
            "95% 커버리지": np.mean(np.abs(y - pred) <= z95 * s),
            "최근 β": e.beta.dropna().iloc[-1],
        })
    return pd.DataFrame(rows).set_index("key")


def scenario(spot: float, beta: float, sigma_d: float, oil_change: float, days: int,
             levels=(0.80, 0.95)) -> pd.DataFrame:
    """유가가 days 거래일 동안 oil_change(예: +0.2) 만큼 선형(로그) 경로로 움직일 때 환율 분포.

    중심값 = spot·exp(β·누적 유가 로그변화), 구간 = 중심 × exp(±z·σ·√t)
    """
    t = np.arange(0, days + 1)
    oil_path = np.log1p(oil_change) * t / days
    center = spot * np.exp(beta * oil_path)
    out = pd.DataFrame({"day": t, "oil_cum": np.expm1(oil_path), "center": center})
    for lv in levels:
        z = stats.norm.ppf(0.5 + lv / 2)
        out[f"lo{int(lv*100)}"] = center * np.exp(-z * sigma_d * np.sqrt(t))
        out[f"hi{int(lv*100)}"] = center * np.exp(z * sigma_d * np.sqrt(t))
    return out


def shock_backtest(px: pd.DataFrame, est: pd.DataFrame, start: str, end: str) -> dict:
    """과거 충격 구간: 당시(시작 전날) β·σ 로 낸 예측 vs 실제 환율 변화."""
    seg = px.loc[start:end, [OIL, FX]].dropna()
    days = len(seg) - 1
    oil_chg = seg[OIL].iloc[-1] / seg[OIL].iloc[0] - 1
    fx_chg = seg[FX].iloc[-1] / seg[FX].iloc[0] - 1
    prev = est.loc[:seg.index[0]].dropna()
    if prev.empty:
        return {"days": days, "oil_chg": oil_chg, "fx_chg": fx_chg, "pred": np.nan, "lo95": np.nan, "hi95": np.nan,
                "beta": np.nan}
    b, s = prev.iloc[-1][["beta", "sigma"]]          # 충격 시작일 종가까지의 정보
    center = b * np.log1p(oil_chg)
    band = stats.norm.ppf(0.975) * s * np.sqrt(days)
    return {"days": days, "oil_chg": oil_chg, "fx_chg": fx_chg, "beta": b,
            "pred": np.expm1(center), "lo95": np.expm1(center - band), "hi95": np.expm1(center + band),
            "fx_start": seg[FX].iloc[0], "fx_end": seg[FX].iloc[-1]}
