"""3단계 통계 검정: 정상성, 공적분, 선후행(그레인저), 구조변화.

모든 검정은 로그 가격(수준) 또는 로그수익률(차분)에 대해 수행한다.
수익률 회귀의 표준오차는 이분산·자기상관에 강건한 HAC(Newey-West)를 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint

from .explore import FX, OIL

# Stock & Watson (Table 14.5) QLR 임계값, 15% 트리밍, 제약 2개(절편·기울기)의 F 통계량
QLR_CRIT_Q2 = {0.10: 5.00, 0.05: 5.86, 0.01: 7.78}


def adf_table(levels: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    lv = np.log(levels[[OIL, FX]].dropna())
    for form, df in [("로그 수준", lv), ("로그수익률", lv.diff().dropna())]:
        for col in (OIL, FX):
            stat, p, lags, n, crit, _ = adfuller(df[col], autolag="AIC",
                                                  regression="c" if form == "로그수익률" else "ct")
            rows.append({"주기": label, "형태": form, "변수": col, "ADF 통계량": stat, "p값": p,
                         "5% 임계값": crit["5%"], "판정": "정상" if p < 0.05 else "단위근(비정상)"})
    return pd.DataFrame(rows)


def coint_test(levels: pd.DataFrame) -> dict:
    """Engle-Granger: log(환율) = a + b·log(유가) + u, u 가 정상이면 공적분."""
    lv = np.log(levels[[OIL, FX]].dropna())
    stat, p, crit = coint(lv[FX], lv[OIL], trend="c")
    b = sm.OLS(lv[FX], sm.add_constant(lv[OIL])).fit().params[OIL]
    return {"EG 통계량": stat, "p값": p, "5% 임계값": crit[1], "장기 탄력성": b, "n": len(lv),
            "판정": "공적분 있음" if p < 0.05 else "공적분 없음"}


def _lagmat(s: pd.Series, lags: int, prefix: str) -> pd.DataFrame:
    return pd.concat({f"{prefix}{k}": s.shift(k) for k in range(1, lags + 1)}, axis=1)


def granger(ret: pd.DataFrame, cause: str, effect: str, lags: int) -> dict:
    """effect_t 를 자기 시차와 cause 시차에 회귀, cause 시차 계수가 모두 0인지 HAC Wald 검정."""
    X = pd.concat([_lagmat(ret[effect], lags, "y"), _lagmat(ret[cause], lags, "x")], axis=1)
    df = pd.concat([ret[effect].rename("dep"), X], axis=1).dropna()
    res = sm.OLS(df["dep"], sm.add_constant(df.drop(columns="dep"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    test = res.f_test(" = 0, ".join(f"x{k}" for k in range(1, lags + 1)) + " = 0")
    return {"방향": f"{cause} → {effect}", "시차": lags, "F": float(test.fvalue), "p값": float(test.pvalue),
            "시차계수 합": sum(res.params[f"x{k}"] for k in range(1, lags + 1))}


def cross_corr(ret: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """k>0: k일 전 유가 수익률과 오늘 환율 수익률의 상관 (유가 선행)."""
    rows = []
    n = ret[[OIL, FX]].dropna().shape[0]
    for k in range(-max_lag, max_lag + 1):
        c = ret[FX].corr(ret[OIL].shift(k))
        rows.append({"시차 k": k, "상관": c, "95% 밴드": 1.96 / np.sqrt(n)})
    return pd.DataFrame(rows).set_index("시차 k")


def _ssr(y, x) -> float:
    return float(sm.OLS(y, sm.add_constant(x)).fit().ssr)


def chow(ret: pd.DataFrame, date: str) -> dict:
    """사전 지정 시점 Chow F + HAC 강건 버전(더미 교호항 Wald)."""
    d = ret[[OIL, FX]].dropna()
    pre, post = d[d.index < date], d[d.index >= date]
    k = 2
    ssr_p, ssr_1, ssr_2 = _ssr(d[FX], d[OIL]), _ssr(pre[FX], pre[OIL]), _ssr(post[FX], post[OIL])
    F = ((ssr_p - ssr_1 - ssr_2) / k) / ((ssr_1 + ssr_2) / (len(d) - 2 * k))
    p = 1 - stats.f.cdf(F, k, len(d) - 2 * k)

    D = (d.index >= date).astype(float)
    X = pd.DataFrame({"oil": d[OIL], "D": D, "D_oil": D * d[OIL]}, index=d.index)
    res = sm.OLS(d[FX], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    rob = res.f_test("D = 0, D_oil = 0")
    return {"시점": date, "Chow F": F, "Chow p값": p, "HAC F": float(rob.fvalue), "HAC p값": float(rob.pvalue),
            "베타(이전)": res.params["oil"], "베타(이후)": res.params["oil"] + res.params["D_oil"],
            "기울기 변화 p값(HAC)": res.pvalues["D_oil"]}


def qlr(ret: pd.DataFrame, trim: float = 0.15, step: int = 5) -> tuple[pd.Series, dict]:
    """모든 후보 시점의 Chow F 를 계산해 최대값(sup-F)을 QLR 임계값과 비교. 시점을 데이터가 고른다."""
    d = ret[[OIL, FX]].dropna()
    n, k = len(d), 2
    ssr_p = _ssr(d[FX], d[OIL])
    Fs = {}
    for i in range(int(n * trim), int(n * (1 - trim)), step):
        a, b = d.iloc[:i], d.iloc[i:]
        s1, s2 = _ssr(a[FX], a[OIL]), _ssr(b[FX], b[OIL])
        Fs[d.index[i]] = ((ssr_p - s1 - s2) / k) / ((s1 + s2) / (n - 2 * k))
    F = pd.Series(Fs)
    return F, {"sup-F": F.max(), "추정 시점": F.idxmax(), "5% 임계값": QLR_CRIT_Q2[0.05],
               "1% 임계값": QLR_CRIT_Q2[0.01], "판정": "구조변화 있음" if F.max() > QLR_CRIT_Q2[0.05] else "없음"}
