"""유가–환율 시나리오 대시보드: streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data, hedge, model
from src.config import SHOCKS
from src.explore import FX, OIL

st.set_page_config(page_title="유가–환율 민감도 분석", page_icon="🛢️", layout="wide")

# dataviz 참조 팔레트
C1, C2, INK2 = "#2a78d6", "#eb6834", "#8a8984"
BAND80, BAND95 = "rgba(42,120,214,0.28)", "rgba(42,120,214,0.13)"


# ── 데이터 ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner="FRED 에서 최신 데이터를 받는 중…")
def load() -> tuple[pd.DataFrame, str]:
    try:
        ds = data.build(refresh=True)
        data.save(ds)
        return ds.daily, "FRED 실시간"
    except Exception:
        daily = pd.read_csv(data.PROCESSED / "daily.csv", index_col=0, parse_dates=True)
        return daily, "저장된 스냅샷 (FRED 응답 없음)"


@st.cache_data(show_spinner=False)
def fit(daily: pd.DataFrame):
    ret = np.log(daily).diff().dropna(how="all")
    est = model.estimates(ret)
    wf = model.walk_forward(ret, est, "2015-01-01")
    wf_recent = model.walk_forward(ret, est, "2025-05-01")
    return est, wf, wf_recent


daily, source = load()
est, wf, wf_recent = fit(daily)
spot, oil_now, asof = daily[FX].iloc[-1], daily[OIL].iloc[-1], daily.index[-1]

# ── 사이드바: 시나리오 입력 ────────────────────────────────────────────────
with st.sidebar:
    st.header("시나리오")
    preset = st.radio("유가 경로", ["직접 입력", *SHOCKS],
                      help="과거 충격을 고르면 그 당시 유가 변화율·기간을 오늘에 적용합니다.")
    if preset == "직접 입력":
        oil_chg = st.slider("Brent 변화율 (%)", -60, 100, 20, 5) / 100
        days = st.slider("기간 (영업일)", 5, 120, 20, 5)
    else:
        bt = model.shock_backtest(daily, est["ewma"], *SHOCKS[preset])
        oil_chg, days = bt["oil_chg"], bt["days"]
        st.info(f"Brent **{oil_chg:+.0%}** / **{days}영업일**\n\n({SHOCKS[preset][0]} → {SHOCKS[preset][1]})")
    method = st.selectbox("민감도(β) 추정 방식", list(model.METHODS), index=2, format_func=model.METHODS.get,
                          help="EWMA 가 표본 밖 검증에서 가장 좋아 기본값입니다.")
    st.divider()
    st.caption(f"데이터: {source}  \n기준일 {asof:%Y-%m-%d}  \n출처: FRED (DCOILBRENTEU, DEXKOUS)")

e_now = est[method].dropna().iloc[-1]
sc = model.scenario(spot, e_now.beta, e_now.sigma, oil_chg, days)
end = sc.iloc[-1]

# ── 헤더 ──────────────────────────────────────────────────────────────────
st.title("유가 변화에 대한 원/달러 환율 민감도 분석")
st.caption("유가로 환율을 예측할 수 있는지 검증한 결과, **예측력은 없었습니다** (전일 유가의 설명력 0.1%). "
           "이 화면은 유가도 환율도 예측하지 않습니다. 사용자가 유가 경로를 **가정**하면, 과거에 유가와 환율이 "
           "같은 날 함께 움직인 정도(β)를 적용해 환율 범위를 계산하는 **민감도·스트레스 분석 도구**입니다.")

k = st.columns(4)
k[0].metric("Brent (달러/배럴)", f"{oil_now:,.1f}", f"{oil_now / daily[OIL].iloc[-21] - 1:+.1%} (1개월)",
            delta_color="off")
k[1].metric("원/달러", f"{spot:,.1f}", f"{spot / daily[FX].iloc[-21] - 1:+.1%} (1개월)", delta_color="inverse")
k[2].metric("현재 β", f"{e_now.beta:+.3f}", "유가↑ → 원화 약세" if e_now.beta > 0 else "유가↑ → 원화 강세",
            delta_color="off")
k[3].metric("환율 일간 변동성 (σ)", f"{e_now.sigma:.2%}")

tab1, tab2 = st.tabs(["시나리오 분석", "헤지 계산기"])

# ── 탭 1: 시나리오 ────────────────────────────────────────────────────────
with tab1:
    hist = daily[FX].iloc[-250:]
    fdates = pd.bdate_range(asof, periods=len(sc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[fdates, fdates[::-1]], y=np.r_[sc.hi95, sc.lo95[::-1]], fill="toself",
                             fillcolor=BAND95, line=dict(width=0), name="95% 구간", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.r_[fdates, fdates[::-1]], y=np.r_[sc.hi80, sc.lo80[::-1]], fill="toself",
                             fillcolor=BAND80, line=dict(width=0), name="80% 구간", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="실제 원/달러", line=dict(color=INK2, width=1.5),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}원<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fdates, y=sc.center, name="조건부 중심", line=dict(color=C1, width=2.5),
        customdata=np.c_[sc.oil_cum * 100, sc.lo95, sc.hi95],
        hovertemplate="%{x|%Y-%m-%d}<br>중심 %{y:,.1f}원<br>95%: %{customdata[1]:,.0f} ~ %{customdata[2]:,.0f}"
                      "<br>누적 유가 %{customdata[0]:+.1f}%<extra></extra>"))
    fig.add_hline(y=spot, line=dict(color=INK2, width=0.8, dash="dot"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title="원/달러")
    st.plotly_chart(fig, width="stretch")

    effect = end.center - spot
    m = st.columns(4)
    m[0].metric(f"{days}영업일 후 중심 환율", f"{end.center:,.0f}원", f"{effect:+,.0f}원 (유가 효과)",
                delta_color="inverse")
    m[1].metric("80% 구간", f"{end.lo80:,.0f} ~ {end.hi80:,.0f}")
    m[2].metric("95% 구간", f"{end.lo95:,.0f} ~ {end.hi95:,.0f}")
    m[3].metric("불확실성 / 유가 효과", f"{(end.hi95 - end.center) / max(abs(effect), 0.01):,.1f}배",
                help="95% 구간 반폭 ÷ |유가 효과|. 1보다 크면 유가 외 요인의 변동이 더 크다는 뜻입니다.")
    st.info(f"유가가 가정대로 {oil_chg:+.0%} 움직여도 과거 관계로 설명되는 환율 변화는 **{effect:+,.0f}원**이고, "
            f"설명되지 않는 변동은 **±{end.hi95 - end.center:,.0f}원**(95%)입니다. 유가 가정만으로는 환율 방향을 판단하기 어렵습니다.")

    st.subheader("과거 충격: 당시 모형 vs 실제")
    rows = []
    for name, (s0, s1) in SHOCKS.items():
        b = model.shock_backtest(daily, est[method], s0, s1)
        has = not np.isnan(b["pred"])
        rows.append({"충격": name, "기간(일)": b["days"], "Brent": f"{b['oil_chg']:+.0%}",
                     "실제 환율": f"{b['fx_chg']:+.1%}",
                     "당시 β": f"{b['beta']:+.3f}" if has else "–",
                     "모형 중심": f"{b['pred']:+.1%}" if has else "–",
                     "모형 95% 구간": f"{b['lo95']:+.1%} ~ {b['hi95']:+.1%}" if has else "–",
                     "구간 안": ("✅" if b["lo95"] <= b["fx_chg"] <= b["hi95"] else "❌") if has else "–"})
    st.dataframe(pd.DataFrame(rows).set_index("충격"), width="stretch")
    st.caption("각 충격 시작일까지의 데이터만으로 추정한 β·σ 로 계산했습니다 (미래 정보 미사용). 구간이 넓어 4건 모두 안에 드는 것은 강한 증거가 아닙니다.")
    st.subheader("β 추정 방식에 따른 차이")
    rows = []
    for key, lab in model.METHODS.items():
        e = est[key].dropna().iloc[-1]
        s_ = model.scenario(spot, e.beta, e.sigma, oil_chg, days).iloc[-1]
        rows.append({"방식": lab, "β": f"{e.beta:+.3f}", "중심": f"{s_.center:,.0f}",
                     "95% 구간": f"{s_.lo95:,.0f} ~ {s_.hi95:,.0f}",
                     "OOS R² (2015~)": f"{wf.loc[key, 'OOS R²']:+.1%}" if key in wf.index else "–",
                     "OOS R² (2025-05~)": f"{wf_recent.loc[key, 'OOS R²']:+.1%}" if key in wf_recent.index else "–"})
    st.dataframe(pd.DataFrame(rows).set_index("방식"), width="stretch")
    st.caption("OOS R² = 1 − MSE(모형)/MSE(0 예측). 음수면 '유가 영향 없음'보다 못하다는 뜻입니다. "
               "'변화 시점 이후'는 변화 시점을 사후에 알고 고른 것이라 참고용입니다.")

# ── 탭 2: 헤지 계산기 ─────────────────────────────────────────────────────
with tab2:
    st.caption(f"위 시나리오(Brent {oil_chg:+.0%}, {days}영업일, {model.METHODS[method]})의 만기 환율 분포로 계산합니다.")
    c = st.columns(4)
    side = c[0].radio("포지션", ["import", "export"],
                      format_func={"import": "수입 (달러 지급)", "export": "수출 (달러 수취)"}.get)
    amount = c[1].number_input("금액 (백만 달러)", 0.1, 10_000.0, 10.0, 1.0) * 1e6
    swap = c[2].number_input("스왑포인트 (원)", -50.0, 50.0, 0.0, 0.5,
                             help="선물환율 = 현물 + 스왑포인트. 원/달러 금리차로 결정되며 기본값 0은 단순화 가정입니다.")
    ratio = c[3].slider("내 헤지 비율 (%)", 0, 100, 50, 10) / 100
    fwd = spot + swap

    bad, good = ("hi95", "lo95") if side == "import" else ("lo95", "hi95")
    q = {"95% 불리": end[bad], "중심": end.center, "95% 유리": end[good]}
    ratios = sorted({0.0, ratio, 1.0})
    tbl = hedge.exposure(amount, side, spot, fwd, q, ratios)
    piv = tbl.pivot(index="헤지 비율", columns="시나리오", values="예산 대비 손익(원)")[list(q)]
    piv = piv.loc[[f"{r:.0%}" for r in ratios]]

    fig2 = go.Figure()
    for i, (lab, r) in enumerate(piv.iterrows()):
        lo, hi = sorted([r["95% 불리"], r["95% 유리"]])
        fig2.add_trace(go.Bar(y=[lab], x=[(hi - lo) / 1e8], base=[lo / 1e8], orientation="h", marker_color=C1,
                              marker_line=dict(width=0), width=0.45, showlegend=i == 0, name="95% 손익 범위",
                              hovertemplate=f"헤지 {lab}<br>{lo / 1e8:,.1f}억 ~ {hi / 1e8:,.1f}억<extra></extra>"))
        fig2.add_trace(go.Scatter(y=[lab], x=[r["중심"] / 1e8], mode="markers", showlegend=i == 0, name="중심",
                                  marker=dict(size=10, color=C2, line=dict(color="white", width=2)),
                                  hovertemplate=f"헤지 {lab}<br>중심 %{{x:,.2f}}억<extra></extra>"))
    fig2.add_vline(x=0, line=dict(color=INK2, width=1))
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), barmode="overlay",
                       xaxis_title="예산(오늘 현물 환산) 대비 손익 (억원)",
                       yaxis=dict(title="헤지 비율", type="category"),
                       legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig2, width="stretch")

    def won(v: float) -> str:
        v = round(v / 1e8, 2) + 0.0                     # -0.00 방지
        return f"손실 {-v:,.1f}억원" if v < 0 else f"이익 {v:,.1f}억원"

    worst0 = piv.loc["0%", "95% 불리"]
    worst_r = piv.loc[f"{ratio:.0%}", "95% 불리"]
    st.success(f"헤지하지 않으면 95% 불리 시나리오에서 **{won(worst0)}**, {ratio:.0%} 헤지하면 **{won(worst_r)}** "
               f"(예산 {amount * spot / 1e8:,.0f}억원 기준). 헤지는 유리한 시나리오의 이익도 같은 비율로 포기합니다.")
    st.dataframe(piv.map(lambda v: f"{round(v / 1e8, 2) + 0.0:+,.2f}억"), width="stretch")
    st.caption("가정: 선물환(통화선물)으로 헤지, 증거금·거래비용·베이시스 위험 무시. 손익은 오늘 현물 환율로 환산한 예산 대비.")

with st.expander("모형과 한계"):
    st.markdown(f"""
- **모형**: `r_fx,t = β_t · r_oil,t + ε_t` (일별 로그수익률, 원점 통과). β·σ 는 각 시점까지 데이터로만 추정.
- **검정 결과**: 수준은 단위근·공적분 없음 → 수익률 모형. 유가→환율 그레인저 유의하나 설명력 0.1% → 선행 예측 불가,
  동시점 관계만 존재. 2020년 이후 QLR 이 **2025-04-29** 를 구조변화 시점으로 식별 (β 부호가 음→양으로 전환).
- **표본 밖 검증 (일별, 2015~)**: EWMA OOS R² {wf.loc['ewma', 'OOS R²']:+.1%},
  95% 구간 커버리지 {wf.loc['ewma', '95% 커버리지']:.1%}. 주·월 단위에선 R² ≈ 0 → 기간이 길수록 중심선보다 **구간**이 의미 있음.
- **한계**: 유가 외 요인(달러 인덱스, 금리차, 외국인 자금) 미포함 · 정규분포 가정(꼬리 위험 과소) ·
  FRED 원/달러는 뉴욕 정오 기준 · 유가 경로는 사용자 가정 (선물 커브 연동은 향후 과제).
""")
