"""보고서용 정적 그래프 (PNG). 대시보드는 app.py 에서 plotly 로 다시 그린다."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .config import ROOT
from .explore import FX, OIL, regime_slices, rolling_corr

OUT = ROOT / "output"

# 참조 팔레트 (dataviz reference palette, light)
C1, C2 = "#2a78d6", "#eb6834"          # 시리즈 1·2
POS, NEG = "#e34948", "#2a78d6"        # 발산: 양(+)=빨강, 음(−)=파랑
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BAND = ["#f0efec", "#fcfcfb"]          # 국면 음영 교차

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _shade(ax, index, label: bool = False):
    for i, (name, start, end, mask) in enumerate(regime_slices(index)):
        s, e = pd.Timestamp(start), index[mask].max()
        ax.axvspan(s, e, color=BAND[i % 2], zorder=0, lw=0)
        if label:
            ax.text(s + (e - s) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=INK2)


def fig_levels(px: pd.DataFrame, path=OUT / "01_levels.png"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax, col, color, unit in [(a1, OIL, C1, "Brent (달러/배럴)"), (a2, FX, C2, "원/달러 (원)")]:
        _shade(ax, px.index, label=ax is a1)
        ax.plot(px.index, px[col], color=color, lw=1.2)
        ax.set_ylabel(unit)
        last = px[col].dropna()
        ax.annotate(f"{last.iloc[-1]:,.1f}", (last.index[-1], last.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Brent 유가와 원/달러 환율 (일별, 2010~)", x=0.06, ha="left", fontsize=13, color=INK, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def fig_rolling(ret: pd.DataFrame, path=OUT / "02_rolling_corr.png"):
    r60, r250 = rolling_corr(ret, 60), rolling_corr(ret, 250)
    full = ret[OIL].corr(ret[FX])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _shade(ax, ret.index, label=True)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(full, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(ret.index[0], full, f" 전체 기간 {full:+.2f}", va="bottom", fontsize=8.5, color=INK2)
    ax.plot(r60.index, r60, color=C1, lw=0.9, alpha=0.55, label="60일 이동상관")
    ax.plot(r250.index, r250, color=C2, lw=2, label="250일 이동상관")
    for s, c in [(r60, C1), (r250, C2)]:
        s = s.dropna()
        ax.annotate(f"{s.iloc[-1]:+.2f}", (s.index[-1], s.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylabel("상관계수 (일별 로그수익률)")
    ax.set_ylim(-0.75, 0.75)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.suptitle("Brent–원/달러 이동상관: 관계는 안정적이지 않다", x=0.06, ha="left", fontsize=13, color=INK, y=1.08)
    fig.text(0.06, 1.01, "0보다 위 = 유가↑ 때 원화 약세 (무역수지 경로)   ·   0보다 아래 = 유가↑ 때 원화 강세 (글로벌 경기 경로)",
             ha="left", fontsize=9, color=INK2)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def fig_regimes(tbl: pd.DataFrame, path=OUT / "03_regime_corr.png"):
    t = tbl.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    vals = t["일별 상관"]
    colors = [INK2 if name == "전체" else POS if v > 0 else NEG for name, v in vals.items()]
    ax.barh(t.index, vals, color=colors, height=0.55, edgecolor=SURF, linewidth=2)
    ax.axvline(0, color=INK2, lw=0.8)
    for y, (v, p) in enumerate(zip(vals, t["일별 p값"])):
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else " (유의X)"
        ax.text(v + (0.008 if v >= 0 else -0.008), y, f"{v:+.3f}{star}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    lim = max(0.3, vals.abs().max() * 1.6)
    ax.set_xlim(-lim, lim); ax.grid(axis="y", visible=False)
    ax.set_xlabel("일별 로그수익률 상관계수   (* p<0.1  ** p<0.05  *** p<0.01)")
    fig.suptitle("국면별 Brent–원/달러 상관", x=0.02, ha="left", fontsize=13, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def fig_qlr(F: pd.Series, crit: float, found: pd.Timestamp, path=OUT / "04_qlr.png"):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(F.index, F, color=C1, lw=1.6)
    ax.axhline(crit, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(F.index[-1], crit, "5% 임계값 ", va="bottom", ha="right", fontsize=8.5, color=INK2)
    ax.set_ylim(0, F.max() * 1.35)
    ax.plot([found], [F.max()], "o", ms=8, color=C1, mec=SURF, mew=2)
    ax.annotate(f"데이터가 찾은 변화 시점\n{found:%Y-%m-%d} (F={F.max():.1f})", (found, F.max()),
                xytext=(-14, 14), textcoords="offset points", ha="right", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Chow F 통계량")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("구조변화 탐색(QLR): 2020년 이후 표본에서 후보 시점마다 Chow 검정", x=0.06, ha="left",
                 fontsize=13, color=INK, y=1.03)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def fig_betas(est: dict, labels: dict, path=OUT / "05_beta_paths.png"):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _shade(ax, est["full"].dropna().index, label=True)
    ax.axhline(0, color=INK2, lw=0.8)
    styles = {"full": (INK2, 1.4, (0, (4, 3)), 0), "rolling250": (C2, 1.1, "-", 6), "ewma": (C1, 2.0, "-", -6)}
    for k, (c, lw, ls, dy) in styles.items():
        b = est[k]["beta"].dropna()
        ax.plot(b.index, b, color=c, lw=lw, ls=ls, label=labels[k])
        ax.annotate(f"{b.iloc[-1]:+.3f}", (b.index[-1], b.iloc[-1]), xytext=(4, dy),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylabel("β (유가 1% → 환율 %)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.suptitle("추정 방식별 민감도 β 추이 (각 시점까지의 데이터만 사용)", x=0.06, ha="left", fontsize=13, color=INK, y=1.04)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def fig_scenario(hist: pd.Series, sc: pd.DataFrame, title: str, path=OUT / "06_scenario_example.png"):
    """hist: 최근 환율, sc: model.scenario() 결과 (영업일 기준 날짜로 변환해 이어 그림)."""
    dates = pd.bdate_range(hist.index[-1], periods=len(sc))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hist.index, hist, color=INK2, lw=1.2, label="실제 원/달러")
    ax.fill_between(dates, sc.lo95, sc.hi95, color=C1, alpha=0.12, lw=0, label="95% 구간")
    ax.fill_between(dates, sc.lo80, sc.hi80, color=C1, alpha=0.22, lw=0, label="80% 구간")
    ax.plot(dates, sc.center, color=C1, lw=2, label="중심 경로")
    ax.axhline(hist.iloc[-1], color=INK2, lw=0.6, ls=(0, (2, 3)))
    for col, va in [("hi95", "bottom"), ("center", "center"), ("lo95", "top")]:
        ax.annotate(f"{sc[col].iloc[-1]:,.0f}", (dates[-1], sc[col].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va=va, fontsize=9, color=INK)
    ax.set_ylabel("원/달러")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower left", frameon=False, ncol=4)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, y=1.04)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path
