"""2단계 탐색 분석 실행: python run_explore.py"""
import pandas as pd

from src import data, explore, figures

ds = data.build()
data.save(ds)
dret, mret = ds.returns("D"), ds.returns("M")

tbl = explore.regime_table(dret, mret, ds.daily)
tbl.to_csv(figures.OUT / "regime_table.csv", encoding="utf-8-sig")
for p in (figures.fig_levels(ds.daily), figures.fig_rolling(dret), figures.fig_regimes(tbl)):
    print("saved", p.name)

pd.set_option("display.width", 200)
fmt = tbl.copy()
for c in ["일별 상관", "월별 상관", "베타(유가 1%→환율 %)"]:
    fmt[c] = fmt[c].map("{:+.3f}".format)
for c in ["일별 p값", "월별 p값"]:
    fmt[c] = fmt[c].map("{:.3f}".format)
for c in ["Brent 변화", "원/달러 변화"]:
    fmt[c] = fmt[c].map("{:+.0%}".format)
print(fmt.to_string())
r250 = explore.rolling_corr(dret, 250).dropna()
print(f"\n250일 상관: 최소 {r250.min():+.2f} ({r250.idxmin():%Y-%m}), 최대 {r250.max():+.2f} ({r250.idxmax():%Y-%m}), "
      f"양(+)인 비율 {(r250 > 0).mean():.0%}, 최근 {r250.iloc[-1]:+.2f}")
