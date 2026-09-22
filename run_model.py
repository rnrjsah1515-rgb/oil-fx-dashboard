"""4단계 모형 추정·검증: python run_model.py"""
import pandas as pd

from src import data, model
from src.figures import OUT

pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.4f}".format)
ds = data.build()
ret = ds.returns("D")
est = model.estimates(ret)

res = []
for label, start, end, h in [("일별 2015~", "2015-01-01", None, 1), ("주간(5일) 2015~", "2015-01-01", None, 5),
                             ("일별 2025-05~", "2025-05-01", None, 1), ("주간(5일) 2025-05~", "2025-05-01", None, 5),
                             ("월간(21일) 2015~", "2015-01-01", None, 21)]:
    t = model.walk_forward(ret, est, start, end, h); t.insert(0, "검증", label); res.append(t)
res = pd.concat(res)
print(res.drop(columns="최근 β").to_string())
res.to_csv(OUT / "t7_walkforward.csv", encoding="utf-8-sig")

print("\n최근 추정치 (2026-09 기준)")
for k, e in est.items():
    last = e.dropna().iloc[-1]
    print(f"  {model.METHODS[k]:24s} β={last.beta:+.4f}  σ(일)={last.sigma:.4%}  → 유가 +20% 시 환율 {((1.2)**last.beta-1):+.2%}")

print("\nEWMA 반감기 민감도 (일별 2015~ OOS R²)")
for hl in [30, 60, 120, 250, 500]:
    e = model.estimates(ret, {"ewma": ""}, halflife=hl)
    t = model.walk_forward(ret, e, "2015-01-01")
    print(f"  반감기 {hl:>3}일  OOS R²={t['OOS R²'].iloc[0]:.4f}  95%커버={t['95% 커버리지'].iloc[0]:.3f}")

from src import figures
figures.fig_betas(est, model.METHODS)
e = est["ewma"].dropna().iloc[-1]
spot = ds.daily["usdkrw"].iloc[-1]
sc = model.scenario(spot, e.beta, e.sigma, oil_change=0.20, days=20)
sc.to_csv(OUT / "t8_scenario_example.csv", index=False, encoding="utf-8-sig")
figures.fig_scenario(ds.daily["usdkrw"].iloc[-120:], sc,
                     f"시나리오 예시: 향후 20영업일 Brent +20% (EWMA β={e.beta:+.3f})")
print()
print(f"시나리오(+20%, 20일): 중심 {sc.center.iloc[-1]:,.1f} (현재 {spot:,.1f}), "
      f"80% [{sc.lo80.iloc[-1]:,.0f}, {sc.hi80.iloc[-1]:,.0f}], 95% [{sc.lo95.iloc[-1]:,.0f}, {sc.hi95.iloc[-1]:,.0f}]")
