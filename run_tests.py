"""3단계 통계 검정 실행: python run_tests.py"""
import json

import pandas as pd

from src import data, econ
from src.config import REGIMES
from src.figures import OUT

pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.4f}".format)
ds = data.build()
dret, mret = ds.returns("D"), ds.returns("M")
out = {}

print("① ADF 단위근 검정")
adf = pd.concat([econ.adf_table(ds.daily, "일별"), econ.adf_table(ds.monthly, "월별")])
print(adf.to_string(index=False)); adf.to_csv(OUT / "t1_adf.csv", index=False, encoding="utf-8-sig")

print("\n② Engle-Granger 공적분 (log 원/달러 ~ log Brent)")
rows = {"전체(월별)": econ.coint_test(ds.monthly), "전체(일별)": econ.coint_test(ds.daily)}
for name, (s, e) in REGIMES.items():
    rows[name + "(일별)"] = econ.coint_test(ds.daily.loc[s:e])
ct = pd.DataFrame(rows).T; print(ct.to_string()); ct.to_csv(OUT / "t2_coint.csv", encoding="utf-8-sig")

print("\n③ 그레인저 인과 (HAC)")
g = []
for label, r, lags in [("일별", dret, [1, 2, 5]), ("월별", mret, [1, 3])]:
    for L in lags:
        for c, e in [("brent", "usdkrw"), ("usdkrw", "brent")]:
            g.append({"주기": label, **econ.granger(r, c, e, L)})
    rr = r.loc["2025-01-01":]
    if label == "일별":
        for c, e in [("brent", "usdkrw"), ("usdkrw", "brent")]:
            g.append({"주기": "일별(2025~)", **econ.granger(rr, c, e, 1)})
g = pd.DataFrame(g); print(g.to_string(index=False)); g.to_csv(OUT / "t3_granger.csv", index=False, encoding="utf-8-sig")

print("\n④ 시차 상관 (k>0: 유가가 k일 선행)")
cc = pd.concat({"전체": econ.cross_corr(dret)["상관"], "2025~": econ.cross_corr(dret.loc["2025-01-01":])["상관"]}, axis=1)
print(cc.T.to_string()); cc.to_csv(OUT / "t4_crosscorr.csv", encoding="utf-8-sig")

print("\n⑤ 구조변화")
ch = econ.chow(dret, "2025-01-01"); print("Chow(사전 지정 2025-01-01):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ch.items()})
F, q = econ.qlr(dret); print("QLR:", q)
F.to_csv(OUT / "t5_qlr_path.csv", encoding="utf-8-sig")
top = F.sort_values(ascending=False)
print("F 상위 시점(서로 60일 이상 떨어진 것):")
picked = []
for d, v in top.items():
    if all(abs((d - p).days) > 60 for p in picked):
        picked.append(d); print(f"  {d.date()}  F={v:.2f}")
    if len(picked) == 5: break

print("\n⑥ 구조변화 재탐색: 2020년 이후 표본 (전체 표본에선 15% 트리밍으로 2025년이 후보에서 빠짐)")
from src import figures
F20, q20 = econ.qlr(dret.loc["2020-01-01":]); print("QLR(2020~):", q20)
figures.fig_qlr(F20, q20["5% 임계값"], q20["추정 시점"])

print("\n⑦ 설명력 비교 (일별 원/달러 수익률 R²)")
import statsmodels.api as sm
r = dret[["brent", "usdkrw"]].dropna()
d = pd.DataFrame({"y": r.usdkrw, "x0": r.brent, "x1": r.brent.shift(1), "y1": r.usdkrw.shift(1)}).dropna()
r2 = {}
for name, cols, sub in [("동시점 유가", ["x0"], d), ("전일 유가", ["x1"], d), ("전일 환율", ["y1"], d),
                        ("전일 환율+전일 유가", ["y1", "x1"], d), ("동시점 유가 (2025~)", ["x0"], d.loc["2025-01-01":])]:
    r2[name] = sm.OLS(sub.y, sm.add_constant(sub[cols])).fit().rsquared
r2 = pd.Series(r2, name="R²"); print(r2.to_string()); r2.to_csv(OUT / "t6_r2.csv", encoding="utf-8-sig")
