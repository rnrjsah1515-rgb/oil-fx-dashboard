"""원천 시계열을 분석용 일별·월별 데이터셋으로 만든다.

정제 원칙
- 결측을 앞 값으로 채우지 않는다. 채우면 수익률이 0인 날이 인위적으로 생겨 상관이 과소추정된다.
  대신 유가와 환율이 모두 관측된 날만 남긴다 (inner join).
- 가격이 0 이하인 관측(2020-04-20 WTI 마이너스 유가)은 로그수익률을 정의할 수 없어 NaN 처리하고 기록한다.
- 월별은 일별 평균으로 만든다 (ECOS 월별 유가와 같은 정의).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ROOT, START, load_env
from .sources import fred

PROCESSED = ROOT / "data" / "processed"


@dataclass
class Dataset:
    daily: pd.DataFrame              # 가격 수준: brent, wti, usdkrw (+ dubai 는 월별만)
    monthly: pd.DataFrame
    log: list[str] = field(default_factory=list)   # 정제 기록 (README·면접 설명용)

    def returns(self, freq: str = "D") -> pd.DataFrame:
        """로그수익률. 날짜 간격이 벌어져도(휴일) 관측 간 변화로 본다."""
        df = self.daily if freq == "D" else self.monthly
        return np.log(df).diff().dropna(how="all")


def _load_dubai(log: list[str]) -> pd.Series | None:
    load_env()
    key = os.environ.get("ECOS_API_KEY")
    if not key:
        log.append("ECOS_API_KEY 없음 → 두바이유(월별) 생략")
        return None
    from .sources import ecos
    s = ecos.get_oil_monthly(key, "010102", START[:7].replace("-", ""), pd.Timestamp.today().strftime("%Y%m"))
    return s.rename("dubai")


def build(refresh: bool = False) -> Dataset:
    log: list[str] = []
    raw = {name: fred.get_series(name, START, refresh) for name in fred.SERIES}

    for name, s in raw.items():
        log.append(f"{name}: 원천 {len(s):,}행, 결측 {s.isna().sum():,}행 "
                   f"({s.index.min().date()} ~ {s.index.max().date()})")

    nonpos = raw["wti"][raw["wti"] <= 0]
    for d, v in nonpos.items():
        log.append(f"wti {d.date()} 가격 {v:.2f} ≤ 0 → NaN 처리 (로그수익률 정의 불가)")
    raw["wti"] = raw["wti"].where(raw["wti"] > 0)

    df = pd.concat(raw, axis=1)
    both = df.dropna(subset=["brent", "usdkrw"])
    log.append(f"Brent·환율 동시 관측일만 사용: {len(df):,} → {len(both):,}일 "
               f"(휴일 불일치 {len(df) - len(both):,}일 제외, 앞값 채움 없음)")

    monthly = both.resample("MS").mean()
    last = both.index[-1]
    if last + pd.offsets.BDay(1) <= last + pd.offsets.BMonthEnd(0):   # 마지막 영업일 전 → 미완결 월
        monthly = monthly.iloc[:-1]
        log.append(f"{last:%Y-%m} 은 {last.date()} 까지만 관측 → 월별에서 제외")
    dubai = _load_dubai(log)
    if dubai is not None:
        monthly = monthly.join(dubai, how="left")

    return Dataset(daily=both, monthly=monthly, log=log)


def save(ds: Dataset) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ds.daily.to_csv(PROCESSED / "daily.csv")
    ds.monthly.to_csv(PROCESSED / "monthly.csv")
    (PROCESSED / "cleaning_log.txt").write_text("\n".join(ds.log), encoding="utf-8")
