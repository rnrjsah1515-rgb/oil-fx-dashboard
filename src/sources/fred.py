"""FRED(세인트루이스 연준) 시계열. 인증키 없이 CSV 엔드포인트로 받는다.

  DCOILBRENTEU  Brent 현물 (달러/배럴, 일별, 유럽 기준)
  DCOILWTICO    WTI 현물 (달러/배럴, 일별, 쿠싱 기준)
  DEXKOUS       원/달러 (원/1달러, 일별, 뉴욕 정오 매입률)
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .cache import cached_json

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "brent": "DCOILBRENTEU",
    "wti": "DCOILWTICO",
    "usdkrw": "DEXKOUS",
}


def _fetch(series_id: str, start: str) -> list[list]:
    resp = requests.get(CSV_URL, params={"id": series_id, "cosd": start}, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df.astype(str).values.tolist()  # [[날짜, 값], ...] 결측은 '.' 또는 'nan'


def get_series(name: str, start: str, refresh: bool = False) -> pd.Series:
    """name: SERIES 의 키. start: YYYY-MM-DD. 결측은 NaN 으로 남긴다."""
    sid = SERIES[name]
    payload = cached_json("fred", {"id": sid, "start": start}, lambda: _fetch(sid, start), refresh)
    rows = payload["data"]
    s = pd.Series(
        pd.to_numeric([r[1] for r in rows], errors="coerce"),
        index=pd.to_datetime([r[0] for r in rows]),
        name=name,
    )
    return s.sort_index()
