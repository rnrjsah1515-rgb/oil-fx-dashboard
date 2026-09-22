"""헤지 계산기: 만기 환율 분포(시나리오 결과) 아래에서 헤지 비율별 원화 금액 분포.

- 수입업체(달러 지급): 원/달러 상승 = 손실.  수출업체(달러 수취): 원/달러 하락 = 손실.
- 헤지 수단은 통화선물/선물환 매수(수입) 또는 매도(수출). 선물환율 = 현물 + 스왑포인트.
- 손익은 '오늘 현물로 환산한 원화 금액(예산)' 대비로 계산한다. 증거금·거래비용은 무시.
"""
from __future__ import annotations

import pandas as pd


def exposure(amount_usd: float, side: str, spot: float, fwd: float, fx_quantiles: dict[str, float],
             ratios=(0.0, 0.5, 1.0)) -> pd.DataFrame:
    """fx_quantiles: {'95% 불리': S, '중심': S, ...} 만기 환율. 반환: 헤지비율 × 시나리오별 예산 대비 손익(원)."""
    sign = -1 if side == "import" else 1          # 수입: 환율 상승 시 원화 지출 증가 → 손익 음(-)
    budget = amount_usd * spot
    rows = []
    for h in ratios:
        for name, s_t in fx_quantiles.items():
            eff = h * fwd + (1 - h) * s_t        # 헤지분은 선물환율로 고정
            rows.append({"헤지 비율": f"{h:.0%}", "시나리오": name, "적용 환율": eff,
                         "예산 대비 손익(원)": sign * (eff * amount_usd - budget)})
    return pd.DataFrame(rows)
