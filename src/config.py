"""프로젝트 경로와 분석 설정."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """.env 의 KEY=VALUE 를 환경변수로 읽는다 (python-dotenv 없이)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# 분석 기간과 국면 구분 (하위 구간 비교용)
START = "2010-01-01"
REGIMES = {
    "셰일 확대기": ("2010-01-01", "2014-06-30"),
    "유가 급락·저유가": ("2014-07-01", "2019-12-31"),
    "코로나·회복": ("2020-01-01", "2021-12-31"),
    "강달러·긴축": ("2022-01-01", "2024-12-31"),
    "최근": ("2025-01-01", None),
}

# 과거 유가 충격 프리셋 (Brent 고점·저점 기준, 데이터에서 확인한 날짜)
SHOCKS = {
    "2014 셰일발 유가 급락": ("2014-06-19", "2015-01-13"),
    "2020 코로나 유가 폭락": ("2020-01-06", "2020-04-21"),
    "2022 러시아-우크라이나 전쟁": ("2022-02-01", "2022-03-08"),
    "2026 유가 급등": ("2025-12-16", "2026-04-07"),
}
