"""
유틸리티 함수들
"""

import re

from fuzzywuzzy import fuzz
from jamo import hangul_to_jamo


def normalize(text: str) -> str:
    """
    텍스트를 정규화합니다.

    Args:
        text: 정규화할 텍스트

    Returns:
        정규화된 텍스트
    """
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", text.lower().strip())


def jamo_similarity(a: str, b: str) -> tuple[bool, float]:
    """
    두 문자열의 자모 유사도를 계산합니다.

    Args:
        a: 첫 번째 문자열
        b: 두 번째 문자열

    Returns:
        (자모 매칭 여부, 유사도 점수)
    """
    a_jamo = " ".join(hangul_to_jamo(a))
    b_jamo = " ".join(hangul_to_jamo(b))
    score = fuzz.ratio(a_jamo, b_jamo)

    if score > 90:
        return True, score
    else:
        matches = sum(x == y for x, y in zip(a_jamo, b_jamo, strict=False))
        return False, matches / max(len(a_jamo), len(b_jamo), 1)


def load_diner_data(file_path: str = "../data/diner_infos.json"):
    """
    음식점 데이터를 로드합니다.

    Args:
        file_path: 데이터 파일 경로

    Returns:
        음식점 정보 리스트
    """
    import json
    import os

    # 현재 파일의 디렉토리를 기준으로 상대 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, file_path)

    with open(data_path, encoding="utf-8") as f:
        return json.load(f)
