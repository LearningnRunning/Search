---
title: 🍽️ 음식점 검색 시스템
emoji: 🍽️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🍽️ 음식점 검색 시스템

다양한 검색 방법을 활용한 한국어 음식점 검색 시스템입니다.

## ✨ 주요 기능

- **정확한 매칭**: 완전히 일치하는 음식점 검색
- **부분 매칭**: 검색어가 포함된 음식점 검색  
- **자모 매칭**: 한글 자모 유사도 기반 검색
- **의미론적 검색**: SBERT 모델을 활용한 의미 기반 검색
- **통합 검색**: 여러 검색 방법을 조합한 최적화된 결과

## 🔍 사용법

1. 검색창에 음식점 이름을 입력하세요
2. 검색 설정을 조정하세요 (결과 수, 자모 임계값)
3. "🔍 검색" 버튼을 클릭하거나 Enter를 누르세요
4. 검색 결과를 확인하세요

## 📊 검색 방법

- **정확한 매칭**: "윤씨네" → "윤씨네"
- **부분 매칭**: "피자" → "피자헛", "도미노피자"
- **자모 매칭**: "윤씨네" → "윤시네", "윤씨내"
- **의미론적 검색**: "맛있는 집" → "맛집", "좋은 음식점"

## 🛠️ 기술 스택

- **Frontend**: Gradio
- **Backend**: Python 3.12+
- **ML/NLP**: Sentence Transformers, PyTorch, Jamo
- **Data Processing**: Pandas
- **Fuzzy Matching**: FuzzyWuzzy, python-Levenshtein 