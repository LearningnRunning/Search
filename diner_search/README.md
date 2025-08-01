# 음식점 검색 시스템

Streamlit을 사용한 음식점 이름 검색 시스템입니다.

## 기능

- 정확한 매칭 (Exact Match)
- 부분 매칭 (Partial Match)
- 자모 기반 유사도 검색 (Jamo-based Search)
- 의미론적 검색 (Semantic Search)
- 통합 검색 (Combined Search)

## 개발 환경

### 전제 조건
- Python 3.12 이상
- uv 패키지 관리 도구

### Python 3.12 설치
```bash
# macOS (Homebrew)
brew install python@3.12

# pyenv 사용
pyenv install 3.12.0
pyenv local 3.12.0

# 공식 사이트
# https://www.python.org/downloads/
```

### uv 설치
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew 사용
brew install uv
```

## 설치 및 실행

### 1. 프로젝트 클론
```bash
git clone <repository-url>
cd diner_search
```

### 2. Python 버전 확인
```bash
# Python 버전 확인
make check-python

# 또는 직접 확인
./scripts/check_python_version.sh
```

### 3. 의존성 설치
```bash
# 개발 환경 동기화 (가상환경 생성 + 의존성 설치)
uv sync

# 또는 개발 의존성 포함
uv sync --dev
```

### 4. 애플리케이션 실행
```bash
# uv로 실행 (가상환경 자동 활성화)
uv run streamlit run src/app.py

# 또는 가상환경 활성화 후 실행
source .venv/bin/activate
streamlit run src/app.py
```

## 개발 도구

### 패키지 관리
```bash
# 의존성 패키지 목록 확인
uv tree

# 패키지 추가
uv add requests

# 개발 의존성 패키지 추가
uv add --dev ruff

# 패키지 삭제
uv remove requests
```

### 코드 품질 관리
```bash
# 코드 포맷팅
uv run ruff format .

# 린팅
uv run ruff check .

# 타입 체크
uv run mypy src/
```

### 테스트
```bash
# 테스트 실행
uv run pytest

# 커버리지 포함 테스트
uv run pytest --cov=src
```

## 프로젝트 구조

```
diner_search/
├── data/
│   └── diner_infos.json      # 음식점 데이터
├── scripts/
│   ├── setup.sh              # 자동 설정 스크립트
│   └── check_python_version.sh # Python 버전 확인
├── src/
│   ├── __init__.py           # 패키지 초기화
│   ├── app.py               # Streamlit 메인 애플리케이션
│   ├── search_engine.py     # 검색 엔진 클래스
│   ├── utils.py             # 유틸리티 함수
│   └── legacy_code.py       # 기존 검색 로직 (참조용)
├── .gitignore               # Git 무시 파일
├── .pre-commit-config.yaml  # pre-commit 설정
├── .python-version          # Python 버전 명시
├── Makefile                 # 개발 작업 자동화
├── pyproject.toml           # 프로젝트 설정 및 의존성
├── README.md                # 프로젝트 설명
└── requirements.txt         # pip 호환성용 (선택사항)
```

## 사용법

1. 웹 브라우저에서 `http://localhost:8501` 접속
2. 검색창에 음식점 이름 입력
3. 검색 결과 확인 (매칭 타입별로 구분됨)

## Python 3.12의 장점

- **향상된 성능**: Python 3.12는 이전 버전보다 5-10% 빠른 실행 속도
- **더 나은 오류 메시지**: 더 명확하고 유용한 오류 메시지
- **타입 힌트 개선**: 더 강력한 타입 시스템 지원
- **최신 라이브러리 지원**: 최신 패키지들과의 호환성 향상

## 의존성 내보내기

```bash
# 프로덕션용 requirements.txt 생성
uv export --no-dev --format requirements.txt > requirements.txt

# 개발용 requirements.txt 생성 (dev 포함)
uv export --format requirements.txt > requirements.txt
```

## 기존 pip 사용자

uv 없이 pip를 사용하려면:
```bash
# requirements.txt 생성 후
pip install -r requirements.txt
```

**주의**: 가상환경이 적용되지 않으면 시스템 전역 Python에 설치될 수 있습니다. 