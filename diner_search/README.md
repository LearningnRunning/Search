# 🍽️ 음식점 검색 시스템

다양한 검색 방법을 활용한 한국어 음식점 검색 시스템입니다. 정확한 매칭, 부분 매칭, 자모 유사도, 의미론적 검색을 조합하여 사용자가 원하는 음식점을 효과적으로 찾을 수 있습니다.

## ✨ 주요 기능

- **정확한 매칭**: 완전히 일치하는 음식점 검색
- **부분 매칭**: 검색어가 포함된 음식점 검색
- **자모 매칭**: 한글 자모 유사도 기반 검색
- **의미론적 검색**: SBERT 모델을 활용한 의미 기반 검색
- **통합 검색**: 여러 검색 방법을 조합한 최적화된 결과
- **벡터 캐싱**: 미리 계산된 벡터로 빠른 검색

## 🚀 Hugging Face Spaces 배포

이 프로젝트는 Hugging Face Spaces에 배포되어 있습니다:

**[🌐 라이브 데모 보기](https://huggingface.co/spaces/YOUR_USERNAME/diner-search)**

## 🛠️ 로컬 실행

### 1. 환경 설정

```bash
# Python 3.11 이상 필요
python --version

# uv 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew 사용
brew install uv
```

### 2. 프로젝트 설정

```bash
# 프로젝트 클론
git clone <repository-url>
cd diner_search

# 개발 환경 설정
make setup
```

### 3. 벡터 생성 (선택사항)

앱 시작 시간을 단축하려면 음식점 이름의 벡터를 미리 생성하세요:

```bash
# 벡터 생성 (처음 한 번만 실행)
make generate-embeddings

# 생성된 벡터 검증
make verify-embeddings
```

**참고**: 벡터 생성에는 시간이 걸릴 수 있습니다 (약 46만 개 음식점).

### 4. 앱 실행

```bash
# Gradio 앱 실행
make run

# 또는 직접 실행
uv run python app.py
```

### 5. 브라우저에서 접속

```
http://localhost:7860
```

## 📁 프로젝트 구조

```
diner_search/
├── app.py                 # 🚀 통합 Gradio 앱 (로컬 + 배포용)
├── pyproject.toml         # 프로젝트 설정 및 의존성 (uv)
├── uv.lock               # 의존성 잠금 파일
├── src/
│   ├── search_engine.py  # 검색 엔진 구현
│   ├── embedding_loader.py # 벡터 로더
│   ├── utils.py          # 유틸리티 함수
│   └── __init__.py
├── data/
│   ├── diner_infos.json  # 음식점 데이터
│   └── embeddings/       # 미리 계산된 벡터 (자동 생성)
│       ├── diner_embeddings.pt
│       ├── diner_embeddings.pkl
│       └── diner_metadata.json
├── scripts/
│   └── generate_embeddings.py # 벡터 생성 스크립트
├── README.md             # 프로젝트 설명
├── README_HF.md          # Hugging Face Spaces 설명
├── DEPLOYMENT_GUIDE.md   # 배포 가이드
└── Makefile              # 개발 작업 자동화 (uv 기반)
```

## 🔧 기술 스택

- **Frontend**: Gradio
- **Backend**: Python 3.11+
- **Package Manager**: uv
- **ML/NLP**: 
  - Sentence Transformers (SBERT)
  - PyTorch
  - Jamo (한글 자모 처리)
- **Data Processing**: Pandas
- **Fuzzy Matching**: FuzzyWuzzy, python-Levenshtein
- **Vector Storage**: PyTorch (.pt), Pickle (.pkl)

## 📊 검색 방법

### 1. 정확한 매칭
- 검색어와 음식점 이름이 완전히 일치하는 경우
- 예: "윤씨네" → "윤씨네"

### 2. 부분 매칭
- 검색어가 음식점 이름에 포함된 경우
- 예: "피자" → "피자헛", "도미노피자"

### 3. 자모 매칭
- 한글 자모 유사도를 기반으로 한 검색
- 발음이 비슷한 음식점을 찾을 수 있음
- 예: "윤씨네" → "윤시네", "윤씨내"

### 4. 의미론적 검색
- SBERT 모델을 활용한 의미 기반 검색
- 유사한 의미의 음식점을 찾을 수 있음
- 예: "맛있는 집" → "맛집", "좋은 음식점"

### 5. 통합 검색
- 자모 유사도와 의미론적 검색을 조합
- 더 정확하고 다양한 결과 제공

## 🎯 사용 예시

```
검색어: "윤씨네"
결과:
- 윤씨네 (정확한 매칭)
- 윤시네 (자모 매칭)
- 윤씨내 (자모 매칭)

검색어: "피자"
결과:
- 피자헛 (부분 매칭)
- 도미노피자 (부분 매칭)
- 피자나라 (부분 매칭)

검색어: "맛있는 집"
결과:
- 맛집 (의미론적 매칭)
- 좋은 음식점 (의미론적 매칭)
- 맛있는집 (정확한 매칭)
```

## 🔧 설정 옵션

- **검색 결과 수**: 1-20개 (기본값: 5개)
- **자모 유사도 임계값**: 0.1-1.0 (기본값: 0.7)

## 🛠️ 개발 도구

### uv 명령어

```bash
# 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --dev

# 앱 실행
uv run python app.py

# 패키지 추가
uv add requests

# 개발 패키지 추가
uv add --dev pytest

# 패키지 제거
uv remove requests

# 의존성 트리 확인
uv tree
```

### Make 명령어

```bash
# 도움말
make help

# 개발 환경 설정
make setup

# 앱 실행
make run

# 코드 품질 검사
make check

# 테스트 실행
make test

# 벡터 생성
make generate-embeddings

# 벡터 검증
make verify-embeddings

# Hugging Face 배포 준비
make prepare-hf
```

## 🔢 벡터 관리

### 벡터 생성

```bash
# 기본 설정으로 벡터 생성
make generate-embeddings

# 또는 직접 실행
uv run python scripts/generate_embeddings.py

# 배치 크기 조정
uv run python scripts/generate_embeddings.py --batch-size 64

# 다른 모델 사용
uv run python scripts/generate_embeddings.py --model "sentence-transformers/all-MiniLM-L6-v2"
```

### 벡터 검증

```bash
# 생성된 벡터 파일 검증
make verify-embeddings

# 또는 직접 실행
uv run python scripts/generate_embeddings.py --verify
```

### 벡터 파일 형식

- **`.pt`**: PyTorch 텐서 형식 (빠른 로딩)
- **`.pkl`**: Pickle 압축 형식 (작은 파일 크기)
- **`.json`**: 메타데이터 (음식점 정보, 모델 정보)

## 🚀 Hugging Face Spaces 배포 방법

1. **Hugging Face 계정 생성**
   - [Hugging Face](https://huggingface.co)에서 계정 생성

2. **새 Space 생성**
   - "New Space" 클릭
   - Space 이름: `diner-search`
   - SDK: Gradio 선택
   - License: MIT 선택

3. **파일 업로드**
   - `app.py` (통합 앱 파일)
   - `pyproject.toml` (의존성)
   - `README_HF.md` → `README.md`로 이름 변경
   - `data/diner_infos.json` (음식점 데이터)
   - `data/embeddings/` 폴더 전체 (벡터 파일들)
   - `src/` 폴더 전체

4. **배포 완료**
   - 자동으로 빌드 및 배포됨
   - 공개 URL 제공

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

---

🍽️ **음식점 검색 시스템** | 다양한 검색 방법으로 원하는 음식점을 찾아보세요! 