# 🚀 Hugging Face Spaces 배포 가이드

이 가이드는 음식점 검색 시스템을 Hugging Face Spaces에 배포하는 방법을 설명합니다.

## 📋 사전 준비

1. **Hugging Face 계정 생성**
   - [Hugging Face](https://huggingface.co)에서 계정 생성
   - 이메일 인증 완료

2. **필요한 파일들 확인**
   - ✅ `app.py` (통합 앱 파일)
   - ✅ `pyproject.toml` (의존성)
   - ✅ `README_HF.md` (Space 설명)
   - ✅ `data/diner_infos.json` (음식점 데이터)
   - ✅ `src/` 폴더 (소스 코드)

## 🎯 배포 단계

### 1단계: 새 Space 생성

1. Hugging Face에 로그인
2. 우측 상단의 "New Space" 클릭
3. 다음 설정으로 Space 생성:
   - **Owner**: 본인의 사용자명
   - **Space name**: `diner-search`
   - **SDK**: `Gradio`
   - **License**: `MIT`
   - **Space hardware**: `CPU` (무료)

### 2단계: 파일 업로드

다음 파일들을 순서대로 업로드:

1. **`app.py`** (통합 앱 파일)
2. **`pyproject.toml`** (의존성)
3. **`README_HF.md`** → **`README.md`**로 이름 변경하여 업로드
4. **`data/diner_infos.json`** (음식점 데이터)
5. **`src/`** 폴더 전체

### 3단계: 배포 확인

1. 파일 업로드 후 자동으로 빌드 시작
2. 빌드 로그 확인 (오류가 있는지 체크)
3. 배포 완료 후 공개 URL 확인

## 🔧 파일 구조

배포 후 Space의 파일 구조:

```
diner-search/
├── app.py                 # 통합 앱 파일
├── pyproject.toml         # Python 의존성 (uv)
├── README.md             # Space 설명 (README_HF.md에서 변경)
├── data/
│   └── diner_infos.json  # 음식점 데이터
└── src/
    ├── __init__.py
    ├── search_engine.py  # 검색 엔진
    ├── utils.py          # 유틸리티
    └── legacy_code.py    # 기존 코드
```

## 🐛 문제 해결

### 일반적인 오류들

1. **모듈을 찾을 수 없음**
   - `src/` 폴더가 제대로 업로드되었는지 확인
   - `app.py`에서 경로 설정 확인

2. **의존성 설치 실패**
   - `pyproject.toml` 파일 형식 확인
   - 버전 충돌이 있는지 확인

3. **메모리 부족**
   - Space hardware를 더 큰 것으로 업그레이드
   - 모델 로딩 최적화

### 로그 확인 방법

1. Space 페이지에서 "Settings" 탭 클릭
2. "Build logs" 섹션에서 빌드 로그 확인
3. 오류 메시지 분석 및 수정

## 🔄 업데이트 방법

코드 변경 후 업데이트:

1. 로컬에서 수정된 파일들 준비
2. Space의 "Files" 탭에서 파일 수정
3. 자동으로 재빌드 및 재배포

## 📊 성능 최적화

1. **모델 캐싱**
   - SBERT 모델을 한 번만 로드
   - 임베딩 미리 계산

2. **메모리 사용량 최적화**
   - 불필요한 데이터 제거
   - 배치 처리 활용

3. **응답 시간 개선**
   - 검색 알고리즘 최적화
   - 인덱싱 활용

## 🌐 공개 URL

배포 완료 후 제공되는 URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/diner-search
```

## 📞 지원

문제가 발생하면:
1. Space의 "Discussion" 탭에서 질문
2. GitHub 이슈 생성
3. Hugging Face 커뮤니티 포럼 활용

## 🔧 로컬 개발

### uv 기반 개발 환경

```bash
# 개발 환경 설정
make setup

# 앱 실행
make run

# 코드 품질 검사
make check

# Hugging Face 배포 준비
make prepare-hf
```

### 패키지 관리

```bash
# 패키지 추가
make add PKG=requests

# 개발 패키지 추가
make add-dev PKG=pytest

# 패키지 제거
make remove PKG=requests

# 의존성 트리 확인
make tree
```

---

🎉 **배포 완료!** 이제 전 세계 누구나 음식점 검색 시스템을 사용할 수 있습니다! 