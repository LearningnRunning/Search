# 음식점 검색 시스템 - Makefile (uv 기반)

.PHONY: help install run test lint format clean check-python setup prepare-hf generate-embeddings verify-embeddings

# 기본 타겟
help:
	@echo "🍽️ 음식점 검색 시스템 - 사용 가능한 명령어:"
	@echo ""
	@echo "  install     - uv로 의존성 설치"
	@echo "  run         - Gradio 앱 실행"
	@echo "  test        - 테스트 실행"
	@echo "  lint        - 코드 린팅"
	@echo "  format      - 코드 포맷팅"
	@echo "  clean       - 캐시 및 임시 파일 정리"
	@echo "  check-python - Python 버전 확인"
	@echo "  setup       - 개발 환경 전체 설정"
	@echo "  prepare-hf  - Hugging Face Spaces 배포 준비"
	@echo ""
	@echo "🔢 벡터 관리:"
	@echo "  generate-embeddings - 음식점 이름 벡터 생성"
	@echo "  verify-embeddings   - 생성된 벡터 파일 검증"
	@echo ""

# Python 버전 확인
check-python:
	@echo "🐍 Python 버전 확인 중..."
	@python --version
	@python -c "import sys; assert sys.version_info >= (3, 11), f'Python 3.11+ 필요, 현재: {sys.version}'"
	@echo "✅ Python 버전 확인 완료"

# uv로 의존성 설치
install: check-python
	@echo "📦 uv로 의존성 설치 중..."
	uv sync
	@echo "✅ 의존성 설치 완료"

# 개발 의존성 포함 설치
install-dev: check-python
	@echo "📦 개발 의존성 포함 설치 중..."
	uv sync --dev
	@echo "✅ 개발 의존성 설치 완료"

# Gradio 앱 실행
run: install
	@echo "🚀 Gradio 앱 실행 중..."
	uv run gradio app.py

# 테스트 실행
test: install-dev
	@echo "🧪 테스트 실행 중..."
	uv run pytest tests/ -v

# 코드 린팅
lint: install-dev
	@echo "🔍 코드 린팅 중..."
	uv run ruff check src/ tests/

# 코드 포맷팅
format: install-dev
	@echo "✨ 코드 포맷팅 중..."
	uv run ruff format src/ tests/

# 타입 체크
type-check: install-dev
	@echo "🔍 타입 체크 중..."
	uv run mypy src/

# 모든 코드 품질 검사
check: format lint type-check
	@echo "✅ 모든 코드 품질 검사 완료"

# 캐시 및 임시 파일 정리
clean:
	@echo "🧹 정리 중..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 정리 완료"

# 개발 환경 전체 설정
setup: clean install-dev
	@echo "🎉 개발 환경 설정 완료!"
	@echo "다음 명령어로 앱을 실행하세요:"
	@echo "  make run"

# 음식점 이름 벡터 생성
generate-embeddings: install-dev
	@echo "🔢 음식점 이름 벡터 생성 중..."
	uv run python scripts/generate_embeddings.py
	@echo "✅ 벡터 생성 완료!"

# 벡터 파일 검증
verify-embeddings: install-dev
	@echo "🔍 벡터 파일 검증 중..."
	uv run python scripts/generate_embeddings.py --verify
	@echo "✅ 벡터 검증 완료!"

# Hugging Face Spaces 배포 준비
prepare-hf:
	@echo "🚀 Hugging Face Spaces 배포 준비 중..."
	@echo "다음 파일들이 준비되었습니다:"
	@echo "  - app.py (통합 앱)"
	@echo "  - pyproject.toml (의존성)"
	@echo "  - README_HF.md (Space 설명)"
	@echo "  - data/diner_infos.json (음식점 데이터)"
	@echo "  - data/embeddings/ (벡터 파일들)"
	@echo "  - src/ (소스 코드)"
	@echo ""
	@echo "Hugging Face Spaces에 업로드할 파일들:"
	@echo "1. app.py"
	@echo "2. pyproject.toml"
	@echo "3. README_HF.md → README.md로 이름 변경"
	@echo "4. data/diner_infos.json"
	@echo "5. data/embeddings/ 폴더 전체"
	@echo "6. src/ 폴더 전체"

# 패키지 관리
add:
	@echo "📦 패키지 추가 (사용법: make add PKG=package_name)"
	@if [ -z "$(PKG)" ]; then echo "❌ PKG 변수를 설정해주세요. 예: make add PKG=requests"; exit 1; fi
	uv add $(PKG)

add-dev:
	@echo "📦 개발 패키지 추가 (사용법: make add-dev PKG=package_name)"
	@if [ -z "$(PKG)" ]; then echo "❌ PKG 변수를 설정해주세요. 예: make add-dev PKG=pytest"; exit 1; fi
	uv add --dev $(PKG)

remove:
	@echo "🗑️ 패키지 제거 (사용법: make remove PKG=package_name)"
	@if [ -z "$(PKG)" ]; then echo "❌ PKG 변수를 설정해주세요. 예: make remove PKG=requests"; exit 1; fi
	uv remove $(PKG)

# 의존성 트리 표시
tree:
	@echo "🌳 의존성 트리:"
	uv tree

# requirements.txt 생성 (Hugging Face Spaces용)
export-requirements:
	@echo "📄 requirements.txt 생성 중..."
	uv export --format requirements.txt > requirements.txt
	@echo "✅ requirements.txt 생성 완료" 