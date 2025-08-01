.PHONY: help install dev-install run format lint type-check test clean

# 기본 타겟
help: ## 도움말 표시
	@echo "🍽️ 음식점 검색 시스템 - 사용 가능한 명령어:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 설치 관련
install: ## 프로덕션 의존성 설치
	uv sync

dev-install: ## 개발 의존성 포함 설치
	uv sync --dev

# 실행
run: ## Streamlit 애플리케이션 실행
	uv run streamlit run src/app.py

# 코드 품질
format: ## 코드 포맷팅
	uv run ruff format .

lint: ## 코드 린팅
	uv run ruff check .

type-check: ## 타입 체크
	uv run mypy src/

check: format lint type-check ## 모든 코드 품질 검사 실행

# 테스트
test: ## 테스트 실행
	uv run pytest

test-cov: ## 커버리지 포함 테스트
	uv run pytest --cov=src --cov-report=html

# 패키지 관리
add: ## 패키지 추가 (사용법: make add PKG=package_name)
	uv add $(PKG)

add-dev: ## 개발 패키지 추가 (사용법: make add-dev PKG=package_name)
	uv add --dev $(PKG)

remove: ## 패키지 제거 (사용법: make remove PKG=package_name)
	uv remove $(PKG)

tree: ## 의존성 트리 표시
	uv tree

# 정리
clean: ## 캐시 및 임시 파일 정리
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

# 의존성 내보내기
export-prod: ## 프로덕션용 requirements.txt 생성
	uv export --no-dev --format requirements.txt > requirements.txt

export-dev: ## 개발용 requirements.txt 생성
	uv export --format requirements.txt > requirements.txt

# 개발 환경 설정
setup: ## 개발 환경 초기 설정
	@echo "🍽️ 개발 환경을 설정합니다..."
	@chmod +x scripts/setup.sh
	@chmod +x scripts/check_python_version.sh
	@./scripts/setup.sh

check-python: ## Python 버전 확인
	@chmod +x scripts/check_python_version.sh
	@./scripts/check_python_version.sh 