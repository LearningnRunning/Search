#!/bin/bash

# 음식점 검색 시스템 설정 스크립트

set -e

echo "🍽️ 음식점 검색 시스템 설정을 시작합니다..."

# Python 버전 확인
echo "🐍 Python 버전을 확인합니다..."
./scripts/check_python_version.sh

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "❌ uv가 설치되지 않았습니다."
    echo "다음 명령어로 uv를 설치해주세요:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "또는: brew install uv"
    exit 1
fi

echo "✅ uv가 설치되어 있습니다."

# 프로젝트 의존성 설치
echo "📦 프로젝트 의존성을 설치합니다..."
uv sync --dev

echo "✅ 의존성 설치가 완료되었습니다."

# 개발 도구 설정
echo "🔧 개발 도구를 설정합니다..."

# pre-commit 훅 설정 (선택사항)
if command -v pre-commit &> /dev/null; then
    echo "📝 pre-commit 훅을 설정합니다..."
    uv run pre-commit install
else
    echo "ℹ️ pre-commit이 설치되지 않았습니다. (선택사항)"
fi

echo "🎉 설정이 완료되었습니다!"
echo ""
echo "다음 명령어로 애플리케이션을 실행할 수 있습니다:"
echo "uv run streamlit run src/app.py"
echo ""
echo "개발 도구 사용법:"
echo "- 코드 포맷팅: uv run ruff format ."
echo "- 린팅: uv run ruff check ."
echo "- 타입 체크: uv run mypy src/" 