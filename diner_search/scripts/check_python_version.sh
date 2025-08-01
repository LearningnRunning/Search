#!/bin/bash

# Python 3.12 버전 확인 스크립트

set -e

echo "🐍 Python 버전을 확인합니다..."

# Python 버전 확인
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.12"

echo "현재 Python 버전: $python_version"
echo "필요한 Python 버전: $required_version 이상"

# 버전 비교 함수
version_compare() {
    if [[ $1 == $2 ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2
        fi
    done
    return 0
}

# 버전 체크
version_compare "$python_version" "$required_version"
case $? in
    0) echo "✅ Python 버전이 정확합니다." ;;
    1) echo "✅ Python 버전이 요구사항을 충족합니다." ;;
    2) 
        echo "❌ Python 버전이 너무 낮습니다."
        echo "Python 3.12 이상을 설치해주세요."
        echo ""
        echo "설치 방법:"
        echo "1. pyenv 사용: pyenv install 3.12.0"
        echo "2. Homebrew 사용: brew install python@3.12"
        echo "3. 공식 사이트: https://www.python.org/downloads/"
        exit 1
        ;;
esac

echo ""
echo "Python 환경이 준비되었습니다! 🎉" 