# Python 3.12 설치 가이드

현재 시스템에 Python 3.8이 설치되어 있지만, 이 프로젝트는 Python 3.12 이상이 필요합니다.

## 설치 방법

### 1. Homebrew 사용 (macOS 권장)

```bash
# Homebrew로 Python 3.12 설치
brew install python@3.12

# PATH에 추가 (zsh 사용 시)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Python 3.12 확인
python3.12 --version
```

### 2. pyenv 사용 (여러 Python 버전 관리)

```bash
# pyenv 설치 (아직 설치하지 않은 경우)
brew install pyenv

# Python 3.12 설치
pyenv install 3.12.0

# 프로젝트 디렉토리에서 Python 3.12 사용
cd Search/diner_search
pyenv local 3.12.0

# Python 버전 확인
python --version
```

### 3. 공식 설치 파일 사용

1. [Python 공식 사이트](https://www.python.org/downloads/)에서 Python 3.12 다운로드
2. 설치 파일 실행
3. "Add Python to PATH" 옵션 체크

### 4. conda 사용

```bash
# conda로 Python 3.12 환경 생성
conda create -n diner-search python=3.12
conda activate diner-search
```

## 설치 후 확인

Python 3.12 설치 후 다음 명령어로 확인:

```bash
# 버전 확인
python3.12 --version

# 또는 프로젝트 스크립트 사용
cd Search/diner_search
./scripts/check_python_version.sh
```

## uv와 함께 사용

Python 3.12 설치 후 uv를 사용하여 프로젝트 설정:

```bash
# uv 설치 (아직 설치하지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 설정
cd Search/diner_search
make setup
```

## 문제 해결

### PATH 문제
Python 3.12가 설치되었지만 `python3` 명령어가 여전히 3.8을 가리키는 경우:

```bash
# 현재 Python 경로 확인
which python3
which python3.12

# 심볼릭 링크 생성 (필요한 경우)
sudo ln -sf /opt/homebrew/bin/python3.12 /usr/local/bin/python3
```

### 권한 문제
설치 중 권한 오류가 발생하는 경우:

```bash
# Homebrew 권한 수정
sudo chown -R $(whoami) /opt/homebrew

# 또는 pyenv 사용 권장
```

## 다음 단계

Python 3.12 설치 완료 후:

1. 프로젝트 디렉토리로 이동
2. `make setup` 실행
3. `make run`으로 애플리케이션 실행 