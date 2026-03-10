# 1. 베이스 이미지: PyTorch 공식 이미지 사용
# (CUDA, cuDNN 등 딥러닝 환경이 이미 세팅되어 있음)
FROM pytorch/pytorch:latest

# 3. 환경 변수 설정 (파이썬 출력 버퍼링 비활성화 - 로그 즉시 확인용)
ENV PYTHONUNBUFFERED=1

# 4. 작업 디렉토리 설정
# (컨테이너 접속 시 이곳이 기본 위치가 됩니다)
WORKDIR /workspace

# 5. 시스템 패키지 설치
# (git, vim, curl 등 필수 도구 설치)
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 6. 파이썬 라이브러리 설치 (가장 중요한 부분!)
# requirements.txt만 먼저 복사해서 설치합니다.
# 이유: 코드가 바뀔 때마다 라이브러리를 다시 설치하지 않도록 '캐시'를 쓰기 위함입니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. 소스 코드 복사
# .dockerignore에 의해 data, outputs 폴더는 자동으로 제외되고
# 나머지 코드와 conf/data 폴더만 복사됩니다.
COPY . .

# 8. 컨테이너 실행 시 기본 명령어 (bash 쉘 실행)
CMD ["/bin/bash"]