# PIRETS: Phonetic Information Retrieval System

📢 2026년 겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

> **PIRETS**: **P**honetic **I**nformation **Ret**rieval **S**ystem

PIRETS는 텍스트의 발음 차원 노이즈를 극복하는 2-Stage 발음 기반 정보 검색 시스템입니다. 

텍스트에 g2p를 적용하여 발음 기호로 변환한 뒤, Lexical search와 XPhoneBERT 기반 Dense retrieval을 앙상블하고, Cross-encoder로 정밀하게 Reranking하여 높은 검색 정확도를 달성합니다.

PIRETS는 범용적인 발음 기반 정보 검색(Phonetic Information Retrieval)을 위한 프레임워크로 설계되었으며, 본 프로젝트에서는 제안된 방법론의 실효성을 입증하기 위한 Testbed로서 '발음 노이즈가 포함된 가사 쿼리에 대한 원곡 검색 task'에 집중하여 학습과 평가를 수행합니다.

PIRETS는 다음과 같은 목표를 달성하고자 합니다.

- 발음 차원 노이즈가 포함된 쿼리 환경에서도 의도한 문서를 정확하게 탐색할 수 있는 검색 모델 개발
- 후보 passage의 규모가 증가해도 성능이 크게 감소하지 않는 Robust한 검색 모델 개발  



### **Dataset License & Citation**

※ 본 레포지토리에는 저작권 보호 및 라이선스 정책에 따라 학습 및 평가에 사용된 원본 데이터셋이 포함되어 있지 않습니다. 공개 데이터셋은 아래의 공식 링크를 통해 직접 다운로드하실 수 있습니다.

- **ASR + 가사 데이터셋(Private)**: 유튜브 자동자막 및 Web scraping을 통해 구축한 노이즈가 포함된 가사 데이터셋입니다. 저작권 보호 규정에 따라 비공개 처리됩니다.

- [**OpenSubtitles:**](http://www.opensubtitles.org/) 본 프로젝트는 OpenSubtitles-v2024 데이터를 활용하였으며, 원본 데이터의 저작권은 각 창작자에게 있습니다. (P. Lison and J. Tiedemann, 2016, *OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles*) 
    
- [**AI Hub:**](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EB%8C%80%ED%99%94%EC%B2%B4&aihubDataSe=data&dataSetSn=543) 한국어 대화체 데이터 구축을 위해 과학기술정보통신부와 한국지능정보사회진흥원(NIA)이 주관하여 구축한 'AI Hub 주제별 텍스트 일상 대화 데이터'를 활용하였습니다.

PIRETS에 대한 더 자세한 설명은 [AIKU 노션](https://www.notion.so/aiku/PIRETS-31da7930e09c80dcba20d8acb16dd63e?source=copy_link)에서 확인하실 수 있습니다.

## 방법론

(문제를 정의하고 이를 해결한 방법을 가독성 있게 설명해주세요)

## 환경 설정

본 프로젝트는 실행 환경과 선호도에 맞추어 두 가지 세팅 방법을 제공합니다.

### 옵션 A: Conda 가상환경 구축 (Quick Start)

터미널에서 아래 코드를 순차적으로 실행합니다.

```bash
conda create -n <env_name> python=3.9 -y

conda activate <env_name>

pip install -r requirements.txt
```


### 옵션 B: Docker 환경 구축 (Robust Deployment)

프로젝트 루트에 포함된 `Dockerfile`을 사용합니다.

```bash
docker build -t <image_name> .

docker run -it --gpus all --name <container_name> <image_name> /bin/bash
```

## 사용 방법
모든 설정은 conf/ 폴더 내의 YAML 파일에서 관리하며, 실행 시 Hydra를 통해 자동으로 로드됩니다.

### 2-1. 평가 (Evaluation)
저장된 모델의 검색 성능을 측정합니다.  
`conf/eval_config.yaml`과 `conf/model/{model_name}.yaml` 파일을 실험에 맞게 수정 후 다음 명령어 중 하나를 실행합니다.

a) 모델 평가 (기본값)
```
python evaluate.py
```

b) 평가 프로세스 프리즈 대응 (Watchdog 기반 자동 재시작)
평가 도중 프로세스가 응답하지 않는 경우, 중단된 시점을 추적하여 자동으로 이어서 평가를 재시작합니다.

```
python evaluate_watchdog.py
```

### 2-2. 학습 (Training)
모델을 학습하거나 검증(Validation)을 수행합니다.  
`conf/train_config.yaml`(또는 `conf/train_reranker_config.yaml`)과 `conf/model/{model_name}.yaml` 파일을 실험에 맞게 수정 후 다음 명령어 중 하나를 실행합니다.

a) retriever 학습
```
python train.py
```

b) reranker 학습
```
python train_reranker.py
```

## 예시 결과

(사용 방법을 실행했을 때 나타나는 결과나 시각화 이미지를 보여주세요)

## 팀원

- [이성은](https://github.com/retnivv): 모델 구현, 학습 및 평가 수행
- [이재승](https://github.com/j-seui): 베이스라인 실험 진행, 코드 정리
- [임시은](https://github.com/bbiibb): 베이스라인 실험 진행, 실험 결과 정리
- [천권욱](https://github.com/KwCCCC): 데이터셋 구축, 코드 정리
