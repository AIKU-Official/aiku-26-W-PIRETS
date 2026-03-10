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

PIRETS에 대한 더 자세한 설명은 [AIKU 노션](https://www.notion.so/aiku/PIRETS-31da7930e09c80dcba20d8acb16dd63e?source=copy_link)에서 확인하실 수 있습니다.

## 방법론

(문제를 정의하고 이를 해결한 방법을 가독성 있게 설명해주세요)

## 환경 설정

(Requirements, Anaconda, Docker 등 프로젝트를 사용하는데에 필요한 요구 사항을 나열해주세요)

## 사용 방법

(프로젝트 실행 방법 (명령어 등)을 적어주세요.)

## 예시 결과

(사용 방법을 실행했을 때 나타나는 결과나 시각화 이미지를 보여주세요)

## 팀원

(프로젝트에 참여한 팀원의 이름과 깃헙 프로필 링크, 역할을 작성해주세요)

- [홍길동](홍길동의 github link): (수행한 역할을 나열)
