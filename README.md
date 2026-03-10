# Lyrics Retrieval System

음소(Phoneme) 기반 인코딩을 활용한 가사 검색 시스템. G2P 변환을 통해 발음 유사도 기반 검색을 지원한다.

---

## 프로젝트 폴더 구조 (Directory Structure)

```
project/
├── conf/                              # [설정] Hydra configuration 파일들
│   ├── data/                          # 데이터 경로 설정 (학습용 / 평가용 분리)
│   │   ├── evaluate.yaml              # evaluate.py가 참조
│   │   └── train.yaml                 # train.py가 참조
│   │
│   ├── model/                         # 모델 파라미터 설정 (19종)
│   │   ├── bm25.yaml                  # BM25 (k1, b 등)
│   │   ├── g2p_bm25_ngram.yaml        # G2P 음소 N-gram BM25
│   │   ├── dpr.yaml                   # DPR (Contriever / KLUE-RoBERTa)
│   │   ├── our_retriever*.yaml        # XPhoneBERT Bi-Encoder 변형 (6종)
│   │   ├── bm25_dense_hybrid*.yaml    # BM25+Dense 하이브리드 (3종)
│   │   ├── g2p_bm25_dense_hybrid*.yaml # G2P 하이브리드 (3종)
│   │   ├── *_with_reranker.yaml       # Reranker 파이프라인 (3종)
│   │   ├── reranker_pipeline.yaml     # 범용 Retriever+Reranker
│   │   └── semantic_pipeline.yaml     # E5 + BGE Reranker
│   │
│   ├── eval_config.yaml               # [Main] 평가 실행 설정 (evaluate.py용)
│   ├── eval_distractor_config.yaml    # Distractor 평가 설정
│   ├── eval_distractor_g2p_bm25_config.yaml
│   ├── train_config.yaml              # [Main] 학습 실행 설정 (train.py용)
│   └── train_reranker_config.yaml     # Reranker 학습 설정
│
├── src/                               # [소스] 핵심 모듈
│   ├── models/                        # 모델 구현체 모음
│   │   ├── base.py                    # 모델 부모 클래스
│   │   ├── bm25.py                    # BM25 구현
│   │   ├── g2p_bm25_ngram.py          # G2P 음소 N-gram BM25
│   │   ├── dpr.py                     # DPR 모델
│   │   ├── our_retriever*.py          # XPhoneBERT Bi-Encoder 변형 (6종)
│   │   ├── our_reranker.py            # Cross-Encoder Reranker
│   │   ├── bm25_dense_hybrid.py       # BM25+Dense 가중합 하이브리드
│   │   ├── bm25_dense_hybrid_rank.py  # RRF 하이브리드
│   │   ├── g2p_bm25_dense_hybrid*.py  # G2P 하이브리드 (가중합/RRF/Z-score)
│   │   ├── conv_meanmax_with_reranker.py  # Dense + Reranker
│   │   ├── bm25_with_reranker.py      # BM25 + Reranker
│   │   ├── hybrid_with_reranker.py    # Hybrid + Reranker
│   │   ├── retriever_with_reranker.py # 범용 2-Stage 파이프라인
│   │   ├── semantic_pipeline.py       # E5 + BGE Reranker
│   │   └── __init__.py                # 모델 팩토리 (get_model)
│   │
│   ├── dataloader.py                  # 데이터 로딩 및 전처리
│   ├── metrics.py                     # 평가 지표 계산 (R@K, MRR 등)
│   ├── g2p.py                         # G2P 변환기
│   ├── utils.py                       # 텍스트 정규화 등
│   └── text_utils.py                  # normalize_text re-export (호환용)
│
├── scripts/
│   ├── evaluate/                      # 고급 평가 스크립트
│   │   ├── evaluate_semantic.py
│   │   ├── evaluate_ensemble.py
│   │   ├── evaluate_ensemble_minmax_search.py
│   │   ├── evaluate_distractor_scaling.py
│   │   ├── evaluate_semantic_watchdog.py
│   │   └── evaluate_init.py
│   ├── distractor/                    # Distractor 독립 실험
│   │   ├── evaluate_distractor_bm25.py
│   │   ├── evaluate_distractor_g2p_bm25.py
│   │   ├── evaluate_distractor_dpr.py
│   │   ├── evaluate_distractor_g2p_dpr.py
│   │   ├── evaluate_distractor_e5.py
│   │   ├── evaluate_distractor_e5_bge.py
│   │   ├── evaluate_distractor_bm25_ngram.py
│   │   └── monitor.py
│   ├── preprocess/                    # 데이터 전처리
│   │   ├── prepare_g2p_corpus.py
│   │   ├── prepare_query_g2p.py
│   │   ├── fix_qrels.py
│   │   ├── merge.py
│   │   └── merge_tsv.py
│   └── (기존: prepare_train.py, prepare_val.py, 등)
│
├── outputs/                           # [결과] 실행 로그 및 결과 자동 저장 (Git ignore 권장)
│   └── {model}_{mode}_{date}_{time}/
│       ├── evaluate.log               # 실행 로그
│       ├── distractor_scaling_trend.json  # Scale별 성능 추이
│       ├── prediction_results_scale_*.tsv # 상세 예측 결과
│       └── rankings_*.json            # RRF 앙상블용 순위 데이터
│
├── evaluate.py                        # [실행] 평가 스크립트 (Distractor Scaling 지원)
├── evaluate_watchdog.py               # [실행] 프로세스 프리즈 감지 및 자동 재시작
├── train.py                           # [실행] 학습 스크립트 (Train / Validation)
├── train_reranker.py                  # [실행] Reranker 학습
└── requirements.txt                   # 필요 라이브러리 목록
```
---

## 실행 방법 (Usage)
모든 설정은 conf/ 폴더 내의 YAML 파일에서 관리하며, 실행 시 Hydra를 통해 자동으로 로드됩니다.

### 2-1. 평가 (Evaluation)
저장된 모델이나 알고리즘(BM25 등)의 검색 성능을 측정합니다.


1. 전체 데이터(Merged) 평가 (기본값)
```
python evaluate.py
```

2. 특정 유저 데이터만 평가 (예: user 0)
```
python evaluate.py user_id=0
```

3. 모델 변경
```
python evaluate.py model=bm25
python evaluate.py model=our_retriever_conv_meanmax
```

4. Distractor Scaling 평가
```
python evaluate.py model=bm25 distractor_sizes=[0,1000,43234]
```

5. Pipeline 모델 평가 (retriever + reranker 체크포인트)
```
python evaluate.py model=reranker_pipeline \
  retriever_checkpoint=path/to/retriever.pt \
  reranker_checkpoint=path/to/reranker.pt
```

6. Watchdog (프로세스 프리즈 시 자동 재시작)
```
python evaluate_watchdog.py model=reranker_pipeline
```

7. 상세 결과 저장 끄기 (속도 향상 필요 시)
```
python evaluate.py save_detail=false
```

참조 설정: conf/eval_config.yaml

사용 데이터: conf/data/evaluate.yaml에 정의된 경로

### 2-2. 학습 (Training)
모델을 학습하거나 검증(Validation)을 수행합니다.

학습 실행 (자동으로 mode=train, 폴더명에 train이 붙음)
```
python train.py
python train.py model=our_retriever_conv_meanmax
```
참조 설정: conf/train_config.yaml

사용 데이터: conf/data/train.yaml에 정의된 경로

### 2-3. 독립 Distractor 실험
```
python scripts/distractor/evaluate_distractor_bm25.py
python scripts/distractor/evaluate_distractor_e5_bge.py
```

---

## 결과 확인 (Outputs)
실행이 완료되면 outputs/ 폴더 아래에 모델명_모드_날짜_시간 형식으로 폴더가 생성됩니다.

주요 파일 설명
- metrics.json / distractor_scaling_trend.json: 최종 성능 점수

  - Passage Level: 정확한 구절을 찾았는지 평가

  - Song Level: 해당 노래를 찾았는지 평가 (사용자 체감 성능)

  - 지표: R@1, R@5, R@10, R@50, R@100, MRR

- prediction_results*.tsv: 상세 예측 리포트 (Excel로 열기 추천)

  - query_text: 입력 쿼리

  - best_gold_rank: 정답이 몇 등에서 나왔는지 (못 찾으면 -1)

  - difficulty: 쿼리 난이도

  - is_song_correct: 노래 정답 여부

- rankings_*.json: RRF 앙상블용 순위/점수 데이터

---

## 사용 가능한 모델 목록

### 단일 검색기
| 모델명 | config | 설명 |
|--------|--------|------|
| `bm25` | `bm25.yaml` | BM25 Lexical 검색 |
| `g2p_bm25_ngram` | `g2p_bm25_ngram.yaml` | G2P 음소 N-gram BM25 |
| `dpr` | `dpr.yaml` | DPR (Contriever / KLUE-RoBERTa) |
| `our_retriever` | `our_retriever.yaml` | XPhoneBERT Bi-Encoder |
| `our_retriever_conv_meanmax` | `our_retriever_conv_meanmax.yaml` | Conv + MeanMax Pooling |
| `our_retriever_meanmax` | `our_retriever_meanmax.yaml` | MeanMax Pooling |

### 하이브리드 검색기
| 모델명 | config | 결합 방식 |
|--------|--------|-----------|
| `bm25_dense_hybrid` | `bm25_dense_hybrid.yaml` | Min-Max + 가중합 |
| `bm25_dense_hybrid_rank` | `bm25_dense_hybrid_rank.yaml` | RRF |
| `g2p_bm25_dense_hybrid` | `g2p_bm25_dense_hybrid.yaml` | G2P + Dense 가중합 |
| `g2p_bm25_dense_hybrid_rank` | `g2p_bm25_dense_hybrid_rank.yaml` | G2P + Dense RRF |
| `g2p_bm25_dense_hybrid_zscore` | `g2p_bm25_dense_hybrid_zscore.yaml` | G2P + Dense Z-score + CombMAX |

### 파이프라인 모델 (2-Stage)
| 모델명 | config | 구조 |
|--------|--------|------|
| `conv_meanmax_with_reranker` | `conv_meanmax_with_reranker.yaml` | Dense + Cross-Encoder |
| `bm25_with_reranker` | `bm25_with_reranker.yaml` | BM25 + Cross-Encoder |
| `hybrid_with_reranker` | `hybrid_with_reranker.yaml` | BM25+Dense Hybrid + Cross-Encoder |
| `reranker_pipeline` | `reranker_pipeline.yaml` | 범용 Retriever + Reranker |
| `semantic_pipeline` | `semantic_pipeline.yaml` | E5 + BGE Reranker |

---

## 확장 가이드 (For Developers)
Q. 새로운 모델(BERT 등)을 추가하려면?
1. src/models/bert.py 파일을 생성하고 클래스를 구현합니다.

2. src/models/__init__.py에 등록합니다.

3. conf/model/bert.yaml 설정 파일을 만듭니다.

실행: python train.py model=bert

Q. 학습 데이터 경로를 바꾸려면?
conf/data/train.yaml 파일을 수정하세요.

Q. 평가 지표를 추가하려면?
src/metrics.py 내의 함수를 수정하세요.
