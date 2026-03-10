import os
import json
import logging
import gc
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RetrievalDataset(Dataset):
    def __init__(self, tsv_path: str, g2p_path: str = None, is_train: bool = True):
        self.tsv_path = tsv_path
        self.g2p_path = g2p_path
        self.is_train = is_train
        
        # ---------------------------------------------------------
        # 1. TSV 로드 (메모리 최적화)
        # ---------------------------------------------------------
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f" TSV file not found: {tsv_path}")
            
        logger.info(f" Loading TSV dataset from {tsv_path}...")
        
        # Pandas로 빠르게 읽고 -> 리스트로 변환 -> Pandas 삭제
        try:
            df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip', dtype=str)
            df = df.dropna(subset=['id', 'text'])
            
            # DataFrame을 통째로 들고 있으면 무거우니까, 가벼운 List[Tuple]로 변환
            # self.data = [('train_0', 'text...'), ('train_1', 'text...')]
            self.data = list(df[['id', 'text']].itertuples(index=False, name=None))
            
            #  [메모리 절약 핵심] 다 쓴 무거운 Pandas 객체 삭제
            del df
            gc.collect() 
            
        except Exception as e:
            logger.error(f" Failed to load TSV: {e}")
            raise e
            
        logger.info(f"    Loaded {len(self.data):,} raw text samples (converted to list).")

        # ---------------------------------------------------------
        # 2. G2P JSONL 로드 (Dict 최적화)
        # ---------------------------------------------------------
        self.passage_phonemes = {}
        
        if g2p_path and os.path.exists(g2p_path):
            logger.info(f" Loading Pre-computed G2P data from {g2p_path}...")
            
            try:
                # 파일 전체를 read() 하지 않고 한 줄씩 읽어서 메모리 절약
                with open(g2p_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Loading G2P JSONL"):
                        if not line.strip(): continue
                        
                        # 필요한 부분만 딱 잘라서 저장
                        item = json.loads(line)
                        p_id = item.get('id')
                        p_ph = item.get('phoneme')
                        
                        if p_id and p_ph:
                            self.passage_phonemes[p_id] = p_ph
                            
                logger.info(f"    Loaded {len(self.passage_phonemes):,} phoneme entries.")
                
            except Exception as e:
                logger.warning(f" Failed to load G2P file fully: {e}")
        else:
            if is_train and g2p_path:
                raise FileNotFoundError(f" Critical: G2P file not found at {g2p_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data는 이제 리스트 안의 튜플 (id, text) 형태
        sample_id, raw_text = self.data[idx]
        
        # 딕셔너리 조회 (O(1))
        passage_phoneme = self.passage_phonemes.get(sample_id, None)

        return {
            'id': str(sample_id),
            'raw_text': str(raw_text),
            'passage_phoneme': passage_phoneme 
        }