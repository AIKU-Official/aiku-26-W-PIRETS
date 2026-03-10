# src/dataloader.py

import os
import csv
import logging
from collections import defaultdict

log = logging.getLogger(__name__)

class RetrievalDataLoader:
    def __init__(self, base_dir, user_id=None):
        self.base_dir = base_dir
        # user_id 처리 로직을 여기로 위임
        self.suffix = str(user_id) if user_id is not None else ""
        self.mode_desc = f"User {self.suffix}" if user_id is not None else "Merged Dataset"

    def _load_tsv(self, filename):
        """내부 유틸리티: 파일 경로 자동 조립 및 로드"""
        # suffix가 있으면 파일명 뒤에 붙임 (예: corpus0.tsv)
        # suffix가 ""이면 (예: corpus.tsv)
        if self.suffix:
            name, ext = os.path.splitext(filename)
            filename = f"{name}{self.suffix}{ext}"
            
        filepath = os.path.join(self.base_dir, filename)
        
        data = []
        if not os.path.exists(filepath):
            log.warning(f"[Warning] File not found: {filepath}")
            return []
            
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                data.append(row)
        return data

    def load_corpus(self):
        log.info(f"[{self.mode_desc}] Loading Corpus...")
        return self._load_tsv('corpus.tsv')

    def load_queries(self):
        log.info(f"[{self.mode_desc}] Loading Queries...")
        return self._load_tsv('queries.tsv')

    def load_qrels(self):
        log.info(f"[{self.mode_desc}] Loading Qrels...")
        return self._load_tsv('qrels.tsv')

    def get_mappings(self, corpus_data, qrels_data):
        """평가에 필요한 매핑 테이블(pid2sid, qrels_map) 생성"""
        # 1. PID -> SID
        pid2sid = {}
        for row in corpus_data:
            pid2sid[row['passage_id']] = row['song_id']
            
        # 2. QID -> Set of Gold PIDs
        qrels_map = defaultdict(set)
        for row in qrels_data:
            qrels_map[row['query_id']].add(row['passage_id'])
            
        return pid2sid, qrels_map