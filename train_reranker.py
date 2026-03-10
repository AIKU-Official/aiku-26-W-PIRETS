import os
import json
import yaml
import torch
import logging
from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.reranker_dataset import RerankerDataset, reranker_collate_fn
from src.models import get_model 
from src.reranker_trainer import RerankerTrainer

# 로그 포맷 설정
log_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('src.trainer')
logger.setLevel(logging.INFO)

def load_config(yaml_path="conf/train_reranker_config.yaml"):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_corpus(tsv_path, jsonl_path):
    import pandas as pd
    corpus = {}
    df = pd.read_csv(tsv_path, sep='\t')
    for _, row in df.iterrows():
        corpus[str(row['id'])] = {'raw': str(row['text']), 'g2p': ''}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = str(data['id'])
            if doc_id in corpus:
                corpus[doc_id]['g2p'] = str(data['phoneme'])
    return corpus

def main():
    config = load_config("conf/train_reranker_config.yaml")
    set_seed(config['training']['seed'])
    
    accelerator = Accelerator(gradient_accumulation_steps=config['training']['gradient_accumulation_steps'])
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"our_reranker_train_{timestamp}")
    config['training']['output_dir'] = output_dir

    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Workspace Created: All logs & checkpoints will be saved to '{output_dir}'")
    
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone'])
    model = get_model(config['model'])

    train_corpus = load_corpus(config['data']['train_tsv'], config['data']['train_jsonl'])
    val_corpus = load_corpus(config['data']['val_tsv'], config['data']['val_jsonl'])
    
    with open(config['data']['train_cache'], 'r') as f:
        train_cache = json.load(f)
    with open(config['data']['val_cache'], 'r') as f:
        val_cache = json.load(f)

    train_dataset = RerankerDataset(
        corpus=train_corpus, hard_negative_cache=train_cache, is_train=True, num_negatives=config['training']['num_negatives']
    )
    val_dataset = RerankerDataset(
        corpus=val_corpus, hard_negative_cache=val_cache, is_train=False, num_negatives=config['training']['num_negatives']
    )

    collate_fn = partial(reranker_collate_fn, tokenizer=tokenizer, max_length=config['model']['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']), 
        weight_decay=float(config['training']['weight_decay'])
    )
    
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Trainer에 output_dir을 명시적으로 전달하여 내부에서 Checkpoint 폴더를 관리하도록 함
    trainer = RerankerTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, config=config['training'],
        device=accelerator.device, accelerator=accelerator, output_dir=output_dir
    )

    for epoch in range(1, config['training']['epochs'] + 1):
        # 이제 train_epoch 내부에서 로깅, Step 단위 평가, 저장이 모두 자동으로 수행됩니다.
        trainer.train_epoch(epoch)

    if accelerator.is_local_main_process:
        logger.info("Training Completed Successfully!")

if __name__ == "__main__":
    main()