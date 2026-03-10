import hydra
import torch
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Custom Modules
from src.dataset import RetrievalDataset
from src.collator import RetrievalCollator
from src.models import get_model
from src.trainer import RetrievalTrainer

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig):
    # 0. 설정 출력 및 시드 고정
    print("="*60)
    print(f"Experiment: {cfg.exp_name}")
    print(f"Output Dir: {cfg.train.output_dir}")
    print("="*60)
    
    # (선택사항) Reproducibility를 위한 Torch Seed 고정
    if cfg.get('seed'):
        torch.manual_seed(cfg.seed)
        
    # 1. 경로 설정 (Hydra 호환 절대 경로 변환)
    base_dir = to_absolute_path(cfg.data.base_dir)
    train_tsv = os.path.join(base_dir, cfg.data.train_file)
    val_tsv = os.path.join(base_dir, cfg.data.val_file)
    g2p_jsonl = os.path.join(base_dir, cfg.data.g2p_file) # 미리 변환된 파일
    val_g2p_jsonl = os.path.join(base_dir, cfg.data.val_g2p_file)
    
    # 2. 데이터셋 로드
    log.info("Loading Datasets...")
    
    # Train: G2P 미리 변환된 파일 사용 (속도 최적화)
    train_dataset = RetrievalDataset(
        tsv_path=train_tsv, 
        g2p_path=g2p_jsonl, 
        is_train=True
    )
    
    # Val: G2P 파일 없으면 실시간 변환 (검증 데이터는 작으니까 OK)
    # 만약 val_g2p.jsonl도 만들었다면 g2p_path에 넣어줘도 됨
    val_dataset = RetrievalDataset(
        tsv_path=val_tsv, 
        g2p_path=val_g2p_jsonl, 
        is_train=False
    )
    
    log.info(f"   - Train Size: {len(train_dataset):,}")
    log.info(f"   - Val Size:   {len(val_dataset):,}")
    
    # 3. Collator 준비 (핵심 전략)
    log.info("Initializing Collators...")
    
    # [Train] Random Slicing + Random Noise (데이터 증강)
    train_collator = RetrievalCollator(cfg, is_train=True)
    
    # [Val] Deterministic Slicing + Fixed Noise (공정한 평가)
    # is_train=False로 설정하면 collator 내부에서 시드(seed)를 고정함
    val_collator = RetrievalCollator(cfg, is_train=False)
    
    # 4. 모델 준비
    log.info(f"Loading Model: {cfg.model.name} ({cfg.model.backbone})")
    model = get_model(cfg.model)
    
    # 5. Trainer 초기화
    log.info("Initializing Trainer...")
    trainer = RetrievalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_collator=train_collator, # 랜덤 노이즈
        val_collator=val_collator,     # 고정 노이즈
        config=cfg
    )
    
    # 6. 학습 시작
    trainer.train()

if __name__ == "__main__":
    main()