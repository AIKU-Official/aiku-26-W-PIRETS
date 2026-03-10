import os
import torch
import logging
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from src.metrics import calculate_retrieval_metrics

#  Accelerate 임포트
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed
from datetime import timedelta

logger = logging.getLogger(__name__)

class RetrievalTrainer:
    def __init__(self, model, train_dataset, val_dataset, train_collator, val_collator, config):
        self.config = config
        
        #  [추가] 노는 파라미터(pooler) 무시 설정
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        #  타임아웃 설정
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=20))
        
        # 1. Accelerator 초기화
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.train.fp16 else "no",
            gradient_accumulation_steps=1,
            log_with="all",
            project_dir=config.train.output_dir,
            #  [추가] 핸들러 등록
            kwargs_handlers=[ddp_kwargs, timeout_kwargs]
        )
        
        # 2. [중요] Device 설정 (Accelerate가 잡아준 장치 사용)
        self.device = self.accelerator.device  #  이게 있어야 evaluate 함수 에러 안 남
        
        # 메인 프로세스에서만 로그 출력
        if self.accelerator.is_main_process:
            logging.basicConfig(level=logging.INFO)
            
        self.model = model
        
        # 3. DataLoader 설정
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=True, 
            collate_fn=train_collator,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=False, 
            collate_fn=val_collator,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # 4. Optimizer & Scheduler
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.train.learning_rate, 
            weight_decay=config.train.weight_decay
        )
        
        self.total_steps = len(self.train_loader) * config.train.epochs
        self.warmup_steps = int(self.total_steps * config.train.warmup_ratio)
        
        #  [수정] Warmup 유무에 따라 스케줄러를 분기 처리합니다.
        if self.warmup_steps > 0:
            scheduler1 = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps)
            scheduler2 = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.total_steps - self.warmup_steps)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[self.warmup_steps])
        else:
            # Warmup이 없을 경우(0.0), 시작부터 바로 1.0 비율로 스케줄러를 가동합니다.
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.total_steps)
            
        # 5. Prepare (Accelerate)
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        # 6. 상태 관리
        self.global_step = 0
        self.best_val_mrr = 0.0  # MRR 기준 (0.0부터 시작)
        self.output_dir = config.train.output_dir
        
        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        if self.accelerator.is_main_process:
            logger.info(f" Start Training with Accelerate... GPUs: {self.accelerator.num_processes}")
            logger.info(f"   - Global Batch Size: {self.config.train.batch_size * self.accelerator.num_processes}")

        for epoch in range(self.config.train.epochs):
            if self.accelerator.is_main_process:
                logger.info(f"\n[Epoch {epoch+1}/{self.config.train.epochs}]")
            
            self.model.train()
            
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Ep {epoch+1} Train", 
                disable=not self.accelerator.is_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                # Forward
                outputs = self.model(batch)
                loss = outputs['loss']

                # Backward
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                self.global_step += 1
                
                # Logging (Step)
                if self.global_step % self.config.train.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if self.accelerator.is_main_process:
                        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
                        logger.info(f"Step {self.global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.8f}")
                
                # Checkpoint Save
                if self.config.train.save_steps > 0 and self.global_step % self.config.train.save_steps == 0:
                    self.save_model(f"checkpoint-{self.global_step}.pt")

                # Validation (Step 주기마다)
                if self.global_step % self.config.train.eval_steps == 0:
                    self.perform_validation(step_name=f"Step {self.global_step}")

            # ====================================================
            #  [추가됨] Epoch 종료 시 무조건 Validation 수행
            # ====================================================
            # 검증 시작하기 전에 "다 모여!" 외치기
            self.accelerator.wait_for_everyone()  #  추가

            if self.accelerator.is_main_process:
                print(" Starting Validation...")
            
            self.perform_validation(step_name=f"End of Epoch {epoch+1}")
            
            # Epoch Checkpoint Save
            self.save_model(f"epoch_{epoch+1}.pt")
            
        if self.accelerator.is_main_process:
            logger.info(" Training Finished!")

    #  검증 로직이 중복되므로 별도 메서드로 분리했습니다.
    def perform_validation(self, step_name):
        # 1. 평가 수행
        eval_results = self.evaluate()
        val_loss = eval_results['loss']
        val_metrics = eval_results['metrics']
        
        #  [추가] 로그 찍기 전에 줄 맞추기 (선택사항, 에러 방지용으로 좋음)
        self.accelerator.wait_for_everyone()
        
        # 2. 로깅
        if self.accelerator.is_main_process:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"    {step_name}")
            logger.info(f"      Val Loss: {val_loss:.4f}")
            logger.info(f"      Metrics : {metric_str}")
            
            # 3. Best Model 저장 (MRR 기준)
            current_mrr = val_metrics['MRR']
            if current_mrr > self.best_val_mrr:
                self.best_val_mrr = current_mrr
                self.save_model("best_model.pt")
                logger.info(f"       New Best Model! (MRR: {current_mrr:.4f})")
        
        self.model.train()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        q_embs = []
        p_embs = []
        
        #  [수정 1] DDP 껍데기 벗기기 (알맹이 모델 접근)
        # self.model.module이 있으면 그걸 쓰고, 없으면(단일 GPU) 그냥 self.model 씀
        model_to_encode = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 메인 프로세스 로컬 데이터만 인코딩 (간소화 버전)
        for batch in tqdm(self.val_loader, desc="Encoding Val Set", leave=False, disable=not self.accelerator.is_main_process):
            q_inputs = {'input_ids': batch['q_input_ids'], 'attention_mask': batch['q_attention_mask']}
            p_inputs = {'input_ids': batch['p_input_ids'], 'attention_mask': batch['p_attention_mask']}
            
            #  [수정 2] self.model.encode -> model_to_encode.encode 로 변경
            q_emb = model_to_encode.encode(q_inputs['input_ids'], q_inputs['attention_mask'])
            p_emb = model_to_encode.encode(p_inputs['input_ids'], p_inputs['attention_mask'])
            
            q_embs.append(q_emb.cpu())
            p_embs.append(p_emb.cpu())
            
        q_embs = torch.cat(q_embs, dim=0)
        p_embs = torch.cat(p_embs, dim=0)
        
        # Loss & Metric 계산
        n_samples = q_embs.size(0)
        batch_size = 1000 
        
        total_loss = 0
        total_metrics = {'R@1': 0, 'R@5': 0, 'R@10': 0, 'R@50': 0, 'R@100': 0, 'MRR': 0}
        
        p_embs_gpu = p_embs.to(self.device) #  self.device 사용 (init에서 정의됨)
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            q_batch = q_embs[i:end].to(self.device)
            current_batch_size = end - i
            
            scores = torch.matmul(q_batch, p_embs_gpu.T) / model_to_encode.temperature
            labels = torch.arange(i, end, device=self.device)
            
            loss = F.cross_entropy(scores, labels)
            total_loss += loss.item() * current_batch_size
            
            batch_metrics = calculate_retrieval_metrics(scores, labels)
            for k, v in batch_metrics.items():
                total_metrics[k] += v * current_batch_size
                
        avg_loss = total_loss / n_samples
        avg_metrics = {k: v / n_samples for k, v in total_metrics.items()}
        
        return {'loss': avg_loss, 'metrics': avg_metrics}
    
    def save_model(self, filename):
        #  [삭제] 여기가 문제였습니다. (Rank 0만 호출하는데 여기서 모두를 기다리라고 해서)
        # self.accelerator.wait_for_everyone()
        
        # 메인 프로세스만 저장 수행
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.output_dir, filename)
            
            # DDP 껍데기 벗기기
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            # 저장
            torch.save(unwrapped_model.state_dict(), save_path)
            logger.info(f"       Model saved to {save_path}")