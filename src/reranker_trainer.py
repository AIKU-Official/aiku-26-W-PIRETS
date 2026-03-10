import os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)

class RerankerTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config, device, accelerator, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.accelerator = accelerator
        self.output_dir = output_dir
        
        self.epochs = config.get('epochs', 5)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # 주기 제어 변수
        self.logging_steps = config.get('logging_steps', 200)
        self.save_steps = config.get('save_steps', 2000)
        self.eval_steps = config.get('eval_steps', 100000)
        
        self.global_step = 0
        self.best_auc = 0.0
        
        # Class Imbalance 제어용 가중치
        num_negatives = config.get('num_negatives', 4)
        pos_weight_tensor = torch.tensor([num_negatives], dtype=torch.float).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def save_model(self, filename):
        """DDP 래핑을 벗겨내고 모델 가중치를 안전하게 저장하는 헬퍼 함수"""
        if self.accelerator.is_local_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_path = os.path.join(self.output_dir, "checkpoints", filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(unwrapped_model.state_dict(), save_path)
            logger.info(f"       Model saved to {save_path}")

    def run_evaluation(self, epoch):
        """Validation을 수행하고 결과를 로깅하는 분리된 함수"""
        val_loss, acc, auc = self.evaluate(epoch)
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_local_main_process:
            logger.info(f"    Val Loss: {val_loss:.4f}")
            logger.info(f"    Metrics : Accuracy: {acc:.4f} | AUC: {auc:.4f}")
            
            if auc > self.best_auc:
                self.best_auc = auc
                self.save_model("best_model.pt")
                logger.info(f"     New Best Model! (AUC: {self.best_auc:.4f})")

    def train_epoch(self, epoch):
        self.model.train()
        
        if self.accelerator.is_local_main_process:
            logger.info(f"\n[Epoch {epoch}/{self.epochs}]")
            
        # tqdm 프로그레스 바는 step 로깅과 겹쳐 출력을 지저분하게 만들 수 있으므로 비활성화할 수 있습니다.
        # 여기서는 Step 로깅이 주 목적이므로 tqdm의 출력 형태를 간소화합니다.
        progress_bar = tqdm(self.train_loader, desc="Training", disable=not self.accelerator.is_local_main_process, leave=False)
        
        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                logits = self.model({
                    'input_ids': batch['input_ids'], 
                    'attention_mask': batch['attention_mask']
                })
                loss = self.criterion(logits, batch['labels'])
                
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            #  [핵심] Optimizer Step이 실제로 수행되었을 때만 Global Step 증가 및 로깅
            if self.accelerator.sync_gradients:
                self.global_step += 1
                
                # 1. Logging Steps
                if self.global_step % self.logging_steps == 0 and self.accelerator.is_local_main_process:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                    logger.info(f"Step {self.global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.8f}")
                
                # 2. Save Steps
                if self.global_step % self.save_steps == 0:
                    self.save_model(f"checkpoint-{self.global_step}.pt")
                
                # 3. Eval Steps
                if self.global_step % self.eval_steps == 0:
                    if self.accelerator.is_local_main_process:
                        logger.info(f"     Evaluation at Step {self.global_step}")
                    self.run_evaluation(epoch)
                    self.model.train() # 평가 후 다시 학습 모드로 복귀

        self.accelerator.wait_for_everyone()
        
        # Epoch 종료 후 로깅 및 평가
        if self.accelerator.is_local_main_process:
            logger.info(f"     End of Epoch {epoch}")
            
        self.run_evaluation(epoch)
        self.save_model(f"epoch_{epoch}.pt")

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        
        for batch in self.val_loader:
            logits = self.model({
                'input_ids': batch['input_ids'], 
                'attention_mask': batch['attention_mask']
            })
            loss = self.criterion(logits, batch['labels'])
            
            gathered_loss = self.accelerator.gather(loss)
            total_loss += gathered_loss.mean().item()
            
            probs = torch.sigmoid(logits)
            
            gathered_labels = self.accelerator.gather_for_metrics(batch['labels'])
            gathered_probs = self.accelerator.gather_for_metrics(probs)
            
            all_labels.extend(gathered_labels.cpu().numpy())
            all_probs.extend(gathered_probs.cpu().numpy())
            
        avg_loss = total_loss / len(self.val_loader)
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        auc = roc_auc_score(all_labels, all_probs)
            
        return avg_loss, acc, auc