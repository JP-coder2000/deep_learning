import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import logging
from typing import Dict
import json
import time

class Trainer:
    def __init__(self, model: nn.Module, config: any):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # Optimizador
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Criterio de pérdida con label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignorar padding
            label_smoothing=0.1
        )
        
        # Métricas y logging
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.grad_norms = []
        self.training_time = []
        self.best_val_loss = float('inf')
        
        # Crear directorios para logs y checkpoints
        self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        
        # Archivo para guardar métricas
        self.metrics_file = os.path.join(self.log_dir, 'metrics.json')
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Guarda checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Guardar último checkpoint
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'last_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Si es el mejor modelo, guardar una copia
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"Nuevo mejor modelo guardado (val_loss: {metrics['loss']:.4f})")
    
    def calculate_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calcula múltiples métricas de evaluación."""
        # Convertir logits a predicciones
        preds = torch.argmax(logits, dim=-1)
        
        # Calcular accuracy ignorando padding
        mask = (labels != 0)  # No considerar padding
        correct = ((preds == labels) * mask).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        
        # Calcular pérdida
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Calcular perplejidad
        perplexity = torch.exp(loss).item()
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'accuracy': accuracy
        }
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Entrena una época y devuelve métricas."""
        self.model.train()
        total_metrics = {'loss': 0, 'perplexity': 0, 'accuracy': 0}
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f'Training')
        for batch in progress_bar:
            # Mover datos al dispositivo
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.GRAD_CLIP
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            
            # Calcular métricas
            metrics = self.calculate_metrics(outputs, labels)
            
            # Actualizar métricas totales
            for k, v in metrics.items():
                total_metrics[k] += v
            
            # Guardar learning rate actual
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'acc': f"{metrics['accuracy']:.2%}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Calcular promedios
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evalúa el modelo y devuelve métricas."""
        self.model.eval()
        total_metrics = {'loss': 0, 'perplexity': 0, 'accuracy': 0}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                metrics = self.calculate_metrics(outputs, labels)
                
                for k, v in metrics.items():
                    total_metrics[k] += v
        
        # Calcular promedios
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
    
    def save_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]):
        """Guarda las métricas en un archivo JSON."""
        metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        # Cargar métricas existentes
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                existing_metrics = json.load(f)
            existing_metrics.append(metrics)
            metrics = existing_metrics
        else:
            metrics = [metrics]
        
        # Guardar métricas
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_metrics(self):
        """Genera y guarda gráficas de métricas."""
        plt.figure(figsize=(15, 10))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Perplexity
        plt.subplot(2, 2, 2)
        plt.plot(self.train_perplexities, label='Train')
        plt.plot(self.val_perplexities, label='Validation')
        plt.title('Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('PPL')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(self.train_accuracies, label='Train')
        plt.plot(self.val_accuracies, label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Learning Rate
        plt.subplot(2, 2, 4)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('LR')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'))
        plt.close()
    
    def train(self, train_dataloader, val_dataloader):
        """Proceso de entrenamiento principal."""
        print(f"\nIniciando entrenamiento en {self.device}")
        total_start_time = time.time()
        
        # Inicializar scheduler
        total_steps = len(train_dataloader) * self.config.NUM_EPOCHS
        warmup_steps = int(total_steps * 0.1)  # 10% de warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\nÉpoca {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Entrenamiento
            train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_metrics['loss'])
            self.train_perplexities.append(train_metrics['perplexity'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validación
            val_metrics = self.evaluate(val_dataloader)
            self.val_losses.append(val_metrics['loss'])
            self.val_perplexities.append(val_metrics['perplexity'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Guardar métricas
            self.save_metrics(epoch, train_metrics, val_metrics)
            
            # Guardar checkpoint si es el mejor modelo
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Calcular tiempo de época
            epoch_time = time.time() - epoch_start_time
            self.training_time.append(epoch_time)
            
            # Logging
            logging.info(f"\nÉpoca {epoch+1} completada en {epoch_time:.2f}s")
            logging.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logging.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Imprimir métricas
            print("\nMétricas de entrenamiento:")
            print(f"Loss: {train_metrics['loss']:.4f}")
            print(f"Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"Accuracy: {train_metrics['accuracy']:.2%}")
            
            print("\nMétricas de validación:")
            print(f"Loss: {val_metrics['loss']:.4f}")
            print(f"Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"Accuracy: {val_metrics['accuracy']:.2%}")
            
            print(f"\nTiempo de época: {epoch_time:.2f}s")
            
            # Graficar métricas
            self.plot_metrics()
            
            # Early stopping
            if len(self.val_losses) > self.config.PATIENCE:
                if all(self.val_losses[-i-1] > self.val_losses[-i-2] 
                      for i in range(self.config.PATIENCE)):
                    print("\nEarly stopping triggered!")
                    break
        
        total_time = time.time() - total_start_time
        print(f"\nEntrenamiento completado en {total_time:.2f}s")
        
        # Guardar métricas finales
        final_metrics = {
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'average_epoch_time': np.mean(self.training_time),
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_perplexity': self.train_perplexities[-1],
            'final_val_perplexity': self.val_perplexities[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_accuracy': self.val_accuracies[-1]
        }
        
        with open(os.path.join(self.log_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=2)