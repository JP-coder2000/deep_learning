import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
import time
import psutil
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

try:
    nltk.download('punkt', quiet=True)
except:
    pass

class ModelEvaluator:
    def __init__(self, model: nn.Module, dataset, device: torch.device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.max_length = self.model.max_seq_length - 2  # -2 para [BOS] y [EOS]

    def truncate_text(self, text: str) -> str:
        """Trunca el texto para no exceder max_length."""
        words = text.split()[:self.max_length]
        return ' '.join(words)
    
    def safe_encode(self, text: str) -> torch.Tensor:
        """Codifica el texto de manera segura sin exceder max_length."""
        tokens = self.dataset.encode(self.truncate_text(text))
        if len(tokens) > self.model.max_seq_length:
            tokens = tokens[:self.model.max_seq_length]
        return torch.tensor(tokens).unsqueeze(0)

    def calculate_loss(self, dataloader) -> float:
        """Calcula la pérdida promedio en un conjunto de datos."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if input_ids.size(1) > self.model.max_seq_length:
                    input_ids = input_ids[:, :self.model.max_seq_length]
                    labels = labels[:, :self.model.max_seq_length]
                
                outputs = self.model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

    def calculate_perplexity(self, dataloader, desc="Calculando perplejidad...") -> float:
        """Calcula la perplejidad del modelo."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if input_ids.size(1) > self.model.max_seq_length:
                    input_ids = input_ids[:, :self.model.max_seq_length]
                    labels = labels[:, :self.model.max_seq_length]
                
                outputs = self.model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()
                total_tokens += (labels != 0).sum().item()
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()

    def calculate_overfitting_metrics(self, train_dataloader, val_dataloader) -> Dict[str, float]:
        """Calcula métricas relacionadas con overfitting."""
        train_loss = self.calculate_loss(train_dataloader)
        val_loss = self.calculate_loss(val_dataloader)
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'overfitting_ratio': val_loss / train_loss if train_loss > 0 else float('inf'),
            'generalization_gap': val_loss - train_loss
        }
        
        return metrics

    def calculate_model_bias(self, test_dataloader) -> Dict[str, float]:
        """Calcula métricas de sesgo del modelo."""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Calculando sesgo"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if input_ids.size(1) > self.model.max_seq_length:
                    input_ids = input_ids[:, :self.model.max_seq_length]
                    labels = labels[:, :self.model.max_seq_length]
                
                outputs = self.model(input_ids)
                preds = torch.argmax(outputs, dim=-1)
                
                predictions.extend(preds.cpu().numpy().flatten())
                actuals.extend(labels.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            'prediction_bias': np.mean(predictions - actuals),
            'absolute_bias': np.mean(np.abs(predictions - actuals)),
            'bias_variance': np.var(predictions - actuals),
            'prediction_variance': np.var(predictions)
        }

    def calculate_bleu(self, reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
        """Calcula diferentes variantes del BLEU score."""
        bleu_scores = {f'bleu-{i}': 0.0 for i in range(1, 5)}
        smoother = SmoothingFunction().method1
        
        for ref, gen in zip(reference_texts, generated_texts):
            ref_tokens = nltk.word_tokenize(ref.lower())
            gen_tokens = nltk.word_tokenize(gen.lower())
            
            for n in range(1, 5):
                score = sentence_bleu([ref_tokens], gen_tokens, 
                                   weights=tuple([1/n]*n + [0]*(4-n)),
                                   smoothing_function=smoother)
                bleu_scores[f'bleu-{n}'] += score
        
        num_samples = len(reference_texts)
        if num_samples > 0:
            for k in bleu_scores:
                bleu_scores[k] /= num_samples
        
        return bleu_scores

    def generate_text(self, seed_text: str, max_length: int = 50, 
                     temperature: float = 0.9, repetition_penalty: float = 1.5) -> str:
        """Genera texto usando el modelo."""
        self.model.eval()
        truncated_seed = self.truncate_text(seed_text)
        input_ids = self.safe_encode(truncated_seed).to(self.device)
        generated_tokens = input_ids[0].tolist()
        banned_tokens = set()
        last_tokens = []
        
        with torch.no_grad():
            for _ in range(min(max_length, self.model.max_seq_length - len(input_ids[0]))):
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Penalizar tokens recientes
                for token in set(generated_tokens[-5:]):
                    next_token_logits[0, token] /= repetition_penalty
                
                # Penalizar tokens especiales y baneados
                for token in [self.dataset.vocab['[PAD]'], self.dataset.vocab['[BOS]'], 
                            self.dataset.vocab['[UNK]']] + list(banned_tokens):
                    next_token_logits[0, token] = float('-inf')
                
                # Top-k filtering
                top_k = 10
                top_probs, top_tokens = torch.topk(next_token_logits, top_k)
                
                # Filtrar tokens repetitivos
                valid_tokens = []
                for token in top_tokens[0]:
                    token = token.item()
                    token_str = self.dataset.reverse_vocab.get(token, '')
                    
                    if (len(last_tokens) >= 2 and token in last_tokens[-2:]) or \
                       (token_str in ['de', 'del', 'la', 'el', 'los', 'las', 'un', 'una'] and 
                        len(last_tokens) >= 1 and last_tokens[-1] == token):
                        continue
                    
                    valid_tokens.append(token)
                
                if not valid_tokens:
                    next_token = torch.tensor([[self.dataset.vocab['[EOS]']]])
                else:
                    next_token = torch.tensor([[np.random.choice(valid_tokens)]])
                
                if next_token.item() == self.dataset.vocab['[EOS]']:
                    break
                
                last_tokens.append(next_token.item())
                if len(last_tokens) > 5:
                    last_tokens.pop(0)
                
                token_counts = {}
                for token in last_tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
                    if token_counts[token] > 2:
                        banned_tokens.add(token)
                
                input_ids = torch.cat([input_ids, next_token.to(self.device)], dim=1)
                generated_tokens.append(next_token.item())
                
                if len(generated_tokens) > 5:
                    last_5_tokens = generated_tokens[-5:]
                    if len(set(last_5_tokens)) <= 2:
                        break
        
        decoded_text = self.dataset.decode(input_ids[0].tolist())
        words = decoded_text.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)

    def generate_text_samples(self, num_samples: int = 10, max_length: int = 50) -> tuple:
        """Genera muestras de texto para evaluación."""
        print("Generando muestras de texto...")
        reference_texts = []
        generated_texts = []
        
        indices = np.random.choice(len(self.dataset.texts), num_samples)
        
        for idx in indices:
            text = self.dataset.texts[idx]
            first_words = ' '.join(text.split()[:3])
            
            try:
                generated = self.generate_text(
                    first_words, 
                    max_length=min(max_length, self.max_length),
                    temperature=0.9,
                    repetition_penalty=1.5
                )
                reference_texts.append(self.truncate_text(text))
                generated_texts.append(generated)
            except Exception as e:
                print(f"Error generando texto: {str(e)}")
                continue
        
        return reference_texts, generated_texts

    def measure_inference_performance(self, text: str, num_trials: int = 10) -> Dict[str, float]:
        """Mide el rendimiento de inferencia del modelo."""
        input_ids = self.safe_encode(text).to(self.device)
        
        latencies = []
        memory_usage = []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_trials):
                memory_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                _ = self.model(input_ids)
                end_time = time.time()
                
                memory_end = psutil.Process().memory_info().rss / 1024 / 1024
                
                latencies.append(end_time - start_time)
                memory_usage.append(memory_end - memory_start)
        
        return {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'mean_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage)
        }

    def evaluate_model(self, train_dataloader, val_dataloader, test_dataloader) -> Dict[str, float]:
        """Realiza una evaluación completa del modelo."""
        print("\nIniciando evaluación completa del modelo...")
        metrics = {}
        
        try:
            # 1. Métricas básicas
            print("\nCalculando métricas básicas...")
            metrics['train_perplexity'] = self.calculate_perplexity(train_dataloader)
            metrics['val_perplexity'] = self.calculate_perplexity(val_dataloader)
            metrics['test_perplexity'] = self.calculate_perplexity(test_dataloader)
            
            # 2. Métricas de overfitting
            print("\nCalculando métricas de overfitting...")
            overfitting_metrics = self.calculate_overfitting_metrics(train_dataloader, val_dataloader)
            metrics.update(overfitting_metrics)
            
            # 3. Métricas de sesgo
            print("\nCalculando métricas de sesgo...")
            bias_metrics = self.calculate_model_bias(test_dataloader)
            metrics.update(bias_metrics)
            
            # 4. Generar y evaluar texto
            print("\nGenerando y evaluando texto...")
            reference_texts, generated_texts = self.generate_text_samples(
                num_samples=min(50, len(self.dataset.texts)),
                max_length=self.max_length
            )
            
            if reference_texts and generated_texts:
                bleu_scores = self.calculate_bleu(reference_texts, generated_texts)
                metrics.update(bleu_scores)
            
            # 5. Rendimiento de inferencia
            print("\nMidiendo rendimiento de inferencia...")
            sample_text = self.truncate_text(self.dataset.texts[0])
            performance_metrics = self.measure_inference_performance(sample_text)
            metrics.update(performance_metrics)
            
        except Exception as e:
            print(f"Error durante la evaluación: {str(e)}")
            raise
        
        return metrics

    def save_evaluation_report(self, metrics: Dict[str, float], output_dir: str = 'evaluation_results'):
        """Guarda un reporte detallado de la evaluación."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar métricas en CSV
        df = pd.DataFrame([metrics])
        df.to_csv(os.path.join(output_dir, f'metrics_{timestamp}.csv'))
        
        # Crear visualizaciones
        plt.figure(figsize=(15, 10))
        
        # 1. Perplexity comparison
        plt.subplot(2, 2, 1)
        perplexity_data = {
            'Train': metrics['train_perplexity'],
            'Validation': metrics['val_perplexity'],
            'Test': metrics['test_perplexity']
        }
        sns.barplot(x=list(perplexity_data.keys()), y=list(perplexity_data.values()))
        plt.title('Perplexity por Split')
        plt.xticks(rotation=45)
        
        # 2. BLEU scores
        plt.subplot(2, 2, 2)
        bleu_metrics = {k: v for k, v in metrics.items() if k.startswith('bleu')}
        sns.barplot(x=list(bleu_metrics.keys()), y=list(bleu_metrics.values()))
        plt.title('BLEU Scores')
        plt.xticks(rotation=45)
        
        # 3. Bias metrics
        plt.subplot(2, 2, 3)
        bias_metrics = {
            'Prediction Bias': metrics['prediction_bias'],
            'Absolute Bias': metrics['absolute_bias'],
            'Bias Variance': metrics['bias_variance']
        }
        sns.barplot(x=list(bias_metrics.keys()), y=list(bias_metrics.values()))
        plt.title('Métricas de Sesgo')
        plt.xticks(rotation=45)
        
        # 4. Performance metrics
        plt.subplot(2, 2, 4)
        perf_metrics = {
            'Latency (s)': metrics['mean_latency'],
            'Memory (MB)': metrics['mean_memory']
        }
        sns.barplot(x=list(perf_metrics.keys()), y=list(perf_metrics.values()))
        plt.title('Métricas de Rendimiento')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_visualization_{timestamp}.png'))
        plt.close()
        
        # Crear reporte de texto
        with open(os.path.join(output_dir, f'evaluation_report_{timestamp}.txt'), 'w') as f:
            f.write("=== Reporte de Evaluación del Modelo ===\n\n")
            
            f.write("1. Métricas de Calidad:\n")
            f.write(f"   Train Perplexity: {metrics['train_perplexity']:.2f}\n")
            f.write(f"   Validation Perplexity: {metrics['val_perplexity']:.2f}\n")
            f.write(f"   Test Perplexity: {metrics['test_perplexity']:.2f}\n")
            
            f.write("\n2. Métricas de Overfitting:\n")
            f.write(f"   Train Loss: {metrics['train_loss']:.4f}\n")
            f.write(f"   Validation Loss: {metrics['val_loss']:.4f}\n")
            f.write(f"   Overfitting Ratio: {metrics['overfitting_ratio']:.4f}\n")
            f.write(f"   Generalization Gap: {metrics['generalization_gap']:.4f}\n")
            
            f.write("\n3. Métricas de Sesgo:\n")
            f.write(f"   Prediction Bias: {metrics['prediction_bias']:.4f}\n")
            f.write(f"   Absolute Bias: {metrics['absolute_bias']:.4f}\n")
            f.write(f"   Bias Variance: {metrics['bias_variance']:.4f}\n")
            f.write(f"   Prediction Variance: {metrics['prediction_variance']:.4f}\n")
            
            f.write("\n4. BLEU Scores:\n")
            for k, v in bleu_metrics.items():
                f.write(f"   {k}: {v:.4f}\n")
            
            f.write("\n5. Métricas de Rendimiento:\n")
            f.write(f"   Latencia Media: {metrics['mean_latency']*1000:.2f}ms\n")
            f.write(f"   Desviación Estándar Latencia: {metrics['std_latency']*1000:.2f}ms\n")
            f.write(f"   Uso de Memoria Medio: {metrics['mean_memory']:.2f}MB\n")
            f.write(f"   Desviación Estándar Memoria: {metrics['std_memory']:.2f}MB\n")