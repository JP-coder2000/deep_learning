# model_evaluation.py

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
import time
import psutil
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Descargar recursos necesarios de NLTK
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class ModelEvaluator:
    def __init__(self, model: nn.Module, dataset, device: torch.device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_perplexity(self, dataloader) -> float:
        """Calcula la perplejidad del modelo en el conjunto de datos dado."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        print("Calculando perplejidad...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                total_tokens += (labels != 0).sum().item()
        
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    def calculate_bleu(self, reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
        """Calcula diferentes variantes del BLEU score."""
        print("Calculando BLEU scores...")
        
        bleu_scores = {
            'bleu-1': 0.0,
            'bleu-2': 0.0,
            'bleu-3': 0.0,
            'bleu-4': 0.0
        }
        
        smoother = SmoothingFunction().method1
        
        for ref, gen in zip(reference_texts, generated_texts):
            ref_tokens = nltk.word_tokenize(ref.lower())
            gen_tokens = nltk.word_tokenize(gen.lower())
            
            # Calcular BLEU-1 a BLEU-4
            for n in range(1, 5):
                score = sentence_bleu([ref_tokens], gen_tokens, 
                                   weights=tuple([1/n]*n + [0]*(4-n)),
                                   smoothing_function=smoother)
                bleu_scores[f'bleu-{n}'] += score
        
        # Calcular promedios
        num_samples = len(reference_texts)
        for k in bleu_scores:
            bleu_scores[k] /= num_samples
        
        return bleu_scores
    
    def calculate_rouge(self, reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
        """Calcula los ROUGE scores."""
        print("Calculando ROUGE scores...")
        
        scores = {
            'rouge1_f': 0.0,
            'rouge2_f': 0.0,
            'rougeL_f': 0.0
        }
        
        for ref, gen in zip(reference_texts, generated_texts):
            results = self.rouge_scorer.score(ref, gen)
            scores['rouge1_f'] += results['rouge1'].fmeasure
            scores['rouge2_f'] += results['rouge2'].fmeasure
            scores['rougeL_f'] += results['rougeL'].fmeasure
        
        # Calcular promedios
        num_samples = len(reference_texts)
        for k in scores:
            scores[k] /= num_samples
        
        return scores
    
    def calculate_bert_score(self, reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
        """Calcula BERTScore."""
        print("Calculando BERTScore...")
        P, R, F1 = bert_score(generated_texts, reference_texts, lang='es')
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
    
    def measure_inference_performance(self, text: str, num_trials: int = 10) -> Dict[str, float]:
        """Mide el rendimiento de inferencia del modelo."""
        # Truncar el texto a las primeras palabras para no exceder max_length
        words = text.split()[:self.model.max_seq_length//2]  # Dividimos por 2 para tener espacio para tokens especiales
        truncated_text = ' '.join(words)
        
        try:
            tokens = self.dataset.encode(truncated_text)
            if len(tokens) > self.model.max_seq_length:
                tokens = tokens[:self.model.max_seq_length]
            
            input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
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
        except Exception as e:
            print(f"Error midiendo rendimiento: {str(e)}")
            return {
                'mean_latency': 0.0,
                'std_latency': 0.0,
                'mean_memory': 0.0,
                'std_memory': 0.0
            }

    def generate_text_samples(self, num_samples: int = 10, max_length: int = 50) -> tuple:
        """Genera muestras de texto para evaluación."""
        print("Generando muestras de texto...")
        reference_texts = []
        generated_texts = []
        
        # Usar textos del dataset como referencia
        indices = np.random.choice(len(self.dataset.texts), num_samples)
        
        for idx in indices:
            text = self.dataset.texts[idx]
            # Usar las primeras palabras como semilla (truncadas)
            first_words = ' '.join(text.split()[:3])
            
            try:
                generated = self.generate_text(
                    first_words, 
                    max_length=min(max_length, self.max_length),
                    temperature=0.8,  # Temperatura un poco más alta para más creatividad
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                reference_texts.append(self.truncate_text(text))
                generated_texts.append(generated)
            except Exception as e:
                print(f"Error generando texto: {str(e)}")
                continue
        
        return reference_texts, generated_texts
    
    def generate_text(self, seed_text: str, max_length: int = 50, 
                    temperature: float = 0.7, top_k: int = 50, 
                    top_p: float = 0.9, repetition_penalty: float = 1.2) -> str:
        """
        Genera texto usando el modelo con técnicas mejoradas para evitar repeticiones.
        
        Args:
            seed_text: Texto inicial para la generación
            max_length: Longitud máxima del texto generado
            temperature: Factor de temperatura para la distribución de probabilidad
            top_k: Número de tokens más probables a considerar
            top_p: Probabilidad acumulada máxima para nucleus sampling
            repetition_penalty: Penalización para tokens repetidos
        """
        self.model.eval()
        
        # Truncar y codificar el texto semilla
        truncated_seed = self.truncate_text(seed_text)
        input_ids = self.safe_encode(truncated_seed).to(self.device)
        
        # Mantener registro de tokens generados para penalización
        generated_tokens = input_ids[0].tolist()
        
        with torch.no_grad():
            for _ in range(min(max_length, self.model.max_seq_length - len(input_ids[0]))):
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Aplicar repetition penalty
                for token in set(generated_tokens):
                    next_token_logits[0, token] /= repetition_penalty
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Eliminar tokens con probabilidad acumulada > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Evitar tokens especiales excepto EOS
                for token in [self.dataset.vocab['[PAD]'], self.dataset.vocab['[BOS]'], self.dataset.vocab['[UNK]']]:
                    next_token_logits[0, token] = float('-inf')
                
                # Sample from filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.dataset.vocab['[EOS]']:
                    break
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Early stopping si se detecta mucha repetición
                if len(generated_tokens) > 4:
                    last_4_tokens = generated_tokens[-4:]
                    if len(set(last_4_tokens)) == 1:  # Si los últimos 4 tokens son iguales
                        break
                
                if len(generated_tokens) >= self.max_length:
                    break
        
        return self.dataset.decode(input_ids[0].tolist())
    def evaluate_model(self, dataloader, num_samples: int = 10) -> Dict[str, float]:
        """Realiza una evaluación completa del modelo."""
        print("\nIniciando evaluación completa del modelo...")
        
        # 1. Generar muestras de texto
        reference_texts, generated_texts = self.generate_text_samples(num_samples)
        
        # 2. Calcular todas las métricas
        metrics = {}
        
        # Perplexity
        metrics['perplexity'] = self.calculate_perplexity(dataloader)
        
        # BLEU scores
        bleu_scores = self.calculate_bleu(reference_texts, generated_texts)
        metrics.update(bleu_scores)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(reference_texts, generated_texts)
        metrics.update(rouge_scores)
        
        # BERTScore
        bert_scores = self.calculate_bert_score(reference_texts, generated_texts)
        metrics.update(bert_scores)
        
        # Rendimiento de inferencia
        performance = self.measure_inference_performance(reference_texts[0])
        metrics.update(performance)
        
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
        
        # 1. BLEU scores
        plt.subplot(2, 2, 1)
        bleu_metrics = {k: v for k, v in metrics.items() if k.startswith('bleu')}
        sns.barplot(x=list(bleu_metrics.keys()), y=list(bleu_metrics.values()))
        plt.title('BLEU Scores')
        plt.xticks(rotation=45)
        
        # 2. ROUGE scores
        plt.subplot(2, 2, 2)
        rouge_metrics = {k: v for k, v in metrics.items() if k.startswith('rouge')}
        sns.barplot(x=list(rouge_metrics.keys()), y=list(rouge_metrics.values()))
        plt.title('ROUGE Scores')
        plt.xticks(rotation=45)
        
        # 3. BERTScore
        plt.subplot(2, 2, 3)
        bert_metrics = {k: v for k, v in metrics.items() if k.startswith('bert')}
        sns.barplot(x=list(bert_metrics.keys()), y=list(bert_metrics.values()))
        plt.title('BERTScore')
        plt.xticks(rotation=45)
        
        # 4. Performance metrics
        plt.subplot(2, 2, 4)
        perf_metrics = {
            'Latency (s)': metrics['mean_latency'],
            'Memory (MB)': metrics['mean_memory'],
            'Perplexity': metrics['perplexity']
        }
        sns.barplot(x=list(perf_metrics.keys()), y=list(perf_metrics.values()))
        plt.title('Performance Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_visualization_{timestamp}.png'))
        plt.close()
        
        # Crear reporte de texto
        with open(os.path.join(output_dir, f'report_{timestamp}.txt'), 'w') as f:
            f.write("=== Reporte de Evaluación del Modelo ===\n\n")
            
            f.write("1. Métricas de Calidad:\n")
            f.write(f"   Perplexity: {metrics['perplexity']:.2f}\n")
            
            f.write("\n2. BLEU Scores:\n")
            for k, v in bleu_metrics.items():
                f.write(f"   {k}: {v:.4f}\n")
            
            f.write("\n3. ROUGE Scores:\n")
            for k, v in rouge_metrics.items():
                f.write(f"   {k}: {v:.4f}\n")
            
            f.write("\n4. BERTScore:\n")
            for k, v in bert_metrics.items():
                f.write(f"   {k}: {v:.4f}\n")
            
            f.write("\n5. Métricas de Rendimiento:\n")
            f.write(f"   Latencia Media: {metrics['mean_latency']*1000:.2f}ms\n")
            f.write(f"   Desviación Estándar Latencia: {metrics['std_latency']*1000:.2f}ms\n")
            f.write(f"   Uso de Memoria Medio: {metrics['mean_memory']:.2f}MB\n")
            f.write(f"   Desviación Estándar Memoria: {metrics['std_memory']:.2f}MB\n")

def evaluate_model(self, train_dataloader, val_dataloader, test_dataloader) -> Dict[str, float]:
    """Realiza una evaluación completa del modelo."""
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
        try:
            print("\nGenerando y evaluando texto...")
            reference_texts, generated_texts = self.generate_text_samples(num_samples=50)
            
            if reference_texts and generated_texts:
                # BLEU scores
                bleu_scores = self.calculate_bleu(reference_texts, generated_texts)
                metrics.update(bleu_scores)
            else:
                print("No se pudieron generar muestras de texto suficientes para calcular BLEU scores")
        except Exception as e:
            print(f"Error en generación de texto: {str(e)}")
            metrics.update({
                'bleu-1': 0.0,
                'bleu-2': 0.0,
                'bleu-3': 0.0,
                'bleu-4': 0.0
            })
        
        # 5. Rendimiento de inferencia
        print("\nMidiendo rendimiento de inferencia...")
        try:
            if hasattr(self.dataset, 'texts') and self.dataset.texts:
            # Usar un texto corto para medir el rendimiento
                sample_text = ' '.join(self.dataset.texts[0].split()[:50])  # Tomar solo las primeras 50 palabras
                performance_metrics = self.measure_inference_performance(sample_text)
                metrics.update(performance_metrics)
        except Exception as e:
            print(f"Error midiendo rendimiento: {str(e)}")
            metrics.update({
                'mean_latency': 0.0,
                'std_latency': 0.0,
                'mean_memory': 0.0,
                'std_memory': 0.0
            })
        
    except Exception as e:
        print(f"Error durante la evaluación: {str(e)}")
        raise
        
    
    return metrics