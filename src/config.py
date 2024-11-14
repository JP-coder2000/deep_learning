# config.py
import torch
from pathlib import Path

class Config:
    def __init__(self):
        # Parámetros existentes
        self.VOCAB_SIZE = 15000
        self.EMBEDDING_DIM = 512
        self.NUM_HEADS = 4
        self.NUM_LAYERS = 2
        self.MAX_SEQ_LENGTH = 128
        self.DROPOUT = 0.1
        self.BATCH_SIZE = 64
        self.NUM_EPOCHS = 15
        self.NUM_WORKERS = 8
        
        # Nuevos parámetros para el trainer mejorado
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY = 0.01
        self.GRAD_CLIP = 1.0
        self.PATIENCE = 3  # Early stopping patience
        self.WARMUP_RATIO = 0.1  # 10% del total de steps para warmup
        
        # Parámetros de logging
        self.LOG_INTERVAL = 100  # Cada cuántos batches hacer logging
        self.SAVE_INTERVAL = 1000  # Cada cuántos batches guardar checkpoint
        
        # Directorios
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.CHECKPOINT_DIR = self.BASE_DIR / "checkpoints"
        self.LOG_DIR = self.BASE_DIR / "logs"
        
        # Asegurar que existan los directorios
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configuración de hardware
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = 4 if torch.cuda.is_available() else 0
        
        # Parámetros de datos
        self.NUM_ARTICLES = 5000  # Número de artículos a descargar
        self.MIN_ARTICLE_LENGTH = 500  # Longitud mínima de artículo
        
        # Parámetros de evaluación
        self.EVAL_BATCH_SIZE = 64
        self.MAX_EVAL_BATCHES = 100  # Limitar número de batches en evaluación
        self.TOP_K = 50  # Para generación de texto
        self.TEMPERATURE = 0.7  # Para generación de texto
        
        # Parámetros de optimización avanzada
        self.USE_AMP = True  # Usar Automatic Mixed Precision
        self.ACCUMULATION_STEPS = 1  # Gradient accumulation steps
        self.MAX_GRAD_NORM = 1.0  # Para gradient clipping
        self.LABEL_SMOOTHING = 0.1  # Factor de label smoothing
        
        # Parámetros de regularización
        self.DROPOUT_RATE = 0.1
        self.ATTENTION_DROPOUT = 0.1
        self.HIDDEN_DROPOUT = 0.1
        self.WEIGHT_DECAY = 0.01
        
        # Parámetros de scheduling
        self.LR_SCHEDULER_TYPE = "linear"  # Tipo de scheduler
        self.WARMUP_RATIO = 0.1  # Porcentaje de steps para warmup
        self.MIN_LR_RATIO = 0.1  # LR mínimo como ratio del inicial