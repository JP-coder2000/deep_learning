from src.config import Config
from src.model import MiniGPT
from src.trainer import Trainer
from src.evaluator import ModelEvaluator
from src.data_utils import prepare_dataloaders
import torch
import logging
import os
from pathlib import Path
from datetime import datetime
import pickle

def setup_logging():
    """Configura el sistema de logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"training_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )

def load_texts(config):
    """Carga los textos desde el archivo pickle."""
    try:
        data_path = os.path.join(config.RAW_DIR, "wiki_articles.pkl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "No se encontró el archivo de textos. "
                "Ejecuta primero preprocess.py para generar los datos."
            )
        
        with open(data_path, 'rb') as f:
            texts = pickle.load(f)
        
        logging.info(f"Textos cargados: {len(texts)} artículos")
        return texts
        
    except Exception as e:
        logging.error(f"Error cargando textos: {str(e)}")
        raise

def initialize_model(config):
    """Inicializa el modelo y opcionalmente carga checkpoints."""
    try:
        logging.info("Inicializando modelo...")
        model = MiniGPT(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        # Verificar si existe un checkpoint previo
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'last_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            logging.info("Cargando checkpoint existente...")
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Checkpoint cargado de época {checkpoint['epoch']}")
        else:
            start_epoch = 0
            logging.info("Iniciando entrenamiento desde cero")
        
        return model, start_epoch
        
    except Exception as e:
        logging.error(f"Error inicializando modelo: {str(e)}")
        raise

def main():
    # Configurar logging
    setup_logging()
    
    try:
        # Inicializar configuración
        config = Config()
        logging.info(f"Usando dispositivo: {config.DEVICE}")
        
        # Cargar textos
        logging.info("Cargando textos...")
        texts = load_texts(config)
        
        # Preparar dataloaders
        logging.info("Preparando dataloaders...")
        train_dataloader, val_dataloader, test_dataloader, dataset = prepare_dataloaders(
            texts=texts,
            config=config
        )
        
        # Inicializar modelo
        model, start_epoch = initialize_model(config)
        
        # Configurar trainer
        trainer = Trainer(model, config)
        
        # Entrenar modelo
        logging.info("Iniciando entrenamiento...")
        trainer.train(train_dataloader, val_dataloader)
        
        # Evaluar modelo
        logging.info("Realizando evaluación completa...")
        evaluator = ModelEvaluator(model, dataset, config.DEVICE)
        
        # Evaluación completa - pasando los dataloaders como argumentos posicionales
        metrics = evaluator.evaluate_model(train_dataloader, val_dataloader, test_dataloader)
        
        # Guardar reporte de evaluación
        evaluator.save_evaluation_report(metrics)
        
        # Imprimir métricas finales
        logging.info("\nMétricas finales:")
        logging.info(f"Test Perplexity: {metrics['test_perplexity']:.2f}")
        logging.info(f"BLEU-4: {metrics['bleu-4']:.4f}")
        logging.info(f"Overfitting Ratio: {metrics['overfitting_ratio']:.4f}")
        logging.info(f"Prediction Bias: {metrics['prediction_bias']:.4f}")
        
        # Generar algunas muestras de texto
        logging.info("\nGenerando muestras de texto...")
        test_prompts = [
            "En el año 1492,",
            "La inteligencia artificial",
            "Durante la Edad Media,"
        ]
        
        for prompt in test_prompts:
            generated_text = evaluator.generate_text(prompt, max_length=50)
            logging.info(f"\nPrompt: {prompt}")
            logging.info(f"Generado: {generated_text}")
        
        logging.info("¡Entrenamiento y evaluación completados exitosamente!")
        
    except Exception as e:
        logging.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()