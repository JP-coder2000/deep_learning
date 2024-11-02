from src.config import Config
from src.data_utils import download_wiki_articles, prepare_dataloaders
import logging
import os
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories():
    """Crea los directorios necesarios."""
    dirs = ['data', 'data/raw', 'data/processed', 'logs', 'checkpoints']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def main():
    # Crear directorios necesarios
    setup_directories()
    
    # Inicializar configuración
    config = Config()
    
    # Descargar datos
    logging.info("Descargando datos...")
    texts = download_wiki_articles(
        num_articles=config.NUM_ARTICLES,
        min_length=config.MIN_ARTICLE_LENGTH
    )
    
    # Preparar datos
    logging.info("Preparando datos...")
    train_dataloader, val_dataloader, test_dataloader, dataset = prepare_dataloaders(
        texts, config
    )
    
    logging.info("Preprocesamiento completado!")

if __name__ == "__main__":
    main()