# data_utils.py
import torch
from torch.utils.data import Dataset
import requests
from tqdm import tqdm
import pickle
import os
import re
from typing import List, Tuple
import time
import logging

class WikiDataset(Dataset):
    def __init__(self, texts: List[str], vocab_size: int = 5000, max_length: int = 128):
        self.texts = texts
        self.max_length = max_length
        
        # Cargar o construir vocabulario
        cache_dir = "data/processed"
        os.makedirs(cache_dir, exist_ok=True)
        vocab_file = os.path.join(cache_dir, "vocab.pkl")
        
        if os.path.exists(vocab_file):
            print("Cargando vocabulario existente...")
            with open(vocab_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.vocab = saved_data['vocab']
                self.reverse_vocab = saved_data['reverse_vocab']
            
            # Verificar que el vocabulario tiene todos los tokens especiales
            special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
            if not all(token in self.vocab for token in special_tokens):
                print("Vocabulario existente no tiene todos los tokens especiales. Reconstruyendo...")
                self._build_vocab(vocab_size)
                with open(vocab_file, 'wb') as f:
                    pickle.dump({
                        'vocab': self.vocab,
                        'reverse_vocab': self.reverse_vocab
                    }, f)
        else:
            print("Construyendo nuevo vocabulario...")
            self._build_vocab(vocab_size)
            with open(vocab_file, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'reverse_vocab': self.reverse_vocab
                }, f)

    def _build_vocab(self, vocab_size: int):
        """Construye el vocabulario."""
        # Primero, agregar tokens especiales
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
        }
        
        # Contar frecuencias de palabras
        word_freq = {}
        for text in tqdm(self.texts, desc="Contando frecuencias"):
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Seleccionar palabras más frecuentes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Agregar palabras al vocabulario (excluyendo el espacio para tokens especiales)
        for word, _ in sorted_words[:vocab_size - len(self.vocab)]:
            self.vocab[word] = len(self.vocab)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Convierte texto a secuencia de tokens."""
        tokens = [self.vocab['[BOS]']]
        tokens.extend([
            self.vocab.get(word, self.vocab['[UNK]']) 
            for word in text.split()
        ])
        tokens.append(self.vocab['[EOS]'])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convierte secuencia de tokens a texto."""
        special_tokens = {
            self.vocab['[PAD]'], 
            self.vocab['[BOS]'], 
            self.vocab['[EOS]'], 
            self.vocab['[UNK]']
        }
        words = [
            self.reverse_vocab[token]
            for token in tokens
            if token not in special_tokens
        ]
        return ' '.join(words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.encode(text)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.vocab['[PAD]']] * (self.max_length - len(tokens))
            
        input_ids = torch.tensor(tokens[:-1])
        labels = torch.tensor(tokens[1:])
        
        return {'input_ids': input_ids, 'labels': labels}

def download_wiki_articles(num_articles: int = 100, min_length: int = 500) -> List[str]:
    """Versión simplificada y más rápida de descarga de Wikipedia."""
    cache_file = os.path.join("data/raw", "wiki_articles.pkl")
    os.makedirs("data/raw", exist_ok=True)

    # Intentar cargar desde caché
    if os.path.exists(cache_file):
        print("Cargando artículos desde caché...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Descargando {num_articles} artículos...")
    session = requests.Session()
    base_url = "https://es.wikipedia.org/w/api.php"
    
    articles = []
    retries = 3
    batch_size = 10  # Número de artículos a solicitar por vez

    with tqdm(total=num_articles) as pbar:
        while len(articles) < num_articles:
            try:
                # 1. Obtener títulos aleatorios
                params = {
                    "action": "query",
                    "format": "json",
                    "generator": "random",
                    "grnnamespace": "0",
                    "grnlimit": batch_size,
                    "prop": "extracts",
                    "explaintext": "",
                    "exintro": "1"  # Solo obtener la introducción del artículo
                }

                response = session.get(base_url, params=params, timeout=10)
                data = response.json()

                if 'query' not in data or 'pages' not in data['query']:
                    continue

                # 2. Procesar artículos obtenidos
                for page in data['query']['pages'].values():
                    if 'extract' in page and len(page['extract']) >= min_length:
                        # Limpiar texto
                        text = page['extract']
                        text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
                        text = text.strip()
                        
                        articles.append(text)
                        pbar.update(1)
                        
                        if len(articles) >= num_articles:
                            break

                time.sleep(0.1)  # Pequeña pausa para no sobrecargar la API

            except Exception as e:
                print(f"Error: {str(e)}")
                retries -= 1
                if retries <= 0:
                    break
                time.sleep(1)

    # Guardar en caché
    with open(cache_file, 'wb') as f:
        pickle.dump(articles[:num_articles], f)

    return articles[:num_articles]

def prepare_dataloaders(texts: List[str], config) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Prepara los dataloaders para entrenamiento."""
    # Split datos
    train_size = int(0.7 * len(texts))
    val_size = int(0.15 * len(texts))
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size+val_size]
    test_texts = texts[train_size+val_size:]
    
    # Crear datasets
    train_dataset = WikiDataset(
        train_texts, 
        vocab_size=config.VOCAB_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    
    val_dataset = WikiDataset(
        val_texts,
        vocab_size=config.VOCAB_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    val_dataset.vocab = train_dataset.vocab
    val_dataset.reverse_vocab = train_dataset.reverse_vocab
    
    test_dataset = WikiDataset(
        test_texts,
        vocab_size=config.VOCAB_SIZE,
        max_length=config.MAX_SEQ_LENGTH
    )
    test_dataset.vocab = train_dataset.vocab
    test_dataset.reverse_vocab = train_dataset.reverse_vocab
    
    # Crear dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_dataloader, val_dataloader, test_dataloader, train_dataset