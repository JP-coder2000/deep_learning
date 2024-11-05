import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import Config
from src.model import MiniGPT
from src.data_utils import WikiDataset
import pickle
import json
from pathlib import Path

def load_model_and_data():
    config = Config()
    try:
        # Cargar vocabulario
        vocab_path = config.PROCESSED_DIR / "vocab.pkl"
        if not vocab_path.exists():
            raise FileNotFoundError("No se encontró el archivo de vocabulario")
            
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        dataset = WikiDataset(['dummy text'], config.VOCAB_SIZE, config.MAX_SEQ_LENGTH)
        dataset.vocab = vocab_data['vocab']
        dataset.reverse_vocab = vocab_data['reverse_vocab']
        
        model = MiniGPT(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        # Cargar métricas
        metrics_path = Path("logs/metrics.json")
        metrics = None
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return model, dataset, config, metrics
        
    except Exception as e:
        st.error(f"Error al cargar el modelo y datos: {str(e)}")
        st.stop()

def generate_text(model, dataset, config, seed_text, max_length=50, 
                 temperature=0.7, top_k=50, repetition_penalty=1.2):
    try:
        tokens = dataset.encode(seed_text)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(config.DEVICE)
        generated_tokens = []
        banned_tokens = set()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Aplicar repetition penalty
                for token in set(input_ids[0].tolist() + generated_tokens):
                    next_token_logits[0, token] /= repetition_penalty
                
                # Bannear tokens especiales y repetitivos
                for token in [dataset.vocab['[PAD]'], dataset.vocab['[BOS]']] + list(banned_tokens):
                    next_token_logits[0, token] = float('-inf')
                
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == dataset.vocab['[EOS]']:
                    break
                
                # Detectar repeticiones
                generated_tokens.append(next_token.item())
                if len(generated_tokens) > 3:
                    if len(set(generated_tokens[-3:])) == 1:  # Si los últimos 3 tokens son iguales
                        banned_tokens.add(generated_tokens[-1])
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return dataset.decode(input_ids[0].tolist())
    except Exception as e:
        st.error(f"Error generando texto: {str(e)}")
        return ""

def plot_metrics(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(len(metrics))
    train_metrics = [m['train']['loss'] for m in metrics]
    val_metrics = [m['validation']['loss'] for m in metrics]
    train_acc = [m['train']['accuracy'] for m in metrics]
    val_acc = [m['validation']['accuracy'] for m in metrics]
    learning_rates = [m['learning_rate'] for m in metrics]
    train_ppl = [m['train']['perplexity'] for m in metrics]
    val_ppl = [m['validation']['perplexity'] for m in metrics]
    
    # Loss plot
    axes[0,0].plot(epochs, train_metrics, label='Train')
    axes[0,0].plot(epochs, val_metrics, label='Validation')
    axes[0,0].set_title('Loss durante entrenamiento')
    axes[0,0].set_xlabel('Época')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Accuracy plot
    axes[0,1].plot(epochs, train_acc, label='Train')
    axes[0,1].plot(epochs, val_acc, label='Validation')
    axes[0,1].set_title('Accuracy durante entrenamiento')
    axes[0,1].set_xlabel('Época')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Learning rate plot
    axes[1,0].plot(epochs, learning_rates)
    axes[1,0].set_title('Learning Rate')
    axes[1,0].set_xlabel('Época')
    axes[1,0].set_ylabel('LR')
    axes[1,0].grid(True)
    
    # Perplexity plot
    axes[1,1].plot(epochs, train_ppl, label='Train')
    axes[1,1].plot(epochs, val_ppl, label='Validation')
    axes[1,1].set_title('Perplexity durante entrenamiento')
    axes[1,1].set_xlabel('Época')
    axes[1,1].set_ylabel('Perplexity')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Mini LLM Demo", layout="wide")
    
    st.title("Mini LLM Demo")
    st.markdown("""
    Esta demo permite interactuar con un modelo de lenguaje entrenado a pequeña escala.
    El modelo fue entrenado con textos de Wikipedia en español.
    """)
    
    # Cargar modelo y datos
    try:
        model, dataset, config, metrics = load_model_and_data()
        st.success("✅ Modelo cargado exitosamente!")
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        st.stop()
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Generación de Texto", "Métricas y Análisis", "Información del Modelo"])
    
    # Tab 1: Generación de Texto
    with tab1:
        st.subheader("🤖 Generación de Texto")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.slider(
                "Temperatura", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.7,
                help="Mayor temperatura = más creatividad pero menos coherencia"
            )
        
        with col2:
            top_k = st.slider(
                "Top-k",
                min_value=1,
                max_value=100,
                value=50,
                help="Número de tokens más probables a considerar"
            )
            
        with col3:
            repetition_penalty = st.slider(
                "Penalización por repetición",
                min_value=1.0,
                max_value=2.0,
                value=1.2,
                help="Mayor valor = menos repeticiones"
            )
        
        seed_text = st.text_area(
            "Texto inicial",
            value="En el año 1492,",
            help="El modelo continuará este texto"
        )
        
        max_length = st.slider(
            "Longitud máxima",
            min_value=10,
            max_value=200,
            value=50,
            help="Número máximo de palabras a generar"
        )
        
        if st.button("✨ Generar", type="primary"):
            with st.spinner("Generando texto..."):
                generated_text = generate_text(
                    model, dataset, config,
                    seed_text, max_length,
                    temperature, top_k,
                    repetition_penalty
                )
                if generated_text:
                    st.markdown("### Texto generado:")
                    st.markdown(f'"{generated_text}"')
    
    # Tab 2: Métricas y Análisis
    with tab2:
        st.subheader("📊 Métricas y Análisis del Modelo")
        
        if metrics:
            fig = plot_metrics(metrics)
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Métricas Finales")
                st.write(f"Train Loss: {metrics[-1]['train']['loss']:.4f}")
                st.write(f"Validation Loss: {metrics[-1]['validation']['loss']:.4f}")
                st.write(f"Train Accuracy: {metrics[-1]['train']['accuracy']:.4f}")
                st.write(f"Validation Accuracy: {metrics[-1]['validation']['accuracy']:.4f}")
            
            with col2:
                st.markdown("### Advertencias")
                st.warning("""
                ⚠️ Las métricas muestran signos de overfitting:
                - Alta accuracy pero baja calidad en generación
                - Brecha significativa entre train y validation
                """)
        else:
            st.warning("No se encontraron métricas de entrenamiento")
    
    # Tab 3: Información del Modelo
    with tab3:
        st.subheader("ℹ️ Información del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Arquitectura")
            st.write(f"- Vocabulary Size: {config.VOCAB_SIZE:,}")
            st.write(f"- Embedding Dimension: {config.EMBEDDING_DIM}")
            st.write(f"- Attention Heads: {config.NUM_HEADS}")
            st.write(f"- Transformer Layers: {config.NUM_LAYERS}")
            st.write(f"- Max Sequence Length: {config.MAX_SEQ_LENGTH}")
        
        with col2:
            st.markdown("### Recursos")
            st.write(f"Dispositivo: {config.DEVICE}")
            total_params = sum(p.numel() for p in model.parameters())
            st.write(f"Parámetros totales: {total_params:,}")
            st.write(f"Dropout: {config.DROPOUT}")
            st.write(f"Batch Size: {config.BATCH_SIZE}")

if __name__ == "__main__":
    main()