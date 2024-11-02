import streamlit as st
import torch
from src.config import Config
from src.model import MiniGPT
from src.data_utils import WikiDataset
import pickle
from pathlib import Path

def load_model_and_data():
    """Carga el modelo entrenado y los datos necesarios."""
    config = Config()
    
    try:
        # Cargar vocabulario
        vocab_path = config.PROCESSED_DIR / "vocab.pkl"
        if not vocab_path.exists():
            raise FileNotFoundError("No se encontró el archivo de vocabulario")
            
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Crear dataset
        dataset = WikiDataset(['dummy text'], config.VOCAB_SIZE, config.MAX_SEQ_LENGTH)
        dataset.vocab = vocab_data['vocab']
        dataset.reverse_vocab = vocab_data['reverse_vocab']
        
        # Crear modelo
        model = MiniGPT(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        # Cargar pesos del modelo
        checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError("No se encontró el archivo del modelo")
            
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, dataset, config
        
    except Exception as e:
        st.error(f"Error al cargar el modelo y datos: {str(e)}")
        st.stop()

def generate_text(model, dataset, config, seed_text, max_length=50, 
                 temperature=0.7, top_k=50):
    """Genera texto usando el modelo."""
    try:
        tokens = dataset.encode(seed_text)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == dataset.vocab['[EOS]']:
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return dataset.decode(input_ids[0].tolist())
    except Exception as e:
        st.error(f"Error generando texto: {str(e)}")
        return ""

def main():
    st.title("Mini LLM Demo")
    
    # Cargar modelo y datos
    try:
        model, dataset, config = load_model_and_data()
        st.success("Modelo cargado exitosamente!")
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        st.stop()
    
    # Generación de texto
    st.subheader("Generación de Texto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperatura", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7,
            help="Controla la creatividad del modelo"
        )
    
    with col2:
        top_k = st.slider(
            "Top-k",
            min_value=1,
            max_value=100,
            value=50,
            help="Número de tokens a considerar en cada paso"
        )
    
    seed_text = st.text_area(
        "Texto inicial",
        value="En el año 1492,",
        help="Escribe el inicio del texto que quieres que el modelo continue"
    )
    
    max_length = st.slider(
        "Longitud máxima",
        min_value=10,
        max_value=200,
        value=50,
        help="Número máximo de tokens a generar"
    )
    
    if st.button("Generar", type="primary"):
        with st.spinner("Generando texto..."):
            generated_text = generate_text(
                model, dataset, config,
                seed_text, max_length,
                temperature, top_k
            )
            if generated_text:
                st.markdown("### Texto generado:")
                st.write(generated_text)
    
    # Información del modelo
    with st.expander("Ver información del modelo"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Parámetros del modelo")
            st.write(f"- Vocabulary Size: {config.VOCAB_SIZE:,}")
            st.write(f"- Embedding Dim: {config.EMBEDDING_DIM}")
            st.write(f"- Num Heads: {config.NUM_HEADS}")
            st.write(f"- Num Layers: {config.NUM_LAYERS}")
        
        with col2:
            st.markdown("### Recursos")
            st.write(f"Dispositivo: {config.DEVICE}")
            total_params = sum(p.numel() for p in model.parameters())
            st.write(f"Parámetros totales: {total_params:,}")

if __name__ == "__main__":
    main()