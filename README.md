# MiniLLM: Implementación de un Modelo de Lenguaje a Pequeña Escala

## Descripción
MiniLLM es una implementación de un modelo de lenguaje basado en la arquitectura transformer. El proyecto incluye el proceso completo desde la obtención de datos hasta la evaluación del modelo, diseñado para comprender los fundamentos de los LLMs.

## Estructura del Proyecto
```
miniLLM/
├── checkpoints/         # Almacena modelos entrenados
├── data/               # Datos crudos y procesados
│   ├── processed/
│   └── raw/
├── evaluation_results/ # Resultados de evaluación
├── logs/              # Registros de entrenamiento
└── src/               # Código fuente
    ├── config.py
    ├── data_utils.py
    ├── evaluator.py
    ├── model.py
    ├── preprocess.py
    └── trainer.py
```

## Características Principales
- Arquitectura transformer simplificada
- Pipeline completo de procesamiento de datos
- Sistema de evaluación comprehensivo
- Generación de texto
- Visualización de métricas

## Requisitos
```
Python 3.8+
PyTorch 1.8+
numpy
pandas
matplotlib
seaborn
nltk
tqdm
requests
```


## Configuración
Los principales parámetros se pueden ajustar en `config.py`:
- VOCAB_SIZE = 5000
- EMBEDDING_DIM = 256
- NUM_HEADS = 4
- NUM_LAYERS = 2
- MAX_SEQ_LENGTH = 128

## Resultados y Métricas
El modelo genera automáticamente:
- Gráficas de pérdida y accuracy
- Métricas de perplexity
- Scores BLEU
- Análisis de sesgos
- Métricas de rendimiento

## Limitaciones Conocidas
- Overfitting significativo
- Generación de texto con coherencia limitada
- Sesgos en las predicciones
- Recursos computacionales limitados
