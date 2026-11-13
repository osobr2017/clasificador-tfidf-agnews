# Clasificador supervisado con TF-IDF - AG News

## Descripción
Pipeline de clasificación de noticias (World, Sports, Business, Sci_Tech)
sobre AG News: preprocesamiento, vectorización TF-IDF, entrenamiento y
evaluación.

## Pipeline
1. **Carga:** `ag_news_train.csv` (8.000 noticias) y `ag_news_test.csv` (2.000).
2. **Preprocesamiento (Módulo 2):** des-escape de HTML, eliminación de tags,
   entidades, URLs, firmas de agencia y puntuación; minúsculas; lematización
   con SpaCy (`en_core_web_sm`).
3. **Vectorización:** `TfidfVectorizer` ajustado solo sobre el train
   (`fit_transform`) y solo `transform` sobre el test, para evitar data leakage.
4. **Modelo:** Regresión Logística.
5. **Evaluación:** `classification_report` y matriz de confusión en el test.

## Parámetros del vectorizador
- `max_features=20000`: limita el vocabulario para reducir overfitting y memoria.
- `ngram_range=(1, 2)`: unigramas y bigramas, para capturar expresiones
  como "prime minister" u "oil price".
- `stop_words="english"`: elimina palabras vacías sin valor predictivo.

Se compararon 4 combinaciones de `max_features` (5000, 20000) y `ngram_range`
((1,1), (1,2)) usando un 20% del train como validación, sin tocar el test.
Accuracy de validación entre 0.887 y 0.893; ganó la configuración elegida (0.893).

## Justificación del modelo
Elegí Regresión Logística porque:
- Funciona muy bien con matrices esparsas y de alta dimensión como TF-IDF.
- Es rápida de entrenar y un baseline estándar en clasificación de texto.
- Es interpretable: los coeficientes indican qué palabras empujan hacia cada clase.
- Frente a Naive Bayes, no asume independencia entre palabras; frente a SVM,
  rinde de forma similar pero es más simple de explicar y de extender.

## Resultados (test)
| Clase    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Business | 0.83      | 0.86   | 0.84 |
| Sci_Tech | 0.88      | 0.87   | 0.87 |
| Sports   | 0.95      | 0.96   | 0.96 |
| World    | 0.91      | 0.88   | 0.90 |

Accuracy global: **0.89** (macro F1 0.89). La accuracy de validación (0.893)
y la de test coinciden, lo que indica que no hay overfitting ni fuga de datos.

## Análisis preliminar (matriz de confusión)
- **Sports** es la más fácil (F1 0.96): vocabulario muy propio (equipos, partidos).
- **Business y Sci_Tech** son las más difíciles (F1 0.84 y 0.87) y se confunden
  entre sí (47 y 40 casos), porque las noticias de empresas tecnológicas
  pertenecen a ambos temas.
- **World** se confunde sobre todo con Business (32 casos), por noticias de
  política económica y petróleo.

## Instalación
pip install -r requirements.txt
python -m spacy download en_core_web_sm