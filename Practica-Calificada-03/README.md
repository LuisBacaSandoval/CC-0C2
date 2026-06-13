# Proyecto 1: Modelo Causal Decoder-Only desde Cero

**Estudiante:** Baca Sandoval Luis Jhonatan
**Curso:** CC0C2 — Procesamiento del Lenguaje Natural · Práctica Calificada 3

## Objetivo

Reconstruir, **desde cero** y sin usar `nn.MultiheadAttention` ni `nn.TransformerEncoderLayer`,
un modelo de lenguaje causal decoder-only estilo GPT-2: atención causal multi-head con máscara
triangular, bloque decoder con Pre-LayerNorm, entrenamiento por máxima verosimilitud autoregresiva
y generación con distintas políticas de decoding.

## Cuaderno base

`Cuaderno12-CC0C2.ipynb` (modelo causal decoder-only pequeño). Se reutilizan pipeline de
tokenización, dataset causal y utilidades de generación; se adaptan entrenamiento y comparación
apoyándose puntualmente en los Cuadernos 13 y 15. Cada celda indica si su contenido es
**reutilizado**, **adaptado** o **desde cero**.

## Resumen de la línea base

- **Arquitectura propia** (`CausalDecoderLM`): embedding + codificación posicional sinusoidal,
  `L=4` bloques decoder Pre-LN (atención causal multi-head + FFN GELU), LayerNorm final y LM head.
- **Datos:** corpus IMDB (solo texto), tokenizador BPE de `distilgpt2`.
- **Entrenamiento:** entropía cruzada por token, AdamW + scheduler coseno, semilla 42.
- **Hiperparámetros:** `SEQ_LEN=64`, `D_MODEL=256`, `N_HEADS=4`, `N_LAYERS=4`, `D_FF=1024`.
- Se evalúa con **perplexidad** y se comparan políticas de decoding (greedy, top-k, top-p, safe).

## Modificación realizada

- **Ejercicio A (común):** variación de `temperature` (0.3 vs 1.5) para observar cómo se agudiza
  o aplana la distribución del siguiente token.
- **Ejercicio B (específico):** reducción de `N_LAYERS` de **4 → 2**, midiendo el impacto en
  parámetros entrenables y en la perplexidad de validación, manteniendo el resto constante.

## Cómo ejecutar el notebook

Se ejecuta sobre el **entorno Docker del curso**, que ya trae todas las dependencias
(`torch`, `transformers`, `datasets`, `pandas`, `matplotlib`); no es necesario instalar nada.

1. Levantar el contenedor del curso y abrir `Proyecto1-CC0C2.ipynb`.
2. Ejecutar las celdas en orden (Run all).
3. La primera ejecución descarga el tokenizador/modelo `distilgpt2` y el dataset IMDB.
   Funciona en CPU; con GPU el entrenamiento es notablemente más rápido.

## Principales resultados

- El modelo desde cero entrena correctamente y la perplexidad de validación desciende a lo
  largo de las épocas.
- La visualización de los pesos de atención del último bloque muestra el patrón triangular
  causal esperado (cero por encima de la diagonal).
- La variante de 2 capas tiene menos parámetros entrenables; la comparación con la base separa
  el efecto de **capacidad/arquitectura** del de **datos y cómputo**.
- Frente a `distilgpt2` preentrenado, el texto generado es más pobre: el gap refleja diferencias
  de escala, no un error de implementación.

## Limitaciones

- Corpus y cómputo reducidos: el modelo aprende patrones locales, no razonamiento lingüístico.
- La variante de 2 capas se entrena solo 1 época (demo), por lo que la comparación
  estrictamente justa es la de parámetros/arquitectura; la perplexidad es indicativa.
- La generación recomputa todo el contexto en cada paso (sin KV cache).

## Qué se muestra en el video

Objetivo y cuaderno base, línea base y arquitectura propia, codificación en vivo de los dos
ejercicios (temperature y `N_LAYERS`) con predicción e interpretación, evidencia de cálculo
interno (formas de `x`, `y`, `logits` y máscara causal), comparación base vs variante y respuesta
a las preguntas avanzadas.

## Declaración de autoría y uso de IA

> Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta práctica.
> Si utilicé herramientas de IA, las usé como apoyo para redacción, depuración o consulta, pero la
> implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mía.

Uso de IA en este trabajo:

- Para revisar la redacción y estructura del notebook y del README.
- Para depurar errores de forma de tensores.
- Para aclarar advertencias de PyTorch.
