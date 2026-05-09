# Proyecto 5: Vision Transformer desde Imagen a Tokens

**Curso:** CC-0C2 Procesamiento del Lenguaje Natural  
**Tema Central:** Transformar imágenes en secuencias de patches y clasificarlas usando Transformers

---

[Link - video explicativo](https://drive.google.com/drive/folders/1KPMrOoHsrVNSmnhrb0Pb6kZcOA9XNsyw?usp=sharing)

## 1. OBJETIVO

El proyecto implementa una arquitectura **Vision Transformer (ViT)** completa desde cero para resolver cuatro pilares fundamentales:

1. **Patch Embedding:** División de imágenes en fragments espaciales (patches)
2. **Tokenización Visual:** Proyección lineal de patches a embeddings
3. **Auto-Atención Multi-Cabecera:** Relaciones globales entre patches mediante self-attention
4. **Clasificación:** Predicción de clase usando token especial [class] sobre CIFAR-10

### El Desafío: Incompatibilidad de Datos

- **El problema:** Las imágenes son tensores 3D `(H, W, C)`, pero self-attention requiere secuencias 1D `(N, d_model)`
- **El reto técnico:** Preservar información espacial al aplanar datos
- **La solución:** Patch embedding + positional encoding para reconstruir relaciones de vecindad

---

## 2. CUADERNO BASE UTILIZADO

**Workbook de Referencia:** Learner-ViT-Workbook (Excel con flujo conceptual IM2COL)

El notebook implementa exactamente el pipeline descrito en el workbook:
```
Imagen 32×32×3 
    ↓ [Sección 2: Patch Extraction]
Patches (64, 48) — flatten de 4×4×3
    ↓ [Sección 3: Patch Embedding]
Embeddings (64, 192) — proyección lineal
    ↓ [Sección 4: Token [class] + Positional Encoding]
Secuencia (65, 192) — 64 patches + 1 token special
    ↓ [Sección 5-7: Transformer Blocks]
Transformer Output (65, 192) — 12 bloques
    ↓ [Sección 7: Classification]
Logits (3,) — predicción final
```

---

## 3. LÍNEA BASE IMPLEMENTADA

### Arquitectura Completa

| Componente | Especificación |
|-----------|----------------|
| **Tamaño imagen** | 32 × 32 × 3 (RGB) |
| **Tamaño patch** | 4 × 4 (P²·C = 48 dims) |
| **Número patches** | (32/4)² = 64 |
| **Embed dimension** | 192 |
| **Cabeceras atención** | 6 (head_dim = 192/6 = 32) |
| **Capas Transformer** | 12 |
| **Dimensión MLP** | 768 (4× embed_dim) |
| **Parámetros totales** | ~11.7M |
| **Clases** | 3 (airplane, automobile, bird) |

### Módulos Principales

1. **PatchEmbedding:** Extrae patches via `unfold()` → proyección lineal
2. **PositionalEncoding:** Codificación sinusoidal (sin/cos alternados)
3. **MultiHeadAttention:** Scaled Dot-Product Attention con 6 cabeceras
4. **TransformerBlock:** Pre-Norm (LayerNorm → MHA → MLP con residual connections)
5. **VisionTransformer:** Orquestación: patches → embeddings → transformer → clasificación

---

## 4. MODIFICACIÓN REALIZADA

### Dataset Customizado
- **Origen:** CIFAR-10 (60,000 imágenes, 10 clases)
- **Filtrado a 3 clases:** airplane (ID 0), automobile (ID 1), bird (ID 2)
- **Distribución:**
  - Entrenamiento: 700 imágenes (70%)
  - Validación: 150 imágenes (15%)
  - Test: 150 imágenes (15%)

### Estrategia de Entrenamiento
- **Optimizador:** AdamW (lr=1e-3, weight_decay=0.05)
- **Scheduler:** Warmup (5 epochs) + Cosine Annealing
- **Loss:** CrossEntropyLoss
- **Epochs:** 20 (con early stopping si validación no mejora en 5 epochs)
- **Técnicas de regularización:**
  - Dropout: 0.1 en embeddings y MLP
  - Gradient clipping: max_norm=1.0
  - Data augmentation: RandomHorizontalFlip, RandomAffine, ColorJitter

---

## 5. RESULTADOS PRINCIPALES

### Métricas de Entrenamiento

```
Total de parámetros: 11,732,227
Parámetros entrenables: 11,732,227

Entrenamiento completado después de ~X epochs (early stopping)
```

### Desempeño en Validación
- **Mejor accuracy en validación:** 62%
- **Curvas de entrenamiento:** Gráficos de Loss y Accuracy mostrados al final

### Predicciones en Test
- Matriz de confusión visual (imágenes con predicción/etiqueta verdadera)
- Colores: Verde (acierto), Rojo (error)

---

## 6. EVIDENCIA DE CÁLCULOS INTERNOS

### Verificación de Dimensiones (Forward Pass)

```python
Input shape:         (1, 3, 32, 32)  # Imagen CIFAR-10
↓ Patch Extraction
Patches:             (1, 64, 48)     # 64 patches × 48 dims (4×4×3)
↓ Patch Embedding  
Embeddings:          (1, 64, 192)    # Proyección lineal
↓ Add [class] token
+ Class token:       (1, 1, 192)
Sequence:            (1, 65, 192)    # 64 patches + 1 class token
↓ Positional Encoding
+ Position enc:      (1, 65, 192)    
Encoded:             (1, 65, 192)
↓ Transformer blocks (×12)
Output:              (1, 65, 192)    # Dimensión preservada
↓ Take [class] token (índice 0)
Class output:        (1, 192)
↓ Classification head
Logits:              (1, 3)          # 3 clases
```

### Matriz de Codificación Posicional (Visualización)

En el notebook se genera:
- **Heatmap:** Matriz 64×192 mostrando valores sin/cos para cada posición
- **Dimensiones individuales:** Gráfico de las 5 primeras dimensiones en función de índice de patch

Patrón esperado: Frecuencias sinusoidales con período creciente.

### Atención Multi-Cabecera (Ejemplo)

Salida durante forward pass:
```python
Attention weights shape: (1, 6, 65, 65)
# 1 imagen, 6 cabeceras, 65×65 matriz de atención
```

Interpretación:
- Cada entrada `attn[batch, head, i, j]` = peso de atención del patch i hacia patch j
- La diagonal tiende a tener valores altos (autorreferencia)
- Patrones espaciales emergen: parches cercanos atienden juntos

---

## 7. ERRORES Y LIMITACIONES

### Problemas Identificados

1. **Dataset Limitado:**
   - Solo 700 imágenes de entrenamiento (muy pequeño para ViT)
   - Riesgo elevado de overfitting a pesar de regularización

2. **Confusión entre Clases Similares:**
   - Airplane vs. Bird: ambos tienen estructura alada
   - Automobile vs. Airplane: ambos tienen forma alargada
   - El modelo aún confunde estas clases visualmente similares

3. **Convergencia Lenta:**
   - ViT típicamente requiere datasets grandes (ImageNet 1.2M imágenes)
   - Con 700 imágenes, el entrenamiento puede no alcanzar óptimo global

4. **Computational Overhead:**
   - Atención O(N²): 65×65 = 4,225 operaciones por cabecera
   - 12 bloques × 6 cabeceras = 72 cálculos de atención por forward pass

---

## 8. CONCLUSIÓN TÉCNICA

### Aprendizajes Clave

1. **Patch Embedding es la clave:** La proyección lineal de patches aplanados a embeddings permite que un transformer procese imágenes como secuencias, pero requiere **positional encoding** para preservar información espacial.

2. **Multi-Head Attention como mecanismo global:** A diferencia de convoluciones locales, cada patch puede atender a todos los demás. Esto captura relaciones globales pero a costo O(N²).

3. **Pre-Norm + Residuals son cruciales:** Sin conexiones residuales, gradientes no fluyen a través de 12 capas. Pre-Norm (LayerNorm antes de operación) estabiliza el entrenamiento.

4. **ViT requiere escala:** Con solo 700 imágenes, el modelo tiende a overfitting. Los artículos originales usaron ImageNet (1.2M imágenes) o pre-entrenamiento.

5. **Token [class] como agregación:** El token especial al inicio actúa como "cabecera de lectura" que integra información de todos los patches para predicción final.

### Aporte al Curso

Este proyecto valida experimentalmente el flujo **IM2COL del workbook**, demostrando:
- ✓ Cómo convertir imagen → patches aplanados
- ✓ Proyección a embeddings aprendibles
- ✓ Rol del embedding posicional (sin/cos)
- ✓ Funcionamiento de multi-head attention en visión
- ✓ Arquitectura end-to-end entrenada en clasificación

### Recomendaciones Futuras

1. Usar dataset más grande (CIFAR-100 o ImageNet100)
2. Implementar Data Augmentation más agresiva (Mixup, CutMix)
3. Comparar con CNN baseline (ResNet-18)
4. Explorar tamaño de patch variable (2×2, 8×8, 16×16)
5. Visualizar mapas de atención por cabecera para interpretabilidad

---

## Estructura del Notebook

| Sección | Contenido | Salida Principal |
|---------|----------|------------------|
| 1 | Setup e Imports | Verificación GPU |
| 2 | Patch Extraction | Visualización grid de patches |
| 3 | Patch Embedding | Verificación de dimensiones |
| 4 | Token [class] + Pos Encoding | Matrices de encoding |
| 5 | Multi-Head Attention | Pesos de atención |
| 6 | Transformer Block | Forward pass completo |
| 7 | ViT Completo | Resumen de parámetros |
| 8 | Dataset CIFAR-10 | Distribución de clases |
| 9 | Training Loop | Curvas de loss/accuracy |
| Final | Evaluación + Predicciones | Matriz confusión visual |