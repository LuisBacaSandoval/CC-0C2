# Examen Parcial: Vision Transformer con Cambio de Patch Size

**Curso:** CC-0C2 Procesamiento del Lenguaje Natural  
**Tema:** Análisis del impacto del tamaño de patch en Vision Transformer

---

[Link - video explicativo](https://drive.google.com/drive/folders/1GLEJyLpTQJbtAALes9dTuRKrlnoJMQkb)

## Objetivo

Comprender cómo variar el **tamaño de patch** en ViT afecta el trade-off entre:
- **Resolución:** Patches pequeños capturan más detalle
- **Complejidad:** Secuencias más largas = mayor costo computacional O(N²)

### Pipeline ViT

```
Imagen 32×32×3
    ↓ [Extracción de Patches]
Patches (16 o 64) — tamaño 8×8 o 4×4
    ↓ [Embedding]
Embeddings (dim=192) 
    ↓ [Positional Encoding + [class] token]
Secuencia (17 o 65, 192)
    ↓ [Transformer Blocks ×12]
    ↓ [Classification Head]
Logits (3 clases)
```

---

## Configuración

| Parámetro | Valor |
|-----------|-------|
| **Imágenes** | CIFAR-10 filtrado a 3 clases |
| **Large patch** | 8×8 → 16 patches |
| **Small patch** | 4×4 → 64 patches |
| **Embed dim** | 192 |
| **Heads** | 6 |
| **Capas** | 12 |
| **Parámetros** | ~11.7M |

---

## Comparativa Patches

| Métrica | 4×4 (Small) | 8×8 (Large) |
|---------|------------|-----------|
| **Num patches** | 64 | 16 |
| **Seq length** | 65 | 17 |
| **Costo atención** | O(65²) | O(17²) |
| **Detalle** | Alto | Bajo |
| **Velocidad** | Lenta | Rápida |

---

## Resultados

- Análisis comparativo de accuracies y tiempos de entrenamiento
- Visualización de trade-off complejidad vs. precisión
- Matriz de confusión para ambas configuraciones

---

## Conclusión

El tamaño de patch es un **hiperparámetro crítico** que balancea capacidad representacional (patches pequeños) vs. eficiencia computacional (patches grandes). Con dataset limitado (700 imágenes), ambas configuraciones sufren overfitting, pero el análisis ilustra el diseño fundamental de ViT.

