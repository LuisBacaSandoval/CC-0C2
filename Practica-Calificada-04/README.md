# Práctica Calificada 4 - CC0C2 — Proyecto 6: Evaluación de retrieval

**Estudiante:** Luis Baca Sandoval
**Cuaderno base:** `Cuaderno23-CC0C2.ipynb` (Evaluación de retrieval y grounded generation)
**Notebook entregable:** `PracticaCalificada4-CC0C2.ipynb`

## Objetivo

Construir un dataset de evaluación con consultas y documentos relevantes conocidos (*gold docs*),
implementar las métricas **precision@k**, **recall@k**, **MRR** y **nDCG**, y **comparar dos
configuraciones de retrieval** con esas métricas. El foco es la **recuperación** (no la
generación): es el techo de calidad de cualquier sistema RAG.

## Cuaderno base

Parto del `Cuaderno23-CC0C2.ipynb`, del que reutilizo el corpus, los recuperadores (BM25 simplificado,
denso TF-IDF, híbrido) y las métricas `precision@k`, `recall@k` (versión binaria) y `MRR`. Todo es
**stdlib de Python**: sin descargas, sin internet, sin GPU, 100% determinista.

## Resumen de la línea base

La línea base es **BM25 puro** (recuperación léxica) evaluado sobre un corpus de **12 documentos**
(8 del cuaderno base + 4 distractores que comparten vocabulario pero no son relevantes) y un
benchmark de **6 consultas parafraseadas** (con desajuste de vocabulario respecto a los gold docs).
Métricas de la línea base: `P@3=0.333`, `R@3=0.833`, `MRR=0.917`, `nDCG@5=0.810`.

## Modificación realizada

Aportes propios sobre el cuaderno base:

1. **nDCG@k agregado desde cero** (`dcgAtK`, `idcgAtK`, `ndcgAtK`) — el Cuaderno23 no lo trae —
   con una **celda de validación** contra valores calculados a mano.
2. **Corrección de `recall@k`**: del *hit-rate* binario original a recall real
   `|gold ∩ top_k| / |gold|`, que penaliza correctamente recuperar solo uno de varios relevantes.
3. **Corpus y benchmark enriquecidos**: distractores + consultas parafraseadas para que el problema
   sea discriminativo (sin esto, todas las configuraciones empatan).
4. **Comparación de configuraciones** (Ejercicio B): BM25 vs denso vs híbrido vs **BM25 + expansión
   de consulta** (basada en `rewriteQuery`/`multiQueryRetrieve` del Cuaderno23).
5. **Barrido de `top_k`** (Ejercicio A): cómo evolucionan P@k, R@k y nDCG@k.

## Cómo ejecutar el notebook

Requisitos: solo Python 3 con `nbformat`/Jupyter para abrirlo. No hay dependencias externas
(`pip`), ni modelos que descargar.

```bash
# Desde la carpeta Practica-Calificada-04/
# Opción 1: abrir en VS Code / Jupyter y "Run All".
# Opción 2 (verificación reproducible desde consola):
jupyter nbconvert --to notebook --execute PracticaCalificada4-CC0C2.ipynb --output salida.ipynb
```

Con `SEED=42` y al ser determinista, dos ejecuciones producen resultados idénticos.

## Principales resultados

Comparación de configuraciones (`k=3`, `nDCG@5`):

| Configuración    | P@3   | R@3   | MRR   | nDCG@5 |
|------------------|-------|-------|-------|--------|
| BM25 (base)      | 0.333 | 0.833 | 0.917 | 0.810  |
| Denso TF-IDF     | 0.333 | 0.833 | 0.917 | 0.810  |
| Híbrido α=0.6    | 0.333 | 0.833 | 0.917 | 0.810  |
| **BM25+expansión** | **0.444** | **1.000** | **1.000** | **1.000** |

- Las variantes léxicas (denso/híbrido) **no mejoran** sobre BM25: siguen siendo bolsa de palabras
  y no resuelven el desajuste de vocabulario.
- La **expansión de consulta** sí mejora claramente (Δrecall@3 = +0.167, ΔnDCG@5 = +0.190): al
  inyectar los términos que faltaban, los gold docs suben de posición. La mejora es **medible, no
  cosmética**.
- El barrido de `top_k` muestra la **tensión precisión–recall**: al crecer `k`, `P@k` cae
  (0.833→0.111) mientras `R@k` sube (0.667→1.000) y `nDCG@k` se satura.

## Limitaciones

- El "denso" es TF-IDF (léxico), **no** embeddings semánticos reales; en un sistema real se
  sustituiría por `sentence-transformers` conservando el mismo harness.
- Dataset pequeño (12 docs, 6 consultas) y autoetiquetado: hay **sesgo del anotador** y las cifras
  absolutas no son generalizables, solo comparables entre configuraciones bajo idénticas condiciones.
- El mapa de sinónimos de la expansión lo definí yo: parte de su ventaja depende de esa elección.

## Qué se muestra en el video

- Objetivo, cuaderno base y problema técnico (evaluar retrieval).
- Línea base BM25 y su evidencia interna (scores por documento, métricas por consulta).
- **Codificación en vivo 1 (Ejercicio A):** cambiar `top_k`; predicción → ejecución → interpretación
  de la tensión P/R.
- **Codificación en vivo 2 (Ejercicio B):** comparar BM25 vs BM25+expansión; predicción →
  ejecución → interpretación.
- Explicación de embeddings/vectores, scores de similitud y métricas (P@k, R@k, MRR, nDCG), con sus
  dimensiones.
- Comparación línea base vs variante, análisis de errores, respuestas a preguntas avanzadas y cierre.

### Guion sugerido (> 12 min, objetivo 14–18)

1. (0:00–1:30) Presentación del proyecto y cuaderno base.
2. (1:30–3:00) Problema técnico y por qué evaluar retrieval; teoría de las 4 métricas.
3. (3:00–5:00) Corpus, distractores, benchmark y tabla de dimensiones (V, matriz doc-término).
4. (5:00–7:00) nDCG agregado + validación a mano; corrección de recall.
5. (7:00–8:30) Línea base BM25 con evidencia interna.
6. (8:30–11:00) **En vivo A:** variar `top_k`; predecir, ejecutar, interpretar.
7. (11:00–14:00) **En vivo B:** BM25 vs BM25+expansión; predecir, ejecutar, interpretar.
8. (14:00–16:00) Comparación final, análisis de errores, preguntas avanzadas.
9. (16:00–17:30) Limitaciones, cierre (qué/por qué/qué significan) y puente al curso.

## Declaración de autoría y uso de IA

> Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta Práctica.
> Si utilicé herramientas de IA, las usé como apoyo para redacción, depuración o consulta, pero la
> implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mía.

Uso concreto de IA: revisión de redacción del README y de explicaciones teóricas; contraste de la
fórmula de nDCG y diseño de los casos de validación. La implementación de `ndcgAtK`, la corrección
de `recallAtK`, el diseño del corpus/benchmark, la variante de expansión y todas las
interpretaciones son míos. Cada celda del notebook indica si fue `REUTILIZADO`, `MODIFICADO` o
`AGREGADO`.
