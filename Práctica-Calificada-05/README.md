# Práctica Calificada 5 - CC0C2 — Proyecto 5: Modelo de difusión básico

**Estudiante:** Luis J. Baca Sandoval
**Cuaderno base:** `Cuaderno25-CC0C2.ipynb` (Semana 12: modelos de difusión, DDPM/DDIM)
**Notebook entregable:** `PracticaCalificada05-CC0C2.ipynb`

## Objetivo

Entrenar **un único denoiser** de difusión sobre MNIST y usarlo para demostrar, con evidencia interna y
comparación cuantitativa, el **trade-off entre calidad de generación y tiempo de inferencia**. Se controlan
dos perillas manteniendo los pesos del modelo fijos: (1) el **número de pasos de muestreo** y (2) el
**tipo de muestreador** — DDPM estocástico vs **DDIM determinista** (`eta=0`) —, de modo que la comparación
aísla el efecto del *sampling* y no del entrenamiento.

## Cuaderno base

Parto del `Cuaderno25-CC0C2.ipynb`, del que reutilizo el *schedule* lineal de ruido, el proceso forward en
forma cerrada (`q_sample`), el `TinyDenoiser` (con *time embedding* estilo FiLM) y el muestreo DDPM
didáctico (`sample_toy_ddpm`). Cada celda de código está marcada en su primera línea como
`REUTILIZADO`, `MODIFICADO` o `AGREGADO por el estudiante`. Todo corre sobre MNIST (`ylecun/mnist`) con
`torch`/`torchvision`/`datasets`.

## Resumen de la línea base

La línea base es **DDPM estocástico completo**: `T=200` pasos, reinyectando ruido en cada paso, sobre un
`TinyDenoiser` (3 convoluciones + *time embedding*) entrenado por `TRAIN_STEPS=300` minimizando el MSE
entre el ruido real y el predicho. Se generan `N_SAMPLES=16` imágenes con semilla `SEED=42`.
Métricas de la línea base (200 pasos): `tiempo=0.377 s`, `nitidez=0.0887`, `confianza=0.675`,
`entropía=0.950` (referencia de imágenes reales: `nitidez=0.1307`, `confianza=0.941`, `entropía=0.136`).

## Modificación realizada

Aportes propios sobre el cuaderno base:

1. **Muestreador respaciado `sample_ddim` agregado desde cero**, que generaliza DDPM y DDIM con dos
   parámetros: `num_steps` (subsecuencia de tiempos, controla la latencia) y `eta` (controla el ruido
   reinyectado: `eta=1.0` ≈ DDPM estocástico, `eta=0.0` = DDIM determinista).
2. **Ejercicio A (modificación común):** barrido de `num_steps ∈ {200, 100, 50, 20}` (y una corrida en vivo
   con `10`) con `eta=1.0` fijo, midiendo tiempo de generación y calidad.
3. **Ejercicio B (modificación específica del Proyecto 5):** comparación, al mismo `num_steps=20`, entre
   DDPM estocástico (`eta=1.0`) y DDIM determinista (`eta=0.0`), incluyendo una **prueba de
   reproducibilidad** (misma semilla → diferencia máxima entre dos corridas).
4. **Evaluador semántico agregado:** un `SmallCNN` clasificador de dígitos entrenado sobre MNIST para medir
   confianza y entropía media de las muestras generadas, más una métrica de **nitidez** (gradiente espacial
   medio), para no depender solo de inspección visual.
5. **Tabla comparativa** de todas las configuraciones (línea base + barrido A + variante B) con tiempo,
   nitidez, confianza y entropía en una sola vista.

## Cómo ejecutar el notebook

Se ejecuta sobre el **entorno Docker del curso**, que ya trae todas las dependencias (`torch`,
`torchvision`, `datasets`, `numpy`, `matplotlib`, `pillow`); no es necesario instalar nada.

1. Levantar el contenedor del curso y abrir `PracticaCalificada05-CC0C2.ipynb`.
2. Ejecutar las celdas en orden (Run all).
3. La primera ejecución descarga el dataset MNIST (`ylecun/mnist`). `get_device()` selecciona
   automáticamente `cuda`/`mps`/`cpu` según lo disponible; el pipeline también corre en CPU. Con
   `SEED=42` fijo en `set_seed`, las corridas DDIM (`eta=0`) son reproducibles bit a bit.

## Principales resultados

Comparación de configuraciones (`N_SAMPLES=16`, denoiser entrenado una sola vez):

| Configuración        | Pasos | Tiempo (s) | Nitidez | Confianza | Entropía |
|-----------------------|-------|-----------|---------|-----------|----------|
| BASE DDPM completo     | 200   | 0.377     | 0.0887  | 0.675     | 0.950    |
| DDPM (eta=1)           | 200   | 0.393     | 0.5694  | 0.681     | 0.792    |
| DDPM (eta=1)           | 100   | 0.127     | 0.2604  | 0.684     | 0.905    |
| DDPM (eta=1)           | 50    | 0.058     | 0.1559  | 0.762     | 0.754    |
| DDPM (eta=1)           | 20    | 0.029     | 0.1005  | 0.600     | 1.119    |
| **DDIM (eta=0)**       | 20    | 0.040     | 0.1584  | 0.597     | 1.066    |

- El **tiempo de generación escala casi linealmente** con `num_steps`: de 0.393 s (200 pasos) a 0.029 s
  (20 pasos), una reducción de más de 10×.
- La **calidad se degrada gradualmente** al reducir pasos: la confianza del clasificador cae y la entropía
  sube conforme `num_steps` disminuye, evidenciando ruido residual no removido.
- La **nitidez no es un proxy confiable por sí sola**: el valor más alto (0.5694, DDPM 200 pasos) es
  **4× el de imágenes reales** (0.1307) porque el ruido de alta frecuencia también genera gradientes
  grandes; por eso se combina con confianza/entropía del clasificador.
- **Prueba de reproducibilidad:** con la misma semilla, dos corridas de DDIM (`eta=0`) e incluso de DDPM
  (`eta=1`, tal como está instrumentado el experimento) dan diferencia máxima `0.00e+00`, confirmando que
  el pipeline es determinista dado el mismo estado de RNG.

## Limitaciones

- **Pocos pasos** dejan ruido residual y dígitos menos coherentes: es el modo de falla esperado al
  recortar demasiado la trayectoria de *denoising*.
- La **nitidez es un proxy barato**, no reemplaza FID: no mide distancia entre distribuciones de
  características, solo da una señal rápida y ejecutable en CPU.
- El **clasificador evaluador** (`SmallCNN`) se entrena pocos pasos y comparte el sesgo de dominio de
  MNIST; puede sobre/infra-estimar la confianza.
- `TinyDenoiser` es un modelo de juguete (pocas convoluciones, 300 pasos de entrenamiento, `T=200`) por
  límites de cómputo; un UNet más profundo mejoraría la calidad absoluta pero no cambia las conclusiones
  sobre el trade-off pasos/tiempo ni sobre el determinismo de DDIM.

## Qué se muestra en el video

- Objetivo, cuaderno base y problema técnico (trade-off calidad–latencia en difusión).
- Marco teórico: proceso forward markoviano, forma cerrada `q_sample`, objetivo de entrenamiento
  (predicción de ruido) y muestreo reverse (DDPM vs DDIM), conectado línea por línea con el código.
- Línea base DDPM completo con evidencia interna (tabla de dimensiones, formas de tensores).
- **Codificación en vivo 1 (Ejercicio A):** variar `num_steps`; predicción → ejecución → interpretación
  del trade-off tiempo/calidad.
- **Codificación en vivo 2 (Ejercicio B):** DDPM (`eta=1`) vs DDIM (`eta=0`) al mismo `num_steps`;
  predicción → ejecución → interpretación, incluyendo la prueba de reproducibilidad.
- Comparación cuantitativa base vs variantes (tabla), análisis de errores, respuestas a las preguntas
  avanzadas y cierre con el puente al curso (Stable Diffusion, latencia/MLOps).

### Guion sugerido (> 12 min, objetivo 14–18)

1. (0:00–1:30) Presentación del proyecto y cuaderno base.
2. (1:30–3:30) Teoría: proceso forward, `q_sample`, objetivo de entrenamiento, muestreo reverse DDPM/DDIM.
3. (3:30–5:30) Denoiser, entrenamiento y línea base DDPM con evidencia interna.
4. (5:30–8:00) **En vivo A:** barrido de `num_steps`; predecir, ejecutar, interpretar.
5. (8:00–10:30) **En vivo B:** DDPM vs DDIM al mismo `num_steps` y prueba de reproducibilidad.
6. (10:30–13:00) Métricas cuantitativas (nitidez, confianza, entropía) y tabla comparativa.
7. (13:00–15:30) Análisis de errores, preguntas avanzadas seleccionadas.
8. (15:30–17:00) Limitaciones, cierre (qué/por qué/qué significan) y puente al curso.

## Declaración de autoría y uso de IA

> Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta Práctica.
> Si utilicé herramientas de IA, las usé como apoyo para redacción, depuración o consulta, pero la
> implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mía.

Uso concreto de IA: revisión de redacción de las secciones teóricas y del README; depuración de la fórmula
del paso DDIM (manejo de `alpha_bar_prev` en el último paso) y verificación de formas de tensores. La
línea base proviene de `Cuaderno25-CC0C2.ipynb` (marcada como reutilizada); el muestreador DDIM, el barrido
de pasos, las métricas de evaluación y las comparaciones son aporte propio. Cada celda del notebook indica
si fue `REUTILIZADO`, `MODIFICADO` o `AGREGADO`.
