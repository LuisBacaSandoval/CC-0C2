# Proyecto 5: Vision Transformer desde Imagen a Tokens

### El Desafío: Incompatibilidad de Datos

- **El problema:** Las imágenes son tensores tridimensionales (alto, ancho, canales), pero el mecanismo de self-attention requiere secuencias unidimensionales.
- **El reto técnico:** Reducir dimensiones conservando la integridad de la información espacial.
- **La solución:** Fragmentar la imagen en patches e inyectar mecanismos para reconstruir la relación de vecindad perdida al aplanar los datos.

### Objetivos del proyecto

Implementar la arquitectura de un Vision Transformer (ViT) para dominar cuatro pilares:

- La división de matrices visuales en patches.
- La proyección de patches a tokens computables.
- La dinámica del mecanismo de self-attention.
- El flujo de datos para la clasificación final de imágenes.

## Pipeline
![Pipeline vision transformer](https://cdn.learnopencv.com/wp-content/uploads/2023/02/04091333/image.png)