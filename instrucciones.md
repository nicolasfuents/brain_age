# Guía de Referencia para el proyecto de Predicción de Edad Cerebral (BAG)

Este documento sirve como instrucción para abordar cada tarea requerida. 

> [!IMPORTANT]
> **Requisito Obligatorio de Entorno de Ejecución (Conda):**
> Todo código, script o comando de Python en este repositorio debe ejecutarse obligatoriamente utilizando el intérprete de Python que se encuentra en el entorno Conda `brain_age_env` del usuario.
> **Ruta absoluta del intérprete:** `/home/nfuentes/miniforge3/envs/brain_age_env/bin/python`
> Nunca uses el python del sistema o de otros entornos.

> [!IMPORTANT]
> **Requisito Obligatorio para escribir las respuestas**
> Como estamos en una GUI IDE, tus respuestas tienen que estar escritas de manera que se muestre todo el texto sin formato latex o fórmulas. Es decir texto plano para poder entender bien las formulas y eventualmente copiar y pegar en google DOCS.

> [!IMPORTANT]
> **Requisito Obligatorio para abordar las tareas:**
> Siempre tienes que revisar el archivo scientific_agent_skills_reference.md para adoptar la mejor o las mejores skills para abordar cada tarea que se te asigna. 

## Scripts Esenciales del Proyecto

A continuación se detallan los scripts principales de este repositorio, su ubicación y su propósito, para facilitar el desarrollo y la navegación de futuros agentes en el proyecto:

1. **Definición de Arquitectura: [GlobalLocalTransformer_soft_labels.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/GlobalLocalTransformer_soft_labels.py)**
   - **Ubicación:** `scripts/GlobalLocalTransformer_soft_labels.py`
   - **Descripción:** Contiene la definición de la arquitectura de red principal: **Global-Local Transformer (GLT)**, adaptada para soportar **soft labels** (clasificación discreta en 100 bins de edad seguida de un operador Softmax y expectativa soft-argmax para obtener una predicción continua). Soporta backbones de extracción como ResNet-18, ResNet-34, VGG-8, VGG-16 y variantes de EfficientNet.

2. **Preprocesamiento y Armonización: [preprocess_all.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/preprocess/preprocess_all.py)**
   - **Ubicación:** `scripts/preprocess/preprocess_all.py`
   - **Descripción:** Pipeline end-to-end de procesamiento de imágenes NIfTI T1 estructurales. Ejecuta la reorientación estándar, corrección de bias-field (N4) y registro afín al espacio MNI152 (1mm) mediante `brainprep.sh`. Luego aplica la máscara espacial intracraneal fija `SOLID_v2`, realiza normalización robusta de contraste por percentiles P1-P99 reescalando a [0, 1] dentro de la máscara, extrae pilas 2.5D de 5 rebanadas centrales para cada uno de los tres planos (axial, coronal, sagital), y los guarda como tensores de PyTorch (`.pt`).

3. **Entrenamiento del Modelo: [train_improved_ipw.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/train/train_improved_ipw.py)**
   - **Ubicación:** `scripts/train/train_improved_ipw.py`
   - **Descripción:** Script principal de entrenamiento para los modelos de plano único. Implementa técnicas avanzadas como **Fourier Domain Augmentation (FDA)** para robustez frente a site domain shifts, entrenamiento de expectativas de bins mediante pérdidas como divergencia KL o Smooth L1, esquemas de aumento de datos, decaimiento de tasa de aprendizaje por plateau, parada temprana (early stopping), y **Inverse Probability Weighting (IPW)** para reponderar las muestras mitigando los sesgos demográficos de edad y procedencia de datos en bases de datos multi-céntricas.

4. **Búsqueda Combinatoria y Stacking: [find_best_ensemble.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/train/find_best_ensemble.py)**
   - **Ubicación:** `scripts/train/find_best_ensemble.py`
   - **Descripción:** Script encargado de realizar la optimización y fusión de predicciones del ensamble multi-plano. Lee las inferencias en validación de los modelos individuales entrenados en los tres planos anatómicos (axial, coronal, sagital), realiza una búsqueda combinatoria y resuelve la regresión de Ridge (con validación cruzada de 5 pliegues) para encontrar el ensamble óptimo de tres modelos planos que minimice el MAE global y guarde los coeficientes de regresión (betas).

5. **Inferencia y Evaluación: [inference_oasis.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/inference/inference_oasis.py)**
   - **Ubicación:** `scripts/inference/inference_oasis.py`
   - **Descripción:** Sirve como un ejemplo de referencia de extremo a extremo para realizar inferencia sobre una base de datos externa (ej. OASIS-3). Carga los modelos optimizados de cada plano, realiza la inferencia utilizando aumentación en tiempo de test (TTA, en planos axial y coronal), aplica el Ridge Stacker ajustado para obtener la predicción consolidada de edad cerebral, calcula métricas de evaluación (MAE, R2, correlación de Pearson, pendiente del sesgo de edad) y genera gráficos de dispersión y boxplots con calidad de publicación.

6. **Diagramado de Arquitectura Global: [plot_architecture_svg_v2.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/plot_architecture_svg_v2.py)**
   - **Ubicación:** `scripts/manuscript/plot_architecture_svg_v2.py`
   - **Descripción:** Script encargado de generar el diagrama vectorial de arquitectura global `fig01_architecture_svg_v2.svg` y su renderizado en PNG `fig01_architecture_svg_v2.png`. Representa el flujo end-to-end de predicción triplanar, los contenedores Dual-Pathway por plano, la atención cruzada K,V,Q, las cabezas multi-predicción (con pérdidas KL Divergence, Smooth L1 y MSE según IPW v1), el operador Softmax + Soft-argmax, la agregación media (mu) y la fusión tardía mediante el Stacker Ridge.

7. **Diagramado de Backbones de Extracción: [plot_backbone_svg.py](file:///home/nfuentes/scratch/brain_age_project/openBHB_dataset/scripts/manuscript/plot_backbone_svg.py)**
   - **Ubicación:** `scripts/manuscript/plot_backbone_svg.py`
   - **Descripción:** Script encargado de generar las figuras de detalle interno para los extractores de características `fig02_backbone_r18.png` y `fig02_backbone_r34.png`. Representa capa a capa la topología del Stem Block, las 4 etapas convolucionales, los bloques BasicBlock (Conv2D 3x3, BatchNorm, ReLU), los atajos residenciales con convolución 1x1 y las dimensiones y canales asociados.