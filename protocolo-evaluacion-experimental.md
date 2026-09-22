# Protocolo de Evaluación Experimental

**Tesis:** Método de extracción y deduplicación de frames con partición anti-fuga para el entrenamiento de modelos de detección de anomalías visuales a partir de video
**Autor:** Jersy Claudio Baltazar — Ingeniería de Software con Inteligencia Artificial
**Versión:** 1.1 — 21 de septiembre de 2026 (revisión del diseño de E2 tras el piloto; ver §5/E2)
**Estado:** borrador para revisión del asesor

---

## 1. Propósito de este documento

Este documento fija, **antes de recolectar datos y antes de ejecutar modelos**, el diseño experimental completo de la tesis: qué se mide, sobre qué datos, con cuántas repeticiones, con qué prueba estadística y bajo qué criterios se acepta o se rechaza cada hipótesis.

Se redacta por anticipado de forma deliberada. En una tesis cuyo objeto de estudio es precisamente la **fuga de información (data leakage)**, congelar el protocolo antes de ver los resultados es la única forma de garantizar que las decisiones de análisis no se tomaron a posteriori para favorecer la hipótesis. Esta práctica sigue la recomendación de Kapoor & Narayanan (2023) de documentar el diseño mediante *model info sheets*.

El documento corrige además tres defectos detectados en la formulación original del plan de tesis, señalados en la sección 3.

---

## 2. Reformulación de objetivos e hipótesis

### 2.1 Objetivo específico 2 (reformulado)

> **OE2 (original):** Comparar cuantitativamente el split anti-leakage vs. el split ingenuo (métricas: AUROC de imagen, AUROC/AUPRO de pixel) en MVTec AD y en un dataset propio, con repetición por múltiples semillas.

> **OE2 (reformulado):** Cuantificar la inflación de las métricas de detección atribuible a la partición ingenua frente a la partición por grupo, mediante (a) un experimento controlado de dosis-respuesta sobre MVTec AD con duplicados sintéticos y (b) un experimento sobre un dataset propio derivado de video real con agrupamiento por pieza física, en ambos casos con repetición sobre múltiples particiones aleatorias y prueba estadística pareada.

### 2.2 Hipótesis (reformuladas)

| ID | Enunciado | Contraste | Criterio de rechazo de H₀ |
|----|-----------|-----------|---------------------------|
| **He1** | La partición ingenua por frame produce una estimación de AUROC significativamente superior a la partición por grupo, sobre los mismos datos y el mismo modelo | Wilcoxon de rangos con signo, pareado por semilla de partición | p < 0.05 **y** Δ AUROC ≥ 2 pp (relevancia práctica) |
| **He1b** | La magnitud de la inflación crece monótonamente con la dosis de fuga | Correlación de Spearman entre λ y Δ AUROC | ρ > 0.7 con p < 0.05 |
| **He2** | La deduplicación reduce el volumen del conjunto de entrenamiento sin degradar significativamente la detección, y reduce proporcionalmente el costo computacional | Equivalencia: IC95 de la diferencia de AUROC contenido en [−2 pp, +2 pp] | AUROC no degradado **y** reducción ≥ 40% de frames **y** reducción medible de RAM pico y tiempo de entrenamiento |

**Nota sobre He2.** Se formula como prueba de **equivalencia**, no de diferencia. Afirmar "no degrada" a partir de un p-valor no significativo es un error estadístico frecuente (ausencia de evidencia ≠ evidencia de ausencia). Se declara un margen de equivalencia de ±2 pp de AUROC y se comprueba que el intervalo de confianza de la diferencia queda contenido en él.

---

## 3. Defectos corregidos respecto del plan original

### 3.1 El experimento de fuga no es ejecutable sobre MVTec AD

**Problema.** MVTec AD es un conjunto de **imágenes fijas** con una partición train/test definida y fijada por sus autores. No contiene video ni frames temporalmente correlacionados. Por construcción, no existe en él la partición ingenua por frame que la hipótesis He1 pretende evaluar. El objetivo OE2, tal como estaba redactado, no era ejecutable sobre ese conjunto.

**Consecuencia no mitigada.** La contribución científica quedaba dependiendo íntegramente del dataset propio, cuya captura depende de hardware aún no adquirido. Un fallo en la captura en la semana 4 dejaría la tesis sin evidencia.

**Corrección adoptada.** Se separa el rol de cada conjunto:

| Conjunto | Rol | Depende de hardware |
|----------|-----|---------------------|
| MVTec AD | (a) Validación de la corrección de la integración con anomalib (E1); (b) experimento controlado de fuga con duplicados sintéticos (E2) | No |
| Dataset propio en video | Experimento principal de fuga con frames reales y agrupamiento por pieza física (E3); ablación de deduplicación (E4) | Sí |

El experimento E2 permite producir evidencia publicable **sin cámara y sin piezas**, eliminando el hardware del camino crítico de la contribución científica.

### 3.2 El número de grupos previsto impide toda inferencia estadística

**Problema.** El protocolo de captura v1 especifica "mínimo 3 videos, cada uno con piezas físicas distintas". Si la unidad de agrupamiento es el video, el experimento dispone de **3 grupos**. Con 3 grupos no existen particiones por grupo suficientemente distintas para generar réplicas independientes, y ninguna prueba estadística es aplicable.

**Corrección adoptada.** La unidad de agrupamiento pasa a ser la **pieza física individual**, y el protocolo de captura se modifica para maximizar el número de piezas, no la duración del video. Ver sección 8.

### 3.3 Las métricas de píxel sobre el dataset propio no estaban presupuestadas

**Problema.** AUROC de píxel y AUPRO exigen **máscaras de segmentación anotadas manualmente** sobre cada frame defectuoso. El cronograma no asigna tiempo a esa anotación.

**Corrección adoptada.** Se declara explícitamente:

- **Dataset propio:** métrica principal AUROC a nivel de **imagen** (requiere únicamente una etiqueta binaria por pieza, que se propaga a sus frames).
- **MVTec AD:** métricas de píxel (AUROC-px, AUPRO) además de imagen, aprovechando el *ground truth* que el conjunto ya incluye.
- La ausencia de métricas de localización sobre el dataset propio se consigna como limitación declarada en la sección 12, no como omisión.

---

## 4. Conjuntos de datos

### 4.1 D1 — MVTec AD

- **Fuente:** Bergmann et al., CVPR 2019. Licencia CC BY-NC-SA 4.0 (uso académico no comercial, compatible con esta tesis).
- **Subconjunto empleado:** 3 categorías de objeto: `bottle`, `screw`, `metal_nut`. Se eligen categorías de **objeto rígido** y no de textura por su analogía directa con las piezas del dataset propio (tapas, tornillos/tuercas).
- **Uso:** experimentos E1 y E2.

### 4.2 D2 — Dataset propio en video
   
- **Piezas:** 2 clases de objeto (tapas plásticas/metálicas y tuercas), capturadas según el protocolo de la sección 8.
- **Volumen objetivo por clase:** ≥ 30 piezas físicas sin defecto y ≥ 15 piezas con defecto inducido.
- **Uso:** experimentos E3 y E4.
- **Definición de grupo:** la pieza física individual. Cada pieza recibe un identificador único que acompaña a todos sus frames a lo largo de todo el pipeline.

---

## 5. Experimentos

### E1 — Validación de la integración (control de calidad, no prueba de hipótesis)

**Propósito.** Verificar que la integración con anomalib reproduce resultados publicados antes de extraer cualquier conclusión. Sin este control, un error de implementación sería indistinguible de un hallazgo.

**Procedimiento.** Entrenar PatchCore y PaDiM sobre las 3 categorías de D1, **usando la partición oficial de los autores sin modificarla**.

**Configuración (corregida en v1.1).** Cada modelo se ejecuta con la configuración del paper contra el que se compara, no con el valor por defecto de la librería:

| Modelo | Configuración | Referencia |
|---|---|---|
| PatchCore | `wide_resnet50_2`, capas (layer2, layer3), coreset **1%**, 256×256 | Roth et al. 2022, Tabla S1, columna PatchCore-1% |
| PaDiM | `wide_resnet50_2`, capas (layer1, layer2, layer3), **550** dimensiones | Defard et al. 2021, columna PaDiM-WR50-Rd550 |

Esto **no** es el valor por defecto de anomalib, que usa `resnet18` para PaDiM. La primera corrida de E1 lo hizo y obtuvo 0,8268 de AUROC en `screw` frente a los 0,975 publicados: PaDiM es muy sensible al backbone. Una cifra publicada solo es comparable contra una corrida de la misma configuración, y por eso la configuración viaja junto al número en el archivo de referencias.

El ratio de coreset importa por el mismo motivo: el paper reporta 96,4 en `screw` con PatchCore-1% y 98,1 con PatchCore-25%, casi 2 pp de diferencia en la misma categoría.

**Criterio de aceptación.** AUROC de imagen dentro de ±2 pp del valor publicado para **esa misma configuración**. Si algún valor queda fuera del margen, el pipeline contiene un defecto y **se detiene el avance experimental** hasta corregirlo.

**Lección del primer intento.** Se usó como criterio de respaldo un «suelo» único por modelo (PatchCore ≥ 0,97) para todas las categorías de objeto. Marcó `screw` como FALLA con 0,9672 — y era un falso positivo: el paper reporta 96,4 para esa configuración. Un umbral único no captura que la dificultad varía mucho entre categorías. El suelo se conserva solo como red de seguridad para cuando no existe referencia verificada; **el veredicto válido es el que se emite contra la referencia por categoría**.

**Resultado obtenido (21 de septiembre de 2026).**

| modelo | categoría | AUROC obtenido | referencia (paper) | Δ | veredicto |
|---|---|---|---|---|---|
| PatchCore-1% | bottle | 1,0000 | 1,000 | **+0,00 pp** | PASA |
| PatchCore-1% | screw | 0,9672 | 0,964 | **+0,32 pp** | PASA |
| PatchCore-1% | metal_nut | 0,9990 | 0,997 | **+0,20 pp** | PASA |
| PaDiM-WR50-Rd550 | bottle | 1,0000 | 0,983 | +1,70 pp | PASA |
| PaDiM-WR50-Rd550 | screw | — | 0,975 | — | **no ejecutable** |
| PaDiM-WR50-Rd550 | metal_nut | — | 0,972 | — | **no ejecutable** |

**E1 queda validado por PatchCore**: tres reproducciones dentro de ±0,32 pp del valor publicado para la misma configuración. Como la instrumentación —carga de datos, partición, emparejamiento de scores, cálculo del AUROC— es idéntica para ambos modelos, esas tres celdas validan el pipeline completo.

**Limitación de hardware declarada.** PaDiM-WR50-Rd550 no es ejecutable en el equipo de desarrollo (15,3 GB de RAM, 6 núcleos, sin GPU). La configuración mantiene una matriz de covarianza de 550×550 por cada posición espacial: a 256×256 con `layer1` son 4096 posiciones, es decir ≈ 4,96 GB solo en covarianzas, más los *embeddings* acumulados. La categoría `bottle` (209 imágenes de entrenamiento) alcanzó 9,7 GB de pico y 494 s; `screw` (320 imágenes) agotó la memoria y el proceso terminó. Se consigna como límite del equipo, no como defecto del método.

**Consecuencia para el resto de los experimentos.** PaDiM se usa en E2–E4 con `resnet18`, que es además su papel real en la tesis: el *baseline liviano*. Esto es metodológicamente correcto porque E2–E4 **no comparan contra ninguna cifra publicada**: contrastan dos brazos entrenados con el mismo modelo y la misma configuración, de modo que el valor absoluto del AUROC es irrelevante y solo importa la diferencia entre brazos.

---

### E2 — Curva dosis-respuesta de fuga (duplicados sintéticos sobre D1)

> **Revisión v1.1 (21 de septiembre de 2026), tras el experimento piloto.** El diseño original de esta sección —barrer el número de duplicados *k* con dos particiones construidas de forma independiente— resultó **no interpretable**, por dos motivos que el piloto hizo visibles y que se documentan aquí porque afectan a la lectura de cualquier resultado previo:
>
> 1. **Varianza entre particiones.** A *k*=0, donde por definición no hay ningún duplicado, el piloto midió Δ = +3,62 pp. No era fuga: eran dos particiones distintas del mismo conjunto, ambas válidas. PaDiM sobre `bottle` osciló entre 0,86 y 0,98 de AUROC según qué imágenes cayeran en entrenamiento. Esa varianza (~5 pp) era **mayor que el efecto buscado**. La afirmación de que a *k*=0 ambas estrategias son «equivalentes por construcción» era incorrecta.
> 2. **Confundido de tamaño de muestra.** Con partición aleatoria por frame, el brazo ingenuo entrenaba con imágenes procedentes de casi todos los originales, mientras que el brazo por grupo solo veía los de su 60%. El ingenuo disponía de más información, y eso elevaba su AUROC al margen de cualquier fuga. Es exactamente la amenaza declarada en la sección 12, materializada.
>
> **Diseño revisado: brazos emparejados y dosis directa.** La variable de dosis pasa a ser **λ**, la fracción del conjunto de entrenamiento que son casi-duplicados de piezas presentes en test. λ describe el mecanismo directamente; *k* era un proxy indirecto y queda relegado a «reserva de casi-duplicados disponibles».
>
> | | diseño v1.0 | diseño v1.1 |
> |---|---|---|
> | Conjunto de test | distinto por brazo y por *k* | **idéntico** en ambos brazos y en todas las dosis |
> | Tamaño de entrenamiento | igual, pero de distintos orígenes | igual; el brazo limpio es además **el mismo** para todas las dosis |
> | Control a dosis cero | Δ = +3,62 pp (varianza) | **Δ = 0,00 exacto**: ambos brazos son el mismo entrenamiento |
> | Única diferencia | *k*, y de rebote el nº de originales vistos | **solo λ** |
>
> Fijar el conjunto de test elimina la mayor fuente de varianza y hace los AUROC directamente comparables. Como efecto secundario, reutilizar el brazo limpio entre dosis reduce el coste de 70 a 40 entrenamientos.
>
> El procedimiento original se conserva abajo como registro de lo que se planeó y por qué se descartó.

**Propósito.** Cuantificar de forma controlada y reproducible cómo crece la inflación de la métrica en función de la tasa de duplicados, sin depender de la captura propia.

**Fundamento del diseño.** En detección de anomalías no supervisada, el entrenamiento usa exclusivamente muestras normales. La fuga relevante se produce cuando una muestra **casi idéntica** a una de entrenamiento aparece en el conjunto normal de test: el modelo ha memorizado literalmente esa muestra, produce menos falsos positivos de los que produciría sobre material no visto, y el AUROC se infla. Este es exactamente el mecanismo que genera un video a 30 fps.

**Procedimiento.**

1. Tomar el conjunto de imágenes normales de una categoría de D1 (unión de `train/good` y `test/good`). Cada imagen original *i* constituye un **grupo**.
2. Generar *k* duplicados sintéticos por original mediante transformaciones que emulan frames consecutivos de un video: traslación ≤ 2% del lado, rotación ≤ 2°, escala ±2%, variación de brillo/contraste ±3%, ruido gaussiano de baja varianza. **No se aplican volteos ni rotaciones amplias**: eso sería *data augmentation*, no duplicación, y cambiaría el fenómeno bajo estudio.
3. Barrer *k* ∈ {0, 1, 3, 7, 15}, equivalente a tasas de redundancia de 0%, 50%, 75%, 87.5% y 93.75%.
4. Para cada valor de *k*, construir dos particiones del conjunto expandido de normales:
   - **S_ingenuo:** asignación aleatoria a nivel de **imagen individual**. Los duplicados de un mismo original se reparten entre train y test.
   - **S_grupo:** asignación aleatoria a nivel de **grupo**. Todos los duplicados de un original caen del mismo lado.
   Ambas particiones respetan las mismas proporciones (60% train / 20% val / 20% test).
5. Las imágenes defectuosas de la categoría se añaden íntegramente al conjunto de test en ambos esquemas, sin duplicar (no participan del entrenamiento y por tanto no son vía de fuga en este diseño).
6. Entrenar PatchCore y evaluar.

**Réplicas.** 10 particiones aleatorias independientes por cada combinación (categoría × *k* × esquema). Semillas 0–9, registradas.

**Métrica de contraste.** Δ AUROC(*k*) = AUROC(S_ingenuo) − AUROC(S_grupo), pareado por semilla.

**Condición de refutación.** Si Δ AUROC ≈ 0 (IC95 contenido en [−2 pp, +2 pp]) para todos los valores de *k*, **He1 y He1b quedan refutadas** y el hallazgo se reporta como tal. Este resultado sería igualmente publicable y debe declararse de antemano para que el experimento sea genuinamente falsable.

**Resultado primario (diseño v1.1, n = 10, registrado el 22 de septiembre de 2026 antes de ejecutar la réplica extendida).** Categoría `bottle`, PaDiM con `resnet18`, reserva de *k* = 3 casi-duplicados, 36,3 min.

| λ | frames filtrados | AUROC limpio | AUROC filtrado | Δ (pp) | IC95 (pp) | p crudo | p Holm | réplicas con Δ>0 |
|---|---|---|---|---|---|---|---|---|
| 0 % | 0 | 0,9629 | 0,9629 | **0,00** | [0,00; 0,00] | — | — | control |
| 5 % | 27 | 0,9629 | 0,9587 | −0,42 | [−2,47; +1,63] | 0,844 | 0,844 | 3/10 |
| 10 % | 55 | 0,9629 | 0,9799 | +1,71 | [+0,02; +3,31] | 0,129 | 0,258 | 7/10 |
| 20 % | 110 | 0,9629 | 0,9912 | **+2,83** | **[+1,00; +4,45]** | **0,018** | 0,053 | 8/10 |

Spearman λ–Δ: ρ = +0,800, p = 0,200.

**Lectura.** El control a dosis cero da Δ = 0 exacto, como exige el diseño. Hay una respuesta creciente con la dosis y, a λ = 20 %, el IC95 excluye el cero y el efecto supera el umbral preregistrado de 2 pp. Sin embargo, tras la corrección de Holm-Bonferroni el p-valor queda en 0,053, por encima de α = 0,05: **con n = 10 no se rechaza H₀ para He1**. El experimento está subpotenciado, no es negativo, y así se reporta.

**Defecto estructural detectado en este diseño.** Con 4 niveles de dosis, la correlación de Spearman tiene un p-valor mínimo alcanzable de 0,083 aun con monotonía perfecta (ρ = 1). **He1b no podía rechazarse en ningún escenario**, con independencia de los datos. Es un error de diseño del mismo tipo que el de las 3 réplicas del piloto, y existe al margen del resultado observado.

**Réplica extendida (declarada, no preregistrada).** Para corregir ese defecto se ejecuta una réplica con **6 niveles de dosis** (0, 5, 10, 15, 20 y 25 %) — con n = 6 y ρ = 1, Spearman alcanza p ≈ 0,003 — y **20 réplicas**, para disponer de margen frente a la corrección de Holm sobre 5 contrastes. Se reconoce explícitamente que ampliar las réplicas tras observar un p-valor limítrofe es una decisión *post hoc*. Por eso: (1) el resultado de n = 10 de la tabla anterior se mantiene como **resultado primario** y se reporta tal cual; (2) la réplica se presenta como análisis secundario, motivado por el defecto de He1b; (3) ambas tablas figuran en la tesis.

---

### E3 — Experimento principal: partición por pieza sobre video real (D2)

**Propósito.** Verificar sobre frames reales, con la correlación temporal genuina de un video, la inflación medida sintéticamente en E2.

**Procedimiento.**

1. Cada pieza física constituye un grupo, identificado desde la captura.
2. Construir dos particiones sobre el mismo conjunto de frames:
   - **S_ingenuo:** asignación aleatoria a nivel de **frame**.
   - **S_pieza:** asignación aleatoria a nivel de **pieza física**. Ninguna pieza aparece en más de un split.
3. Proporciones: piezas normales 60/20/20 (train/val/test); piezas defectuosas repartidas entre val (≈30%) y test (≈70%). Las piezas defectuosas nunca entran a train: el modo es no supervisado.
4. Entrenar PatchCore con la configuración validada en E1.

**Réplicas.** 10 particiones aleatorias independientes, pareadas por semilla entre ambos esquemas.

**Métrica de contraste.** Δ AUROC de imagen, con IC95 y tamaño del efecto.

---

### E4 — Ablación de la deduplicación (He2)

**Propósito.** Determinar el punto de operación de la deduplicación: cuántos frames pueden eliminarse sin degradar la detección, y qué se gana computacionalmente al hacerlo.

**Fundamento adicional.** La deduplicación no es únicamente una medida de higiene metodológica. El *memory bank* de PatchCore escala con el número de muestras nominales de entrenamiento: un video de 90 s a 30 fps aporta ~2 700 frames casi idénticos que multiplican el consumo de memoria y el tiempo de construcción del banco sin aportar información. La ablación mide simultáneamente el efecto sobre la métrica y sobre el costo, y es lo que convierte a He2 en un resultado de ingeniería y no en un mero resultado negativo.

**Procedimiento.** Con la partición **S_pieza fija** (para aislar el efecto de la deduplicación de cualquier efecto de partición), barrer el umbral de distancia de Hamming sobre pHash de 64 bits: *d* ∈ {0, 2, 4, 6, 8, 10, 12}. Repetir con la variante basada en embeddings + agrupamiento.

**Variables medidas en cada punto del barrido:**

| Variable | Unidad |
|----------|--------|
| Frames retenidos | n y % del total |
| AUROC de imagen | [0,1], media ± IC95 sobre 10 réplicas |
| Tiempo de entrenamiento | s |
| Memoria pico durante el entrenamiento | MB |
| Tamaño del memory bank | n de parches |
| Latencia de inferencia p50 / p95 | ms |

**Criterio de éxito.** Existe un *d* tal que se retiene ≤ 60% de los frames (≥ 40% eliminados) con AUROC cuyo IC95 de diferencia respecto del baseline sin deduplicación queda contenido en [−2 pp, +2 pp], y con reducción medible de memoria y tiempo. Ese *d* se adopta como valor por defecto del producto.

---

## 6. Métricas

### 6.1 Métrica principal

**AUROC a nivel de imagen.** Se adopta como métrica principal por tres razones: (a) es independiente del umbral de decisión, de modo que separa la calidad del modelo de la calibración operativa; (b) es la métrica estándar de la literatura de detección de anomalías industrial, lo que hace los resultados comparables; (c) requiere únicamente una etiqueta binaria por muestra, sin anotación de máscaras.

### 6.2 Métricas secundarias

- **Solo sobre D1 (MVTec AD):** AUROC a nivel de píxel y AUPRO, aprovechando el *ground truth* incluido.
- **Operativas, en el umbral calibrado (sección 7):** F1, **recall de defectos** (la métrica que importa al usuario industrial: un defecto no detectado llega al cliente), tasa de falsas alarmas.
- **De sistema:** FPS de extremo a extremo, latencia p50/p95, memoria pico. Requeridas por RNF-01.

### 6.3 Fuera de alcance declarado

Métricas de localización (píxel, AUPRO) sobre el dataset propio, por ausencia de máscaras anotadas. Consignado como limitación en la sección 12.

---

## 7. Protocolo de calibración del umbral

Esta sección es crítica para la validez de la tesis. Seleccionar el umbral de decisión que maximiza F1 **observando el conjunto de test** constituye un ajuste de parámetro sobre el test, es decir, una fuga de información por la puerta de atrás. En una tesis sobre fuga de información, incurrir en ella sería un defecto fatal.

**Regla.**

1. Se mantienen **tres** conjuntos disjuntos por pieza: `train` (solo piezas normales), `val` (piezas normales + una fracción de las defectuosas), `test` (piezas normales + defectuosas, todas distintas de las anteriores).
2. El umbral se selecciona como el que maximiza F1 **sobre `val`**, y queda congelado.
3. Ese umbral congelado se aplica a `test` sin reajuste alguno.
4. La selección de modelo, backbone, tasa de coreset y umbral de deduplicación se realiza **exclusivamente sobre `val`**.
5. `test` se evalúa **una sola vez por configuración final**. El número de evaluaciones realizadas sobre `test` se registra y se reporta.

**Verificación explícita.** El código de evaluación incluye una aserción que falla si cualquier estadístico de calibración se computa sobre particiones marcadas como `test`.

---

## 8. Modificaciones al protocolo de captura v1

El protocolo de captura vigente especifica "mínimo 3 videos con piezas distintas". Como se argumentó en 3.2, ese volumen impide toda inferencia estadística. Se sustituye por:

| Parámetro | Valor v1 | **Valor v2 (este protocolo)** | Justificación |
|-----------|----------|-------------------------------|---------------|
| Unidad de agrupamiento | Video | **Pieza física individual** | Es el nivel en el que ocurre la correlación que causa la fuga |
| Nº de piezas normales por clase | No especificado (≈3) | **≥ 30** | Habilita ≥ 10 réplicas de partición por grupo |
| Nº de piezas con defecto por clase | No especificado | **≥ 15** | Suficientes positivos en val y test |
| Duración por pieza | 60–120 s | **10–20 s** | Tras deduplicación, 15 s cubren las orientaciones útiles; el exceso es redundancia pura |
| Nº de clases de objeto | 2–3 | **2** | Concentrar el esfuerzo de captura en nº de piezas, no en nº de clases |
| Identificación | No especificada | **ID único por pieza, registrado en el nombre del clip** | La trazabilidad del grupo debe existir desde la captura, no reconstruirse después |

**Volumen resultante estimado por clase:** 45 piezas × 15 s × 30 fps ≈ 20 000 frames brutos → tras deduplicación, del orden de 1 500–2 500 frames útiles. Volumen adecuado para PatchCore y manejable en el hardware previsto.

El resto del protocolo v1 (iluminación difusa constante, enfoque/exposición/balance de blancos fijos, fondo neutro, cámara perpendicular y rígida, prohibición de zoom digital) se mantiene sin cambios y sigue siendo el factor de calidad dominante.

---

## 9. Diseño estadístico

| Elemento | Decisión | Justificación |
|----------|----------|---------------|
| Unidad de réplica | La **partición aleatoria**, no la semilla del modelo | PatchCore es prácticamente determinista salvo el submuestreo del coreset; la variabilidad relevante está en cómo se reparten los datos |
| Nº de réplicas | 10 por condición | Compromiso entre potencia y costo; suficiente para efectos de ≥ 2 pp, insuficiente para efectos pequeños (declarado en 12) |
| Emparejamiento | Sí: misma semilla genera S_ingenuo y S_grupo sobre los mismos datos | Elimina la varianza entre réplicas y aumenta la potencia |
| Prueba | **Wilcoxon de rangos con signo** (pareada, no paramétrica) | No asume normalidad; apropiada para n = 10 |
| Nivel de significancia | α = 0.05 | Convención |
| Corrección por comparaciones múltiples | **Holm–Bonferroni** sobre el conjunto de categorías/valores de *k* | Se realizan múltiples contrastes; sin corrección, la probabilidad de falso positivo se acumula |
| Tamaño del efecto | **Delta de Cliff** | Se reporta siempre junto al p-valor. Un p < 0.05 sobre un Δ de 0.3 pp es estadísticamente detectable y prácticamente irrelevante |
| Umbral de relevancia práctica | **Δ AUROC ≥ 2 pp**, preregistrado | Separa significancia estadística de importancia real |
| Reporte | Media, IC95, p-valor corregido y tamaño del efecto, en todos los casos | |

---

## 10. Controles anti-fuga (lista de verificación obligatoria)

Se ejecuta antes de cada corrida experimental y su resultado se archiva junto a los resultados.

- [ ] Ninguna pieza física / grupo aparece en más de un split.
- [ ] **Orden de operaciones correcto: agrupar → particionar por grupo → deduplicar dentro de cada split.** Deduplicar globalmente *antes* de particionar eliminaría del test los frames que duplican a los de train, ocultando la fuga en lugar de corregirla, y alteraría el fenómeno bajo medición.
- [ ] Ningún estadístico de normalización, calibración o umbral se computa sobre `test`.
- [ ] La selección de hiperparámetros se realizó sobre `val`.
- [ ] Las semillas están registradas y los manifiestos de partición (lista explícita de archivos por split) están serializados en disco junto al resultado.
- [ ] El número de evaluaciones efectuadas sobre `test` está registrado.
- [ ] En E2, los duplicados sintéticos de un grupo nunca cruzan el límite del split en el esquema S_grupo (verificado por aserción automática).

---

## 11. Reproducibilidad

- **Versiones fijadas** de anomalib, PyTorch y dependencias en `pyproject.toml`. La serie v2.x de anomalib introdujo cambios de API respecto de v1.x; el anclaje de versión es obligatorio.
- **Manifiestos de partición serializados** en JSON: cada split se archiva como lista explícita de rutas de archivo con su ID de grupo, no como una semilla que haya que volver a ejecutar.
- **Registro en MLflow:** un *run* por cada (experimento, categoría, esquema, réplica), con parámetros, métricas, manifiesto de partición y hash del commit.
- **Hardware declarado** en cada corrida: CPU, GPU, RAM, versiones de driver.
- Toda cifra que aparezca en la tesis debe ser regenerable ejecutando un único script sobre los manifiestos archivados.

---

## 12. Amenazas a la validez

| Tipo | Amenaza | Mitigación |
|------|---------|------------|
| **Interna** | Los duplicados sintéticos de E2 no reproducen exactamente la estructura de frames reales de video | E3 valida el mismo fenómeno sobre video real; E2 aporta control y dosis-respuesta, E3 aporta realismo |
| **Interna** | Un defecto de implementación podría confundirse con un hallazgo | E1 actúa como control de calidad previo y bloqueante |
| **Externa** | Una única configuración de iluminación y un único entorno; los resultados no se extrapolan a una planta real | Declarado como limitación. MVTec AD 2 documenta caídas > 10 pp bajo cambio de iluminación en varios métodos |
| **Externa** | Dos clases de objeto, elegidas por disponibilidad y no por muestreo representativo de la manufactura | Declarado. Las categorías se eligen por analogía con MVTec AD para facilitar la comparación |
| **De constructo** | El AUROC de imagen no evalúa la localización del defecto | Métricas de píxel reportadas sobre MVTec AD; limitación declarada para el dataset propio |
| **De constructo** | Los defectos son inducidos manualmente, no generados por el proceso productivo real | Declarado. Se documenta el procedimiento de inducción para que sea reproducible |
| **De conclusión** | n = 10 réplicas limita la potencia para detectar efectos < 2 pp | El umbral de relevancia práctica se fija en 2 pp precisamente por esta razón |
| **De conclusión** | La partición por grupo reduce el conjunto efectivo de entrenamiento respecto de la ingenua, lo que por sí solo puede bajar el AUROC | **Resuelto en E2 por el diseño emparejado v1.1**: ambos brazos entrenan con el mismo número de imágenes y evalúan sobre el mismo conjunto de test. En E3 (video real) sigue vigente y se controla igualando el nº de **grupos** de entrenamiento, reportando ambos volúmenes |

La última fila era la objeción más sólida que un jurado podía plantear contra He1, y el piloto confirmó que era real y no teórica: con el diseño original el brazo ingenuo veía más originales distintos en entrenamiento, de modo que parte del Δ no era fuga sino tamaño de muestra. El rediseño de E2 la elimina por construcción. En E3 no puede eliminarse igual —los frames reales no se pueden reasignar sin romper el escenario que se quiere medir—, así que allí se controla igualando grupos y se reportan ambos volúmenes para que la comparación sea auditable.

---

## 13. Resumen de experimentos

| ID | Datos | Compara | Réplicas | Métrica de contraste | Hipótesis | Depende de hardware |
|----|-------|---------|----------|----------------------|-----------|---------------------|
| **E1** | MVTec AD, split oficial | Resultados propios vs. publicados | — | AUROC imagen, ±2 pp | Ninguna (control) | No |
| **E2** | MVTec AD + duplicados sintéticos | Brazos emparejados, λ ∈ {0, 5%, 10%, 20%} | 10 × categoría × λ | Δ AUROC(λ) | He1, He1b | No |
| **E3** | Dataset propio en video | S_ingenuo (frame) vs. S_pieza | 10 × clase | Δ AUROC | He1 | Sí |
| **E4** | Dataset propio, S_pieza fija | Barrido de umbral de deduplicación | 10 × punto | AUROC, % retenido, RAM, tiempo | He2 | Sí |

---

## 14. Puntos a validar con el asesor

1. ¿Acepta el jurado el experimento E2 con duplicados **sintéticos** como evidencia complementaria, o exige que toda la evidencia provenga de video real?
2. ¿Es aceptable renunciar a las métricas de píxel sobre el dataset propio, dado el costo de anotación de máscaras? (Sección 3.3.)
3. ¿Es viable conseguir ≥ 45 piezas físicas por clase? (Sección 8.) De no serlo, hay que replantear la potencia estadística antes de capturar.
4. ¿Se acepta un margen de equivalencia de ±2 pp de AUROC como umbral de relevancia práctica? (Secciones 2.2 y 9.)
5. ¿Debe preinscribirse este protocolo formalmente ante el asesor con fecha, para dejar constancia de que precede a la recolección de datos?
6. **(Nuevo en v1.1)** ¿Acepta el jurado que E2 mida la dosis como λ —fracción del entrenamiento que reaparece en test— en lugar del número de duplicados *k*? El cambio se motiva en el piloto documentado arriba, no en los resultados: se decidió tras observar que el diseño original no era interpretable, y **antes** de medir ningún efecto con el diseño nuevo. La trazabilidad de esa secuencia está en el historial del repositorio.

---

## 15. Referencias del diseño experimental

- Kapoor, S., Narayanan, A. (2023). *Leakage and the Reproducibility Crisis in ML-based Science*. Patterns 4(9), 100804. DOI: 10.1016/j.patter.2023.100804 — taxonomía de fugas; la categoría [L1.4] "duplicados en datasets" es la abordada aquí. Origen de la práctica de *model info sheets* adoptada en este documento.
- Botache, D. et al. (2024). *Analyzing Information Leakage on Video Object Detection Datasets by Splitting Images into Clusters with High Spatiotemporal Correlation* — antecedente directo del agrupamiento por correlación espacio-temporal.
- *Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets* (2025). arXiv:2511.13944 — método de agrupamiento con embeddings previo a la partición; contraste directo.
- Roth, K. et al. (2022). *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022. arXiv:2106.08265 — modelo principal; referencia de las cifras de E1.
- Defard, T. et al. (2021). *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2021 Workshops. arXiv:2011.08785 — baseline liviano.
- Bergmann, P. et al. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019. DOI: 10.1109/CVPR.2019.00982 — conjunto D1.
- Heckler-Kram, L. et al. (2025). *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*. arXiv:2503.21622 — evidencia sobre sensibilidad a la iluminación, base de la amenaza de validez externa.
