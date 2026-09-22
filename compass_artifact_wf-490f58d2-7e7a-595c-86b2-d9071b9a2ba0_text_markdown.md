# Plan de la Semana 1 — Tesis: Plataforma No-Code de Inspección Visual Industrial (Ingeniería de Software con IA, Huánuco, Perú)

## TL;DR
- La Semana 1 debe cerrar con siete entregables articulados: borrador completo del plan de tesis, marco teórico inicial con ~15 referencias primarias (con DOI/arXiv), matriz de trazabilidad PEA↔proyecto, lista de compras de hardware con presupuesto en 3 escenarios, protocolo de captura de video v1, esqueleto de repositorio + CI, y una definición de "hecho" con criterios de aceptación.
- La contribución científica (método de deduplicación de frames + partición anti-*leakage* por video/cluster, no aleatoria por frame) está sólidamente respaldada por literatura 2023-2025 (Kapoor & Narayanan 2023; Botache et al. 2024; "Find the Leak, Fix the Split" 2025), lo que valida el diseño; su novedad debe posicionarse en la **aplicación a detección de anomalías industrial no supervisada y a un flujo no-code**.
- Ajustes críticos recomendados frente al plan original: usar **anomalib v2.x (Apache-2.0)** con PatchCore/PaDiM, deduplicar con **imagehash (BSD-2)** e **imagededup (Apache-2.0)** y **descartar fastdup** por su licencia no comercial (CC BY-NC-ND 4.0); **posponer el edge (Jetson/RPi 5) fuera del MVP** por costo y disponibilidad en Perú; y priorizar **iluminación constante y montaje rígido** por encima de una cámara cara.

## Key Findings

1. **Estructura de tesis peruana confirmada**: la secuencia estándar (planteamiento, formulación, objetivos, justificación, hipótesis, operacionalización de variables, matriz de consistencia, metodología, cronograma, presupuesto) es requisito en universidades peruanas (UNI, UNC), y la matriz de consistencia usa habitualmente la clasificación metodológica de Sánchez Carlessi (tipo/nivel/diseño).
2. **Marco teórico de primer nivel disponible**: PatchCore (Roth et al., CVPR 2022), PaDiM (Defard et al., ICPR 2021), MVTec AD (Bergmann et al., CVPR 2019), MVTec AD 2 (Heckler-Kram et al., 2025), anomalib de Intel (Apache-2.0).
3. **El *leakage* por frames de video está documentado y cuantificado**: múltiples papers muestran inflación de métricas cuando frames temporalmente correlacionados se reparten entre train/test — base empírica directa de la contribución de la tesis.
4. **Datos citables para el planteamiento**: el costo de la mala calidad (CoPQ) es de 5-30% de las ventas brutas en manufactura (Quality Digest); las MYPE son el 99.1% de las empresas formales del Perú y las Mipyme emplean al 89.1% del empleo privado nacional (PRODUCE 2024).
5. **Hardware**: la iluminación difusa constante es más determinante que la cámara para la detección de anomalías; la Logitech C920 cuesta ~S/350-400 en Perú y el edge es caro/escaso localmente.
6. **Repos y librerías verificados (2025-2026)**: anomalib activo (v2.x, Python 3.10+, PyTorch 2.0+); imagehash e imagededup con licencias permisivas; fastdup con licencia restrictiva a evitar.

## Details

### Actividad 1 — Plan de Tesis (borrador completo)

**Estructura estándar de un plan de tesis de ingeniería en Perú.** La revisión de fuentes universitarias peruanas (UNI — Facultad de Ingeniería Industrial y de Sistemas; UNC; guías metodológicas nacionales) confirma esta secuencia: (1) Planteamiento y descripción del problema; (2) Formulación del problema (general y específicos); (3) Objetivos (general y específicos); (4) Justificación (teórica, práctica, metodológica, económica/social); (5) Hipótesis (general y específicas); (6) Variables y operacionalización (matriz de operacionalización); (7) Marco teórico y antecedentes; (8) Metodología (tipo, nivel, diseño); (9) Matriz de consistencia; (10) Cronograma; (11) Presupuesto; (12) Referencias. En Perú se emplea comúnmente la clasificación metodológica de **Sánchez Carlessi** (tipo básica/aplicada; nivel descriptivo/correlacional/explicativo/predictivo; diseño experimental/no experimental). La **matriz de consistencia** es requisito obligatorio y su función es verificar la coherencia problema→objetivos→hipótesis→variables→metodología antes de recolectar datos.

**Títulos tentativos propuestos (elegir 1, mantener 2 de respaldo):**
- **Opción A** (describe el artefacto): *"Plataforma no-code de inspección visual industrial basada en detección de anomalías no supervisada para el control de calidad en PYMEs manufactureras"*.
- **Opción B** (expone la contribución científica): *"Método de extracción y deduplicación de frames con partición anti-fuga (anti-leakage) para el entrenamiento de modelos de detección de anomalías visuales a partir de video"*.
- **Opción C** (enfoque de usuario): *"Sistema web de vigilancia de línea de producción mediante visión por computadora entrenable por operarios sin conocimientos de machine learning"*.

**Recomendación**: adoptar la **Opción B** como título académico porque expone la contribución más publicable y defendible; las opciones A/C sirven como descripción del artefacto y del enfoque de usuario respectivamente.

**Planteamiento del problema (borrador con datos citables).** Las MYPE representan el **99.1% (2 326 126 empresas) del total de empresas formales del Perú**, y el segmento Mipyme (99.3%) **empleó a 10.5 millones de trabajadores, equivalente al 89.1% del empleo privado nacional**, generando **S/ 389 819 millones en ventas (22.7% del total formal)** según PRODUCE/OGEIEE, "Las Mipyme en Cifras 2024". Del segmento, el 85.2% se dedica a comercio y servicios y **el 14.8% restante a actividad productiva (manufactura, construcción, agropecuario, minería y pesca)**, y las Mipyme "generan casi el 91% de la PEA ocupada en el sector privado" (PRODUCE, "Las MIPYME en Cifras 2023"). En estas empresas el control de calidad depende mayoritariamente de la inspección visual manual, que introduce inconsistencia, fatiga y costos de mano de obra. El impacto económico es cuantificable: según Quality Digest ("What is Your Company's Cost of Poor Quality?", A. Krishnamoorthi, 2004), *"Experts have estimated that COPQ typically amounts to 5-30 percent of gross sales for manufacturing and service companies"* — es decir, el costo de la mala calidad ronda el **5-30% de las ventas brutas** en manufactura y servicios. Las soluciones de ML/visión de estado del arte requieren datasets etiquetados grandes, ajuste continuo y hardware costoso, lo que crea barreras de adopción especialmente para PYMEs (recursos limitados, falta de expertise técnico). Existe, por tanto, una brecha entre las capacidades de la IA industrial y su accesibilidad para el tejido manufacturero peruano regional (p. ej., Huánuco).

**Problema general.** ¿En qué medida una plataforma no-code de inspección visual, que entrena modelos de detección de anomalías a partir de video grabado por el propio operario, permite detectar defectos en línea de producción con precisión comparable al estado del arte sin requerir conocimientos de ML?

**Problemas específicos.**
- **PE1**: ¿Cómo extraer y deduplicar frames de un video de piezas de modo que la partición train/val/test evite la fuga de información (data leakage) por frames correlacionados?
- **PE2**: ¿Qué diferencia cuantitativa de desempeño existe entre un split anti-leakage (por video/cluster) y un split ingenuo (aleatorio por frame) en MVTec AD y en un dataset propio?
- **PE3**: ¿Qué arquitectura de software permite integrar de forma desacoplada el motor ML, el almacenamiento, la fuente de video y el runtime de inferencia para un MVP mantenible?

**Objetivo general.** Desarrollar y evaluar una plataforma no-code de inspección visual industrial que entrene modelos de detección de anomalías no supervisada a partir de video, incorporando un método de deduplicación y partición anti-leakage.

**Objetivos específicos.**
- **OE1**: Diseñar e implementar el método de extracción/deduplicación de frames con split por video/cluster.
- **OE2**: Comparar cuantitativamente el split anti-leakage vs. el split ingenuo (métricas: AUROC de imagen, AUROC/AUPRO de pixel) en MVTec AD y en un dataset propio, con repetición por múltiples semillas.
- **OE3**: Implementar el MVP (backend FastAPI, pipeline ML con anomalib, frontend TS/React o Next.js) bajo arquitectura de monolito modular con hexagonal selectivo.
- **OE4**: Validar la usabilidad del flujo no-code con un operario sin conocimientos de ML.

**Justificación.** *Técnica*: aporta un método reproducible de partición anti-leakage, evitando métricas infladas (fenómeno documentado por Kapoor & Narayanan, 2023, como el tipo de fuga [L1.4] "duplicados en datasets"). *Económica*: apunta a reducir el CoPQ (5-30% de ventas) y a bajar la barrera de inversión para PYMEs. *Social*: democratiza la IA industrial para el tejido manufacturero regional peruano.

**Hipótesis.**
- **Hi (general)**: La plataforma no-code con split anti-leakage alcanza un desempeño de detección comparable al estado del arte sin requerir expertise de ML.
- **He1**: El split ingenuo por frame produce métricas significativamente infladas respecto del split por video/cluster (diferencia estadísticamente significativa a lo largo de ≥3 semillas).
- **He2**: El método de deduplicación reduce la redundancia del dataset sin degradar la detección respecto de usar todos los frames.

**Metodología sugerida.** Investigación **aplicada/tecnológica**, nivel **explicativo**, con componente **experimental** (OE2: comparación controlada de splits con repetición por semillas). Como marco global recomiendo **Design Science Research (DSR)** de Peffers et al. (2007), cuyas seis actividades —identificación y motivación del problema, definición de objetivos de la solución, diseño y desarrollo, demostración, evaluación y comunicación— encajan naturalmente con una tesis que produce un artefacto de software. Para el subsistema de datos/ML: **CRISP-DM** (metodología del curso "Big Data y Machine Learning" del PEA), citándola explícitamente en las fases de comprensión de datos, preparación, modelado y evaluación. Para el desarrollo del software: **Scrum adaptado a tesis** (sprints semanales alineados a las 16 semanas).

> **Observación crítica del asesor**: combinar DSR + Scrum + CRISP-DM puede parecer eclecticismo y ser objetado por el jurado. Recomiendo posicionar **DSR como paraguas metodológico** (justifica el artefacto y su evaluación rigurosa), **CRISP-DM circunscrito al pipeline de datos/ML**, y **Scrum únicamente como práctica de gestión del proyecto** (no como metodología de investigación). Esta jerarquía es defendible y evita la crítica de mezclar marcos sin justificación.

### Actividad 2 — Marco teórico inicial / referencias científicas

**(a) Métodos de detección de anomalías (memory-bank / one-class).**
- **Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P. (2022).** *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022, pp. 14318-14328. arXiv:2106.08265. DOI: 10.1109/CVPR52688.2022.01392. **Relevancia**: algoritmo central del MVP; según el paper, *"On the challenging, widely used MVTec AD benchmark PatchCore achieves an image-level anomaly detection AUROC score of up to 99.6%, more than halving the error compared to the next best competitor"* (con AUROC de pixel 98.1 y PRO 93.5). Usa un *memory bank* de patches nominales y resuelve el *cold-start* con solo imágenes buenas — exactamente el caso del MVP.
- **Defard, T., Setkov, A., Loesch, A., Audigier, R. (2021).** *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2021 Workshops, pp. 475-489. arXiv:2011.08785. DOI: 10.1007/978-3-030-68799-1_35. **Relevancia**: modelo baseline más liviano (Gaussianas multivariadas por patch, backbone ResNet18); buen candidato para comparación y para edge por su bajo costo computacional.

**(b) Datasets benchmark.**
- **Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. (2019).** *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019, pp. 9592-9600. DOI: 10.1109/CVPR.2019.00982. Licencia de imágenes CC BY-NC-SA 4.0. **Relevancia**: benchmark estándar (15 categorías: objetos y texturas; ~5354 imágenes; 73 tipos de anomalías con ground truth pixel-preciso) para la evaluación comparativa del OE2.
- **Heckler-Kram, L., Neudeck, J-H., Scheler, U., König, R., Steger, C. (2025).** *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*. arXiv:2503.21622; IJCV 134(4), 2026. DOI: 10.48550/arXiv.2503.21622. Licencia CC BY-NC-SA 4.0. **Relevancia**: benchmark no saturado — según el paper, *"their performance remains below 60% average AU-PRO"*; >8000 imágenes de alta resolución con cambios de iluminación (EfficientAD y MSFlow caen >10 pp bajo cambio de iluminación, mientras PatchCore/RD son más robustos ≤3 pp) — muy pertinente para argumentar la importancia de la iluminación constante.

**(c) Librería base del pipeline.**
- **Anomalib (Intel / open-edge-platform).** Repo: github.com/open-edge-platform/anomalib (antes openvinotoolkit/anomalib). Licencia **Apache-2.0** (confirmada en CITATION.cff). **Relevancia**: implementa PatchCore, PaDiM, EfficientAD y otros; exportación a ONNX/OpenVINO; logger MLflow integrado; base directa del pipeline ML del proyecto.

**(d) Data leakage (fundamento de la contribución central).**
- **Kapoor, S., Narayanan, A. (2023).** *Leakage and the Reproducibility Crisis in ML-based Science*. Patterns 4(9), 100804. DOI: 10.1016/j.patter.2023.100804. arXiv:2207.07048. **Relevancia**: presenta una taxonomía de 8 tipos de fuga; la categoría **[L1.4] "duplicates in datasets"** es exactamente lo que aborda la tesis; documenta 294-329 papers afectados por métricas infladas en 17 campos, y propone "model info sheets" como mitigación (aplicable como instrumento metodológico de la tesis).
- **Botache, D., et al. (2024).** *Analyzing Information Leakage on Video Object Detection Datasets by Splitting Images into Clusters with High Spatiotemporal Correlation*. **Relevancia**: algoritmo de split por clusters de frames de video con alta correlación espacio-temporal — antecedente directo del método propuesto.
- **(2025).** *Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets*. arXiv:2511.13944. **Relevancia**: método cluster-based con embeddings (DINO-V3) + HDBSCAN para agrupar frames similares antes del split; estado del arte más reciente y contraste directo para la tesis.

**(e) Perceptual hashing / deduplicación.**
- **Jain, T., Lennan, C., John, Z., Tran, D. (2019).** *imagededup* (idealo). Apache-2.0. **Relevancia**: pHash/dHash/wHash/aHash + CNN (MobileNetV3) para near-duplicates, con framework de evaluación integrado.
- **Buchner, J.** *imagehash*. BSD-2. **Relevancia**: implementación de referencia de aHash/pHash/dHash/wHash/colorhash para la etapa de deduplicación de frames.

**(f) No-code / AutoML para visión industrial y barreras PYME.**
- **A conceptual framework for machine vision integration in manufacturing SMEs** (Discover Artificial Intelligence, Springer, 2026). DOI: 10.1007/s44163-026-01363-4. **Relevancia**: documenta barreras de adopción de machine vision en PYMEs (recursos limitados, brecha de expertise) — justificación del enfoque no-code.

**(g) Surveys recientes de anomalía industrial (2023-2025).**
- **Liu, J., et al. (2023).** *Deep Industrial Image Anomaly Detection: A Survey*. Machine Intelligence Research. DOI: 10.1007/s11633-023-1459-z. **Relevancia**: taxonomía por arquitectura/nivel de supervisión/loss/métricas/datasets; mantiene la lista "awesome-industrial-anomaly-detection".
- **A survey on industrial image anomaly detection: methods, benchmarks and rethinks** (Expert Systems with Applications, vol. 289, 2025, art. 128349). **Relevancia**: revisión de >200 documentos con datasets y métricas actualizadas.

### Actividad 3 — Matriz de trazabilidad PEA↔Proyecto

| Componente del proyecto de tesis | Cursos del PEA que demuestra | Metodología/técnica a citar |
|---|---|---|
| Pipeline ML (anomalib, PatchCore/PaDiM) | Big Data y Machine Learning; Redes Neuronales (Neuralnet, H2O); Prototipado de Aplicaciones de IA (Watson) | **CRISP-DM** (explícito) |
| Preproc. de frames, pHash/embeddings, OpenCV | Big Data y ML; Algoritmia de Programación; Base y Estructura de Datos | CRISP-DM (preparación de datos) |
| Backend FastAPI, arquitectura hexagonal | Ingeniería del Software; Modelado y Diseño del Software; Lenguaje de Programación (N-Capas, Singleton, transacciones, Threads, LINQ) | UML, patrones de diseño, SOLID |
| Diagramas C4/UML (casos de uso, clases, despliegue, componentes, estados, actividad) | Modelado y Diseño del Software; Ingeniería del Software | UML |
| BD PostgreSQL + MinIO | Database Foundation / Design and Programming with SQL (Oracle); Base y Estructura de Datos (Transact SQL, DCL/TCL, procedimientos) | Modelo relacional, normalización |
| Frontend React/Next.js (TypeScript) | Desarrollo de Aplicaciones Web I/II/III (HTML/CSS/JS, PHP, AngularJS, Laravel) | MVC/componentes |
| Dashboard de resultados/métricas | Inteligencia de Negocios y Data Warehouse (dashboards, SQL Server Reporting) | BI, KPIs |
| Redis + worker asíncrono (pipeline) | Lenguaje de Programación (Threads, transacciones); Red Hat System Administration I (Linux) | Colas de trabajo, concurrencia |
| Despliegue Docker / infra / cloud | AZ-900 Fundamentos de Microsoft Azure; Red Hat System Administration I (Linux) | Contenedores, IaaS |
| App móvil de monitoreo (opcional) | Diseño y Desarrollo de Aplicaciones Móviles I/II (MVVM, REST) | MVVM, REST |
| Modelo de dominio (entidades, puertos/adaptadores) | Programación Orientada a Objetos; Java Fundamentals/Foundations (Oracle) | SOLID, DDD selectivo, puertos y adaptadores |
| Modelos predictivos / evaluación experimental | Big Data y ML (R, modelos predictivos); Redes Neuronales | CRISP-DM (evaluación) |
| Prototipado de la plataforma | Software y Prototipado; Realidad Aumentada (visión) | Prototipado iterativo |

**Metodología a citar explícitamente en la tesis**: **CRISP-DM** (curso Big Data y ML) para el subsistema de datos/ML, y las prácticas de **UML/procesos de ingeniería de software** (Modelado y Diseño del Software; Ingeniería del Software) para el diseño del artefacto. Este mapeo demuestra que el proyecto integra y evidencia la mayoría de las competencias del plan de estudios.

### Actividad 4 — Hardware y lista de compras (precios 2025-2026)

**Principio rector** (fundamentado en MVTec AD 2): la **iluminación difusa y constante importa más que la cámara**. Los cambios de iluminación y las sombras móviles son el mayor enemigo de la detección de anomalías no supervisada — MVTec AD 2 demuestra caídas de >10 pp en EfficientAD/MSFlow bajo iluminación variable. Por eso la inversión prioritaria es **iluminación + montaje rígido + fondo neutro**, no una cámara premium.

**(a) Webcam USB 1080p.** Logitech C920/C920s/C922 es el estándar (lente de vidrio de 5 elementos, 1080p/30fps, buena consistencia cromática). En Perú ~**S/350-400 (≈ USD 93-107)** en MercadoLibre Perú (rango observado S/324-500 con descuentos frecuentes). Característica crítica: **poder bloquear el autofoco, la autoexposición y el balance de blancos** (la familia C920, vía UVC/Logitech, lo permite). Lo que importa técnicamente: enfoque fijo, exposición manual, balance de blancos fijo y fps estable — no la resolución máxima.
**(b) Alternativa económica.** Webcam genérica 1080p S/40-150 (≈ USD 11-40); muchas de foco fijo (verificar por listado, pues foco fijo puede ser una ventaja aquí). Alternativa IP: cámara RTSP económica para simular el escenario de línea real (confirmar soporte RTSP y fps estable).
**(c) Iluminación.** Aro de luz LED / panel difuso ~18" con trípode: **S/80-130 (≈ USD 21-35)**. La luz difusa y constante reduce brillos especulares y sombras móviles.
**(d) Soporte/brazo.** Trípode pequeño o brazo articulado/gooseneck: **S/40-120 (≈ USD 11-32)** para una posición cámara-pieza rígida y repetible (un mini-trípode de escritorio puede costar S/25-40).
**(e) Fondo neutro.** Cartulina o tela mate de color contrastante: ~S/10-20.
**(f) Edge (opcional).** Raspberry Pi 5 8GB ~**S/770-795 (≈ USD 205-212)** en Perú (Electromania S/795, The Pi Box S/768; MSRP global ~USD 80, con fuerte markup local ~2.5×). NVIDIA Jetson Orin Nano Super (MSRP global USD 249, 67 TOPS, 25 W) es escaso localmente: ~**S/2 500-3 000 (≈ USD 670-800)** cuando hay stock (Electromania lista S/2 999.90 en lista de espera), o importar (~USD 249 + 30-40% de aranceles/envío).

**Presupuesto (3 escenarios):**
- **Mínimo (~S/200 / USD 55)**: webcam genérica 1080p + aro de luz básico + mini-trípode + fondo. Inferencia en la PC/laptop del tesista.
- **Recomendado (~S/550-650 / USD 150-175)**: Logitech C920 + aro/panel LED de calidad + brazo articulado + fondo mate. Inferencia en PC. **Este es el escenario aconsejado para el MVP.**
- **Con edge (~S/1 400+ / USD 380+)**: recomendado + Raspberry Pi 5 8GB (Jetson elevaría a ~S/3 500+).

> **Recomendación crítica**: para el MVP, ejecutar la inferencia en la PC/laptop con **ONNX Runtime** y **posponer el edge** a una fase posterior. El costo y la disponibilidad del Jetson en Perú no se justifican mientras no exista un requisito de tiempo real o de despliegue on-line en planta demostrado.

### Actividad 5 — Selección de piezas físicas y protocolo de captura v1

**Categorías representativas de MVTec AD.** Los objetos rígidos con defectos de superficie reproducibles funcionan mejor para un MVP que las texturas complejas. Análogos cotidianos/industriales conseguibles en Perú, inspirados en las categorías de MVTec AD (screw, hazelnut, pill, capsule, metal nut, bottle):

**Piezas propuestas (elegir 2-3):**
1. **Tapas de botella metálicas o plásticas** — defectos generables: rayaduras, abolladuras, impresión defectuosa, manchas. Muy baratas y con buena variabilidad inter-pieza.
2. **Tornillos/tuercas** — defectos: rebabas, rosca dañada, óxido. Análogo directo a "screw"/"metal nut" de MVTec.
3. **Empaque impreso / etiqueta** — defectos: impresión corrida, faltante de tinta, manchas. Análogo a defectos de impresión.

**Protocolo de captura de video v1** (fundamentado en buenas prácticas de datasets de anomaly detection y en la evidencia sobre leakage):
- **Resolución**: 1080p (1920×1080). **Fps**: 25-30. **Duración por video**: 60-120 s por pieza en rotación lenta.
- **Distancia cámara-pieza**: fija (p. ej. 20-30 cm), documentada y reproducible; cámara perpendicular a la superficie de inspección.
- **Iluminación**: constante, difusa, sin sombras móviles; **idéntica configuración en todos los videos**.
- **Rotación de la pieza**: cubrir todas las orientaciones relevantes (giro 360° lento + volteo para ambas caras); apuntar a **~8-12 orientaciones distinguibles**.
- **Variabilidad inter-pieza**: grabar **varias piezas físicas distintas** de la misma clase (no una sola) para capturar la variación normal.
- **Fondo**: neutro, mate, contrastante y fijo.
- **Qué NO hacer**: zoom digital, autoexposición, autofoco variable, balance de blancos automático, sombras móviles, cambiar la iluminación entre videos, o mover la cámara entre tomas.
- **Número de videos separados para split sin leakage**: **mínimo 3 videos, cada uno con piezas físicas distintas** → *train* (piezas buenas), *val* (piezas buenas distintas), *test* (piezas buenas + defectuosas distintas). **Regla de oro: ninguna pieza física aparece en más de un split** (split por video/pieza, nunca por frame). Para el MVP no supervisado, el video de entrenamiento contiene **solo piezas BUENAS**; los defectos se introducen manualmente (rayaduras con punzón, manchas con tinta, rebabas) únicamente en las piezas de test. Este diseño es lo que permite el experimento del OE2 (anti-leakage vs. ingenuo).

### Actividad 6 — Repositorio y CI (esqueleto)

**Estructura de monorepo recomendada:**
```
proyecto/
├── backend/                # FastAPI, src layout
│   ├── src/<paquete>/
│   │   ├── domain/         # entidades, value objects, puertos (interfaces)
│   │   ├── application/    # casos de uso, servicios
│   │   ├── adapters/       # ml_engine, storage (PostgreSQL/MinIO), video_source, inference (ONNX)
│   │   └── config/         # DI container, settings
│   ├── tests/
│   └── pyproject.toml
├── frontend/               # React/Next.js + TypeScript
├── ml/                     # experimentos anomalib, notebooks, scripts de split/dedup
│   ├── dedup/              # pHash/embeddings
│   ├── experiments/        # MLflow
│   └── data/               # (gitignored) MVTec AD, dataset propio
├── infra/                  # docker-compose (PostgreSQL, MinIO, Redis, worker)
├── .github/workflows/      # CI
├── .pre-commit-config.yaml
└── README.md
```
Los cuatro puertos del enunciado (motor ML, almacenamiento, fuente de video, runtime de inferencia) se definen como interfaces en `domain/` y se implementan en `adapters/` — hexagonal selectivo aplicado solo donde aporta (integraciones externas), evitando sobre-ingeniería en CRUD simples.

**Repos de referencia (FastAPI + hexagonal en Python), verificados:**
- `szymon6927/hexagonal-architecture-python` (blog.szymonmiks.pl) — ejemplo bien explicado de puertos/adaptadores + inyección de dependencias; el más didáctico.
- `akshanshgusain/hexagonal_architecture_fastapi` — implementación limpia de hexagonal.
- `marcosvs98/hexagonal-architecture-with-python` — FastAPI + DDD (bounded contexts, entidades, value objects).
- Plantillas DDD/CQRS con MinIO en GitHub Topics "hexagonal-architecture" (Python).
- **SQR-072 (LSST)**, "One design pattern for FastAPI web applications" — guía de arquitectura inspirada en hexagonal, con recomendaciones de versión de Python (3.12) y CI.

**CI mínimo (GitHub Actions):** `ruff` (lint + format), `mypy` (tipos), `pytest` (backend), `ESLint` (frontend). **Pre-commit hooks**: ruff, mypy, end-of-file-fixer, trailing-whitespace. **Convenciones**: Conventional Commits + SemVer. Python objetivo: **3.12** (o 3.10+ por compatibilidad con anomalib).

**Estado de anomalib y librerías de deduplicación (verificado 2025-2026):**
- **anomalib**: activo en `open-edge-platform/anomalib`, licencia **Apache-2.0**. Requisitos: **Python 3.10+, PyTorch 2.0+, Lightning 2.2+, OpenVINO 2024.0+**. Serie **v2.x**: v2.0 (beta) introdujo cambios de API respecto de v1.2 (nuevas dataclasses, Preprocessor/Postprocessor, Metrics API); v2.2.0 añadió datasets y **~30% de aceleración en el coreset de PatchCore** con menor uso de memoria en PatchCore/PaDiM. Instalación recomendada con `uv`. *Nota de instalación*: existen cambios de API entre v1.x y v2.x — **fijar la versión en `pyproject.toml`** para reproducibilidad.
- **imagehash** (JohannesBuchner): **v4.3.2 (feb 2025)**, licencia **BSD-2**, activamente mantenido, Python 3.x (deps: numpy, pillow, pywavelets, scipy). Implementa aHash/pHash/dHash/wHash/colorhash. **Recomendado.**
- **imagededup** (idealo): licencia **Apache-2.0**, Python 3.9+, mantenimiento estable pero de baja actividad; pHash/dHash/wHash + CNN con framework de evaluación. **Recomendado** para el módulo de dedup con evaluación.
- **fastdup** (visual-layer): **licencia CC BY-NC-ND 4.0 (no comercial, sin derivados)** — ⚠️ **evitar** si la tesis contempla eventual comercialización o modificación del código; además no soporta Windows de forma nativa (requiere WSL2/Linux) e incluye telemetría opt-out. Usar imagehash/imagededup en su lugar.

### Actividad 7 — Definición de "Hecho" de la Semana 1

**Checklist de entregables con criterios de aceptación (AC):**
- [ ] Borrador del plan de tesis (título elegido + problema/objetivos/hipótesis/justificación/metodología) → **AC**: coherencia verificada en una matriz de consistencia preliminar.
- [ ] Marco teórico inicial con ≥12 referencias primarias con DOI/arXiv → **AC**: cada cita con 1 línea de relevancia.
- [ ] Matriz de trazabilidad PEA↔proyecto completa → **AC**: cada componente mapeado a ≥1 curso.
- [ ] Lista de compras con 3 escenarios de presupuesto → **AC**: cada precio con fuente y fecha.
- [ ] Protocolo de captura v1 → **AC**: especifica mínimo 3 videos y split anti-leakage por pieza.
- [ ] Repo inicial con estructura + CI en verde → **AC**: ruff + mypy + pytest + ESLint corren en Actions; pre-commit instalado.
- [ ] Documento de definición de "hecho" + agenda de reunión → **AC**: revisado y aprobado por el asesor.

**Agenda sugerida para la reunión con el asesor de tesis:**
1. Validar el **alcance del MVP** (¿solo anomalía no supervisada con video de piezas buenas? ¿una sola clase inicialmente?).
2. Validar el **título** y la contribución científica (split anti-leakage) como núcleo de la tesis.
3. Validar la **metodología** (DSR como paraguas + CRISP-DM para datos + Scrum para gestión) y confirmar que el jurado la aceptará.
4. Confirmar la **disponibilidad de piezas** físicas y la capacidad de generar defectos reproducibles.
5. Acordar las **métricas de evaluación** (AUROC de imagen, AUPRO de pixel) y el número de semillas para el OE2.
6. Confirmar el **acceso a MVTec AD** (licencia no comercial CC BY-NC-SA — apta para investigación académica).

## Recommendations
1. **Adoptar la Opción B como título** y centrar la defensa en el método de split anti-leakage: es la contribución más publicable y defendible frente al jurado.
2. **Fijar el stack ML en anomalib v2.x + PatchCore (principal) y PaDiM (baseline liviano)**, con exportación a ONNX. Deduplicar con **imagehash/imagededup**; **descartar fastdup** por su licencia CC BY-NC-ND 4.0.
3. **Priorizar iluminación difusa constante y montaje rígido** sobre la cámara; ejecutar el escenario "Recomendado" (~S/550-650). **Posponer el edge fuera del MVP** (inferencia en PC con ONNX Runtime).
4. **Grabar como mínimo 3 videos con piezas físicas distintas desde la Semana 1** para no bloquear el experimento del OE2; sin este dato el core científico no puede avanzar.
5. **Estructurar la metodología jerárquicamente** (DSR paraguas → CRISP-DM datos → Scrum gestión) para evitar la objeción de eclecticismo metodológico.
6. **Benchmarks que cambiarían el plan**: (a) si la latencia de PatchCore en PC supera ~1-2 s/imagen y el caso real exige tiempo real, reconsiderar OpenVINO/edge; (b) si el dataset propio no alcanza ≥8-10 orientaciones distinguibles por pieza, ampliar la captura antes de modelar; (c) si la deduplicación elimina >40-50% de frames sin degradar métricas, es evidencia fuerte a favor de la contribución.

## Caveats
- Las licencias de **MVTec AD y MVTec AD 2 son CC BY-NC-SA 4.0** (solo investigación no comercial): aptas para la tesis, pero no para un producto comercial sin acuerdo con MVTec.
- Los **precios en Perú son volátiles** (MercadoLibre con descuentos frecuentes); tratar como estimaciones y confirmar al comprar. El precio y la disponibilidad del Jetson son los datos más blandos; el precio de la gooseneck se infirió del rango de categoría.
- **anomalib** tuvo cambios de API entre v1.x y v2.x; fijar la versión evita romper el pipeline a mitad del cronograma.
- El dato **CoPQ 5-30%** proviene de Quality Digest (fuente sectorial internacional, 2004, no oficial peruana); complementar en el planteamiento final con datos de manufactura peruana de INEI/PRODUCE (p. ej., boletines de la Industria Manufacturera de OGEIEE).
- La **contribución "anti-leakage por video" ya tiene antecedentes 2024-2025** (Botache et al.; "Find the Leak, Fix the Split"); la novedad de la tesis debe posicionarse en su **aplicación a la detección de anomalías industrial no supervisada dentro de un flujo no-code entrenable por operarios**, no en el concepto de cluster-split en sí. Conviene declararlo explícitamente para blindar la originalidad.
- Precisión sobre las cifras MYPE/MIPYME: el **99.1%** corresponde a MYPE (2 326 126 empresas) y el **99.3%** a Mipyme (2 331 173) según PRODUCE 2024; usar la etiqueta correcta según la fuente exacta que se cite en el documento final.