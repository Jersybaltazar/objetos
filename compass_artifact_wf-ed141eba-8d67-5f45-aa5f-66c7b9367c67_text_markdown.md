# Evaluación de Arquitecto Senior: Plataforma No-Code de Inspección Visual Industrial con Entrenamiento "Un Video por Clase"

## TL;DR
- **La idea es sólida y comercialmente relevante, pero NO es novedosa en su núcleo**: el mercado de inspección visual no-code está saturado (LandingLens, Cognex, Roboflow, Zebra Aurora, Matroid, Edge Impulse), y el mecanismo de "grabar por clase con la cámara" ya existe en Google Teachable Machine ("Hold to Record" por clase). La diferenciación real no está en la idea sino en el nicho (PYMEs latinoamericanas, hardware barato, edge offline) y en la ingeniería del pipeline de curación de frames.
- **Como tesis es excelente y viable**; como producto comercial es defendible solo con un foco muy estrecho. El mayor riesgo técnico es el **data leakage entre train/test por frames consecutivos correlacionados**, que puede inflar las métricas y arruinar la validez científica de la tesis si no se maneja con splitting por video/cluster.
- **Stack recomendado: políglota**. Python para todo el pipeline de ML (PyTorch, Ultralytics, anomalib, ONNX/TensorRT/OpenVINO); Go para el gateway/ingesta de video/alta concurrencia; TypeScript/React para el frontend. Empezar con **monolito modular + arquitectura hexagonal ligera**, no microservicios. Añadir detección de anomalías no supervisada (entrenar solo con piezas buenas) como diferenciador técnico de peso.

## Key Findings

1. **El "corazón" del concepto ya existe.** Google Teachable Machine permite a un no-experto crear clases y grabar muestras con la webcam ("Hold to Record") por clase; es el precedente directo del enfoque "grabar por clase". Roboflow y LandingLens ya extraen frames de video/RTSP automáticamente. Ninguna plataforma industrial de nivel empresarial comercializa exactamente el eslogan "un video por clase", así que hay un hueco de *packaging/UX*, no de invención técnica.

2. **Amazon Lookout for Vision fue descontinuado.** Según la FAQ oficial de AWS: "Effective October 10, 2024, new customers will no longer have access to the service. Active customers will be able to continue to use the service normally until October 31, 2025... After October 31, 2025, any cloud application that tries to access Lookout for Vision will no longer work." Esto confirma que incluso un hiperescalador falló en monetizar la inspección visual genérica como servicio autónomo. Es una señal de mercado importante: el producto puro "sube imágenes/entrena/detecta defectos" no basta; el valor está en la integración vertical y el servicio.

3. **El data leakage por frames consecutivos es un riesgo científico de primer orden**, documentado en la literatura: al partir aleatoriamente frames de video, frames casi idénticos caen en train y test, inflando las métricas. La mitigación correcta es splitting por video/grupo o clustering por similitud (p. ej. HDBSCAN sobre embeddings) antes de particionar.

4. **La detección de anomalías no supervisada (entrenar solo con piezas buenas) es el mayor diferenciador técnico disponible**: anomalib de Intel implementa PatchCore, PaDiM, EfficientAd. Según Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022): "On the challenging, widely used MVTec AD benchmark PatchCore achieves an image-level anomaly detection AUROC score of up to 99.6%, more than halving the error compared to the next best competitor" (99,1% en la configuración base de la Tabla 1; hasta 99,6% con backbone mayor/ensemble; segmentación pixelwise AUROC de 98,1%; solo 42 de 1725 imágenes mal clasificadas al umbral óptimo de F1). Esto elude el problema del desbalance de clases (los defectos son raros) que es letal para el enfoque supervisado "un video de piezas defectuosas por clase".

5. **Go encaja en el plano de datos/ingesta, no en ML.** El ecosistema de entrenamiento e inferencia está dominado por Python. La arquitectura correcta es "entrenar en Python, servir/orquestar donde convenga", con Python (FastAPI) o Go para la API y Go para streaming de alta concurrencia.

6. **La inferencia en edge es viable en hardware barato pero con límites reales.** Según el benchmark de Nature Scientific Reports (2026): "On Raspberry Pi 5, CPU-only execution of large models is impractical due to multi-second per-frame latency". En un Jetson Nano, el benchmark de Seeed Studio reporta que YOLOv5n corre "about 0.060s = 60ms, which is nearly 1000/60 = 16.7fps" sin optimizar, subiendo a **27 FPS con TensorRT** (y ~60 FPS en Jetson Xavier NX FP32). La elección hardware/optimización es decisiva para "vigilar la línea en vivo".

## Details

### 1. Viabilidad técnica y comercial

**Como tesis académica: viabilidad alta.** El proyecto tiene un alcance perfecto para una tesis de ingeniería de software con IA: combina un problema real, un pipeline de ML no trivial, decisiones de arquitectura defendibles y oportunidades de contribución medible (p. ej., un módulo de deduplicación/curación de frames y una comparación honesta de splitting con vs sin leakage). El hecho de que el estudiante sea ingeniero senior en ejercicio favorece la ejecución. La contribución investigable debe ser específica: no "construí una plataforma", sino "diseñé y evalué un método de extracción y curación de frames desde video que reduce redundancia y evita leakage, y lo comparé contra el baseline ingenuo".

**Como producto de mercado: viabilidad media-baja en el genérico, media-alta en un nicho estrecho.** El mercado está poblado por actores con años de ventaja, equipos grandes y hardware propio. La descontinuación de Amazon Lookout for Vision demuestra que el "producto horizontal" es difícil de monetizar. El espacio real está en: PYMEs de manufactura ligera en LATAM, con hardware de bajo costo, despliegue offline, en español, con precio accesible y servicio de implementación local.

**Riesgos técnicos (con severidad):**

- **Data leakage train/test por frames consecutivos (CRÍTICO).** Es el riesgo #1 para la validez de la tesis. Frames consecutivos de un mismo video son casi idénticos; un split aleatorio los reparte entre train y test e infla artificialmente las métricas. Mitigación obligatoria: split por video (todos los frames de un video van al mismo lado) o clustering por similitud (HDBSCAN sobre embeddings) antes de particionar. Documentar el delta de métricas con y sin leakage sería una contribución valiosa de la tesis. La literatura confirma que el splitting rayado ("1-in-10 frames") o aleatorio genera solapamiento masivo train/test en el espacio PCA de las dos componentes principales.
- **Calidad y redundancia del dataset extraído de video (ALTO).** Un video a 30 fps genera cientos de frames casi idénticos: mucho volumen, poca diversidad efectiva. Sin curación, se entrena con datos redundantes que dan falsa sensación de "muchos datos". Mitigación: muestreo inteligente (frame sampling por diferencia perceptual / hashing perceptual pHash + deduplicación por embeddings).
- **Desbalance de clases (ALTO).** En inspección real, las piezas buenas abundan y los defectos son raros. Pedir "un video por clase de defecto" es poco realista si el defecto casi no ocurre. Esto empuja fuertemente hacia detección de anomalías no supervisada (entrenar solo con "buenas").
- **Generalización con pocas condiciones de iluminación (ALTO).** Si todos los videos se graban en una sesión con una iluminación, el modelo no generaliza a cambios de turno/luz. Mitigación: augmentation, guías de captura, y detección de drift en producción.
- **Drift en producción (MEDIO-ALTO).** Cambios de lote, iluminación, desgaste de cámara degradan el modelo silenciosamente. Necesita monitoreo de drift y reentrenamiento; la literatura de MLOps indica que sobre el 70% de las organizaciones reportan drift significativo en los primeros seis meses de despliegue.
- **Latencia de inferencia en vivo (MEDIO, dependiente de hardware).** "Vigilar en vivo" exige throughput acorde a la velocidad de línea. En hardware barato sin optimización puede ser inviable; requiere ONNX/TensorRT/OpenVINO y elección de modelo ligero.

### 2. Competencia y estado del arte

**Plataformas no-code/low-code de visión industrial (2024-2026):**

- **Landing AI — LandingLens** (Andrew Ng). Plataforma no-code data-centric: subir imágenes, etiquetar, entrenar con un botón, desplegar en cloud o edge (LandingEdge/Docker/ONNX). Soporta extracción de frames de video y RTSP vía su librería Python (`landingai-python`: `Webcam`, `NetworkedCamera`, extracción de frames de archivos). Modelo de negocio freemium: tier gratuito con **1.000 créditos/mes** (1 crédito = 1 imagen entrenada/inferencia/imagen desplegada, no acumulables) y tier Enterprise a medida (con descarga de modelo ONNX); disponible en AWS Marketplace y como app nativa de Snowflake. Landing AI publica oficialmente solo Free + Enterprise; las cifras intermedias de agregadores terceros ($39, $250/mes) son poco fiables. **Es el competidor conceptual más directo del enfoque no-code.**
- **Cognex In-Sight D900 / ViDi / VisionPro Deep Learning**. Líder industrial. Cámara inteligente con deep learning embebido; entrena "en minutos con tan solo 5-10 imágenes por clase, sin código". Modos supervisado y no supervisado (ViDi Red en modo unsupervised: solo referencias normales). Sin precio de lista público (venta por cotización/distribuidor); puntos de referencia de reventa sitúan las unidades D905 en ~18.000-27.000 USD, confirmando el rango premium (~15.000-30.000 USD/cámara + licencia de entrenamiento ViDi). Basado en imágenes, no en video-por-clase.
- **Keyence**. Sistemas de visión industrial con IA, modelo comercial similar a Cognex (hardware + servicio, ventas por cotización).
- **Roboflow**. Plataforma end-to-end: extracción de frames de video en el navegador (tasa de muestreo elegible), auto-etiquetado (RF-DETR, SAM), entrenamiento, despliegue en edge (Jetson, Raspberry Pi) y RTSP. Freemium: plan público gratuito con **60 USD/mes en créditos** (datos/modelos públicos); plan Core **desde 79 USD/mes** (anual; 99 USD/mes si mensual), datos privados; Enterprise a medida (con triggers MQTT/OPC/PLC y frame-grabbers industriales). Guía oficial de migración desde Lookout for Vision.
- **Edge Impulse**. Edge ML/TinyML. **FOMO-AD**: detección de anomalías visuales en edge (GPU a MCU) entrenando solo con muestras normales (~100 imágenes "No Anomaly"). Plan Developer gratuito; Enterprise a medida; **FOMO-AD es exclusivo de Enterprise**. Fuerte en hardware barato (demostraciones en placas de ~100 USD como Rubik Pi 3).
- **Google Teachable Machine**. Herramienta educativa/consumo, gratuita, en navegador: crea clases y graba muestras con la webcam ("Hold to Record") por clase (~30-60+ muestras/clase), entrena localmente (TensorFlow.js). **Es el precedente directo del mecanismo "grabar por clase"**, aunque no es industrial.
- **Google Vertex AI / AutoML Vision** y **Visual Inspection AI**. Entrenamiento gestionado de clasificación/detección no-code; Visual Inspection AI orientado a defectos de manufactura con active learning. Facturación por node-hour (entrenamiento ~3,465 USD/node-hora; edge on-device ~18 USD/node-hora; predicción online ~1,375 USD/node-hora, según guías 2026). En migración hacia la plataforma de agentes Gemini Enterprise.
- **Azure Custom Vision**. Clasificación/detección custom no-code. Tier gratuito F0 (2 proyectos, 1 h de entrenamiento/mes, 10.000 predicciones/mes) y estándar S0 (**2 USD por 1.000 transacciones**, entrenamiento **10 USD/hora de cómputo**, almacenamiento **0,70 USD/1.000 imágenes**). Nota: varias APIs de custom image de Azure Computer Vision fueron retiradas el 31 mar 2025 (fragmentación del portfolio de Microsoft).
- **Amazon Lookout for Vision**. **DESCONTINUADO** (sin nuevos clientes desde 10 oct 2024; apagado 31 oct 2025, según FAQ oficial de AWS). AWS recomienda migrar a SageMaker/Bedrock o partners. Señal de mercado clave.
- **Zebra Aurora (ex-Adaptive Vision) / Aurora Deep Learning**. Suite de visión industrial; en 2024 añadió herramientas de anomalía por deep learning no supervisado ("only needing normal references") y OCR no-code. Entrena con "20-30 imágenes". Programación visual (Studio) o librerías C++/.NET; motor de inferencia WEAVER.
- **Matroid**. Plataforma no-code empresarial, cámara-agnóstica, nativa de video (VMS con IA que genera datos de entrenamiento desde grabaciones). Clientes como Mercedes-Benz y Stanley Black & Decker. Orientada a detección/SOP, no a "un video por clase".
- **Neurala**. Visión industrial (VIA — Vision Inspection Automation) para defectos, orientada a manufactura.
- **Instrumental**. Inspección con IA en líneas de ensamblaje electrónico (hardware + software + servicio), foco en detección de anomalías de proceso.
- **Ultralytics HUB / Platform**. No-code sobre YOLO (hasta YOLO26): etiquetado (SAM), entrenamiento en 26 GPUs cloud **desde 0,24 USD/hora**, export a ONNX/TensorRT/OpenVINO/TFLite. Plan gratuito; Pro **29 USD/asiento/mes**. Licencia AGPL-3.0 del código base (derivados comerciales requieren licencia Enterprise) — **implicación legal relevante si el producto integra YOLO de Ultralytics**.
- **V7 (Darwin)**, **Superb AI**, **Datature**. Plataformas de data labeling/MLOps de visión con automatización (auto-anotación, modelos fundacionales); Datature publica guías prácticas de anomalib. Compiten más en el plano de datos/labeling que en inspección industrial llave en mano.
- **Adyacentes industriales de bajo-shot** detectados: Elementary (VisionStream), Lincode LIVIS (operarios entrenan "AI Inspectors" con captura móvil), Averroes.ai ("20-40 imágenes por clase de defecto"), Tulip Vision (cámaras básicas que enrutan a APIs cloud). Todas image-based o frame-extraction, ninguna "un video por clase".

**¿Alguna usa exactamente "un video por clase"?** No a nivel industrial. El precedente más cercano del *mecanismo* es Teachable Machine (grabar webcam por clase); Roboflow y LandingLens hacen extracción de frames de video pero como paso de *adquisición* que alimenta un pipeline convencional de subir-y-etiquetar, no como paradigma de entrenamiento comercializado "un video por clase". Esto confirma que hay un hueco de UX/empaquetado, no una invención algorítmica.

### 3. Diferenciación y mejoras

Dado el panorama, la idea "genérica" no es defendible; hay que estrecharla y profundizarla técnicamente:

- **Foco vertical: PYMEs latinoamericanas de manufactura ligera.** Precio accesible, español, servicio de implementación local, hardware barato (webcam USB + Raspberry Pi 5 / Jetson Orin Nano). Este es el diferenciador de *mercado* más honesto.
- **Detección de anomalías no supervisada como default (DIFERENCIADOR TÉCNICO CLAVE).** En vez de exigir "un video por clase de defecto" (irreal por desbalance), grabar solo piezas buenas y usar PatchCore/PaDiM/EfficientAd (anomalib). Elude el desbalance y el etiquetado. PatchCore alcanza hasta 99,6% AUROC en MVTec AD. El enfoque "un video de piezas buenas" es más realista y vendible que "un video por clase".
- **Edge deployment offline.** Muchas fábricas no quieren/pueden enviar video a la nube. Inferencia local (ONNX/TensorRT/OpenVINO/TFLite) es un diferenciador frente a plataformas cloud-first.
- **Active learning.** Pedir más video/frames solo cuando el modelo tiene baja confianza, reduciendo el esfuerzo del operario.
- **Few-shot / modelos fundacionales.** DINOv2, CLIP/WinCLIP, SAM, Grounding DINO reducen datos necesarios: WinCLIP hace anomaly detection zero/few-shot (91,8%/85,1% AUROC zero-shot en MVTec AD; 93,1%/95,2% en 1-normal-shot con WinCLIP+); DINOv2 y SAM permiten auto-etiquetado y few-shot. Reduce el número de videos requeridos.
- **Curación automática de frames (NÚCLEO DE LA TESIS).** Deduplicación por hashing perceptual + embeddings, muestreo por diversidad, y splitting anti-leakage. Aquí está la contribución investigable y el verdadero valor de ingeniería.
- **MLOps automatizado y monitoreo de drift** integrados para el usuario no-experto.

### 4. Lenguajes y stack tecnológico

**Veredicto honesto sobre Go:** Go es una buena elección para partes del sistema, pero **no** para el ML.

- **Donde Go SÍ encaja**: API gateway, orquestación de workflows, servicios de ingesta/streaming de video de alta concurrencia (goroutines), servidor de eventos en tiempo real. Go ofrece mejor throughput y menor uso de memoria que Python en serving/gateway (benchmarks reportan hasta ~7x más throughput y ~80% menos RAM al migrar un microservicio de serving de Python a Go), y binarios únicos ideales para edge.
- **Donde Go NO encaja**: entrenamiento e inferencia de modelos. El ecosistema está en Python (PyTorch, Ultralytics YOLO, anomalib, OpenCV). Forzar Go aquí sería reinventar la rueda con herramientas inmaduras (Gonum, GoLearn, GoMLX siguen siendo marginales).

**Arquitectura políglota recomendada:**
- **Pipeline ML (Python)**: extracción de frames (OpenCV/GStreamer), deduplicación (pHash + embeddings DINOv2), entrenamiento (PyTorch/Ultralytics/anomalib), evaluación, export a ONNX.
- **Inferencia**: ONNX Runtime (portátil), TensorRT (Jetson/NVIDIA), OpenVINO (Intel/CPU), TFLite (dispositivos muy limitados).
- **Backend/API**: FastAPI (Python) para máxima cohesión con el pipeline ML **o** Go si se prioriza concurrencia y el equipo lo domina. Para una tesis, FastAPI reduce fricción; Go se justifica si el streaming de video en vivo es el foco.
- **Frontend no-code**: TypeScript + React/Next.js.
- **Ingesta de video en vivo**: RTSP/WebRTC + GStreamer; Go para el servicio de ingesta de alta concurrencia.
- **Colas/mensajería**: Redis (simple, para empezar) o NATS (ligero, Go-native); RabbitMQ/Kafka solo si el volumen lo exige (Kafka es sobreingeniería para una tesis).
- **Almacenamiento**: MinIO/S3 para video y frames; PostgreSQL para metadatos (proyectos, modelos, versiones, resultados de inferencia).
- **Comunicación Python↔Go**: gRPC (contratos tipados con Protocol Buffers) o REST; empaquetar el modelo Python como contenedor y tratarlo como caja negra, con health checks y circuit breakers.

### 5. Arquitectura

**Monolito modular vs microservicios: empezar modular.** Para una tesis, los microservicios son sobreingeniería (complejidad operativa, despliegue, observabilidad). Recomendación: **monolito modular** bien organizado, con límites internos claros que permitan extraer microservicios después si el producto crece. Separar desde el día uno el **plano de control** (gestión de proyectos, datasets, modelos, versiones) del **plano de datos** (inferencia en tiempo real), porque tienen requisitos de latencia y escalado muy distintos.

**Arquitectura orientada a eventos para el pipeline de entrenamiento:** el flujo `video subido → extracción de frames → curación/dedup → split anti-leakage → entrenamiento → evaluación → despliegue` se modela naturalmente como una cadena de eventos/estados. Cada etapa consume un evento y emite el siguiente (pub/sub sobre Redis/NATS). Esto da trazabilidad, reintentos y desacoplamiento.

**Arquitectura hexagonal (puertos y adaptadores) — evaluación para este dominio:**
- **Ventajas fuertes aquí**: el dominio tiene múltiples puntos de integración volátiles que son candidatos perfectos a puertos/adaptadores: (a) *motor de ML* (intercambiar YOLO ↔ anomalib PatchCore ↔ modelo fundacional), (b) *almacenamiento* (MinIO ↔ S3 ↔ disco local en edge), (c) *fuente de video* (archivo ↔ RTSP ↔ webcam ↔ WebRTC), (d) *runtime de inferencia* (ONNX ↔ TensorRT ↔ OpenVINO). La lógica de dominio (gestión de proyectos, orquestación del pipeline, reglas de curación) queda pura e independiente de tecnología, y se testea con adaptadores mock sin infraestructura. La literatura de MLOps documenta hexagonal precisamente para sistemas ML-enabled (caso Ocean Guard, arXiv 2506.06202 / 2512.08657), dividiendo el sistema en core (lógica con DDD), ports (contratos) y adapters (implementaciones), conectados por inyección de dependencias.
- **Cuándo es sobreingeniería**: si se aplica dogmáticamente a *cada* componente (puertos para todo), añade capas de transformación y curva de aprendizaje sin retorno. La regla: aplicar hexagonal solo en las fronteras que realmente van a variar (motor ML, storage, fuente de video, runtime). El resto, código directo.
- **Combinación con Clean Architecture y DDD**: hexagonal, clean y onion son variaciones del mismo principio (dominio en el centro, dependencias hacia adentro). Usar DDD ligero para modelar el dominio (entidades: Proyecto, Dataset, Video, Frame, ModeloEntrenado, Inspección) y hexagonal para las fronteras técnicas. No hace falta DDD táctico completo (agregados complejos, event sourcing) para una tesis.

**Diagrama conceptual (descrito):**

```
┌──────────────────── FRONTEND (React/Next.js, TS) ────────────────────┐
│   UI no-code: crear proyecto, grabar/subir video por clase,           │
│   ver métricas, desplegar, monitorear línea en vivo                    │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │ REST/WebSocket
┌───────────────────────────────▼───────────────────────────────────────┐
│                    PLANO DE CONTROL (API - Go o FastAPI)                │
│  [Puerto: Proyectos] [Puerto: Datasets] [Puerto: Modelos]             │
│         NÚCLEO DE DOMINIO (puro, testeable, DDD ligero)                 │
│  Reglas: orquestación del pipeline, políticas de curación/split        │
└───┬───────────────┬────────────────┬───────────────────┬───────────────┘
    │ Adaptador      │ Adaptador      │ Adaptador          │ Eventos (pub/sub
    │ Storage        │ Motor ML       │ Fuente Video       │ Redis/NATS)
┌───▼────┐    ┌──────▼───────┐   ┌────▼─────────┐   ┌──────▼──────────────┐
│MinIO/S3│    │ Pipeline ML  │   │ RTSP/WebRTC/ │   │ Cola de trabajos     │
│Postgres│    │ (Python):    │   │ archivo/     │   │ extracción→dedup→    │
│        │    │ extracción,  │   │ webcam       │   │ train→eval→deploy    │
│        │    │ dedup (pHash/│   │ (GStreamer)  │   └──────────────────────┘
│        │    │ DINOv2),     │   └──────────────┘
│        │    │ YOLO/anomalib│
│        │    │ →ONNX        │
└────────┘    └──────────────┘
                                 │ modelo exportado
┌────────────────────────────────▼──────────────────────────────────────┐
│              PLANO DE DATOS (Inferencia en tiempo real)                 │
│  [Puerto: Runtime] → Adaptador ONNX / TensorRT / OpenVINO / TFLite     │
│  Edge (Jetson/RPi) offline  ó  Cloud   →  eventos de defecto (pub/sub)  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6. Normas y patrones de código

**Estándares por lenguaje:**
- **Python**: PEP 8, type hints obligatorios, `ruff` (linter+formatter rápido) o `black`+`ruff`, `mypy` para tipado estático.
- **Go**: Effective Go, `gofmt`, `golangci-lint`, manejo explícito de errores.
- **TypeScript**: ESLint + Prettier, `strict` en tsconfig.

**Patrones de diseño útiles en este dominio:**
- **Strategy**: motores de ML intercambiables (YOLO / PatchCore / few-shot) tras una interfaz común.
- **Pipeline / Chain of Responsibility**: procesamiento de video por etapas (decodificar → extraer → deduplicar → augmentar → particionar).
- **Repository**: acceso a datos (datasets, modelos, metadatos) desacoplado de Postgres/MinIO.
- **Factory**: construcción de modelos/adaptadores según configuración.
- **Observer / pub-sub**: eventos de inferencia y de pipeline (defecto detectado → alerta).
- **Circuit Breaker**: proteger el servicio de inferencia ante fallos/latencia (los modelos ML fallan de formas impredecibles); health checks + fallback.

**Testing y MLOps:**
- Unit + integración clásicos.
- **Tests de modelos ML**: datasets de regresión (un "gold set" fijo con umbral mínimo de métrica que debe superarse antes de desplegar), tests de invarianza, y verificación explícita de ausencia de leakage en los splits.
- **Versionado de datos con DVC**, tracking de experimentos con **MLflow**, CI/CD que ejecute la evaluación sobre el gold set y bloquee despliegues que regresen.
- Monitoreo de drift en producción (Evidently AI u similar) con reentrenamiento disparado por umbral.

### 7. Roadmap

**Fase 0 — Fundaciones (tesis, semanas 1-4):** definir el alcance investigable (curación de frames + anti-leakage), montar repos, CI, estándares de código, esqueleto hexagonal. Dataset de validación (usar MVTec AD como benchmark de referencia + un caso propio grabado en video).

**Fase 1 — MVP de tesis (el núcleo):**
- Ingesta de un video por clase (o solo "buenas" para modo anomalía).
- Pipeline: extracción → deduplicación (pHash/embeddings) → split anti-leakage → entrenamiento.
- **Dos modos**: (a) clasificación/detección supervisada (YOLO/Ultralytics) y (b) anomalía no supervisada (anomalib PatchCore/PaDiM) — este último como diferenciador.
- Evaluación honesta: reportar métricas con y sin control de leakage (contribución científica).
- Inferencia sobre stream RTSP/webcam con export ONNX.
- Frontend no-code mínimo.

**Fase 2 — Robustez (fin de tesis / pre-producto):** active learning (pedir video solo con baja confianza), monitoreo de drift, augmentation guiada, despliegue edge (Jetson/RPi con TensorRT/OpenVINO), versionado DVC/MLflow.

**Fase 3 — Producto comercial:** multi-tenant, RBAC, gestión de flotas de cámaras, integración PLC/MQTT (como hace Roboflow para manufactura), modelos fundacionales para few-shot, HMI para operarios, soporte y precio para PYMEs LATAM.

## Recommendations

1. **Reposicionar la tesis alrededor de la curación de frames y el anti-leakage, no del "no-code".** El no-code ya está resuelto por otros; la contribución defendible y original es el método de extracción/deduplicación/split desde video. Medir cuantitativamente el impacto del leakage (delta de F1/AUROC con y sin control) es la joya científica del trabajo. Benchmark: usar MVTec AD para comparabilidad y un caso propio en video. **Umbral de éxito**: demostrar un delta de métrica medible (p. ej. >5 puntos de AUROC/F1) entre el split ingenuo y el split anti-leakage valida la contribución.

2. **Añadir el modo de anomalía no supervisada (anomalib) como diferenciador técnico central.** "Graba solo piezas buenas" resuelve el desbalance y es más realista y vendible que "un video por clase de defecto". **Umbral de decisión**: si el defecto ocurre en <5-10% de las piezas, el modo anomalía debe ser el default.

3. **Stack: Python para ML (no negociable); FastAPI para el backend de la tesis, reservando Go para el servicio de ingesta de video en vivo** si el streaming concurrente resulta ser cuello de botella. No usar Go para ML. Empezar monolito modular + hexagonal *selectivo* (puertos solo en motor ML, storage, fuente de video, runtime de inferencia).

4. **No competir en el genérico; elegir el nicho PYME LATAM + edge offline + hardware barato.** Validar con 1-2 fábricas reales durante la tesis (aunque sea un piloto). **Benchmark de viabilidad comercial**: si no se consigue al menos un piloto real con una PYME, el producto queda como prueba de concepto académica (lo cual es perfectamente válido para una tesis, pero conviene declararlo explícitamente).

5. **Establecer desde el inicio un "gold set" de regresión y CI que bloquee despliegues con métricas por debajo del umbral**, más monitoreo de drift. Esto convierte la tesis en un ejemplo de MLOps serio, no solo un demo.

6. **Presupuestar la latencia contra la velocidad real de línea.** Antes de prometer "en vivo", medir FPS del pipeline completo (decodificación + preproceso + inferencia + postproceso) en el hardware objetivo. Si es Raspberry Pi, casi seguro se necesita optimización (OpenVINO/NCNN) o pasar a un Jetson; en Jetson Nano, TensorRT llevó YOLOv5n de ~16,7 FPS a 27 FPS. **Umbral**: definir el FPS mínimo requerido por la velocidad de la cinta y no desplegar hasta alcanzarlo con margen.

7. **Revisar la licencia AGPL-3.0 de Ultralytics YOLO** antes de comercializar: un producto que integre ese código exige licencia Enterprise de pago o liberar el código fuente. Alternativas: RF-DETR, modelos propios en PyTorch, o anomalib (Apache-2.0).

## Caveats

- **Precios y estados de producto cambian rápido.** Las cifras de pricing (Roboflow 79 USD/mes, Ultralytics 29 USD/asiento y GPUs desde 0,24 USD/h, Azure 2 USD/1.000 transacciones + 10 USD/h de cómputo, Vertex ~3,465 USD/node-hora, LandingLens 1.000 créditos/mes gratis) provienen en parte de páginas oficiales y en parte de agregadores terceros que pueden desactualizarse; verificar en las páginas oficiales antes de citarlas en la tesis. Landing AI publica oficialmente solo tier gratuito + Enterprise a medida; las cifras intermedias de terceros son poco fiables. Cognex no publica precio de lista (los ~18k-27k USD son datos de reventa, orientativos).
- **Vertex AI AutoML está migrando** a la plataforma de agentes Gemini Enterprise; algunas APIs de Azure Computer Vision fueron retiradas en 2025. El portfolio de los hiperescaladores en visión custom está en flujo.
- **Las métricas de anomaly detection (PatchCore hasta 99,6% AUROC en MVTec AD) son de benchmark académico**; el rendimiento real en una PYME con iluminación pobre y cámara barata será menor. MVTec introdujo MVTec AD 2 (8.004 imágenes, 8 escenarios, publicado 2025) precisamente porque los métodos saturaron el benchmark original (segmentación AU-PRO de 92-97% "near-saturation"); no extrapolar cifras de laboratorio a producción.
- **Los números de latencia en edge son muy dependientes de modelo, resolución, precisión (FP32/FP16/INT8) y runtime.** Las cifras citadas (multi-segundo por frame para modelos grandes en Raspberry Pi 5 según Nature 2026; 16,7→27 FPS YOLOv5n con TensorRT en Jetson Nano según Seeed Studio; ~1.817 ms para YOLOv5s en Raspberry Pi según el paper Eagle, arXiv 2304.04356) son puntos de referencia, no garantías para este caso concreto.
- **"Un video por clase" no es original como mecanismo** (Teachable Machine lo hace con "Hold to Record"); la originalidad debe construirse en la ingeniería del pipeline (curación anti-leakage, deduplicación, modo anomalía) y el nicho, no en el eslogan.