# Hoja de Ruta del Prototipo de Tesis
## Plataforma No-Code de Inspección Visual Industrial (entrenamiento por video)

**Autor:** Jersy Claudio Baltazar — Ingeniería de Software con Inteligencia Artificial
**Supuesto de duración:** 16 semanas (un semestre académico). Si tu ventana real es distinta, las fases escalan proporcionalmente.
**Alcance del prototipo (MVP):** un modo de entrenamiento principal (detección de anomalías con piezas "buenas"), una cámara fija, dashboard web, inferencia en vivo local. Todo lo demás es trabajo futuro declarado.

---

## 1. Requerimientos

### 1.1 Requerimientos funcionales

| ID | Requerimiento | Prioridad |
|----|---------------|-----------|
| RF-01 | El usuario puede crear un "proyecto de inspección" con nombre, descripción y tipo de pieza | Alta |
| RF-02 | El usuario puede grabar o subir un video de piezas buenas (modo anomalía) desde la interfaz web, con guía de captura en pantalla (distancia, iluminación, rotación de la pieza) | Alta |
| RF-03 | El sistema extrae frames del video automáticamente con muestreo inteligente (no todos los frames: filtrado por diferencia perceptual) | Alta |
| RF-04 | El sistema deduplica frames redundantes usando hashing perceptual (pHash) y/o embeddings | Alta |
| RF-05 | El sistema particiona los datos con split anti-leakage (por video o por cluster de similitud, nunca aleatorio por frame) | Alta |
| RF-06 | El usuario inicia el entrenamiento con un botón; el sistema entrena un modelo de detección de anomalías (anomalib: PatchCore o PaDiM) sin intervención técnica | Alta |
| RF-07 | El sistema muestra el progreso del pipeline (extracción → curación → entrenamiento → evaluación) en tiempo real | Media |
| RF-08 | Al finalizar, el sistema muestra métricas comprensibles para un operario (ej. "detecta 96 de cada 100 defectos" + AUROC/F1 para el informe técnico) y ejemplos visuales de aciertos/fallos | Alta |
| RF-09 | El usuario puede activar el "modo vigilancia": el sistema procesa el stream de la cámara (webcam USB o RTSP) y clasifica cada pieza como OK / defecto en vivo | Alta |
| RF-10 | Cada detección de defecto genera una alerta visual en el dashboard con el frame capturado, timestamp y score de anomalía | Alta |
| RF-11 | El usuario puede ajustar el umbral de sensibilidad con un control simple (más estricto ↔ más permisivo) | Media |
| RF-12 | El sistema guarda un historial de inspecciones consultable con filtros por fecha y resultado | Media |
| RF-13 | El usuario puede exportar el modelo entrenado (ONNX) y un reporte de inspecciones (CSV/PDF) | Baja |
| RF-14 | (Opcional, si el tiempo alcanza) Modo supervisado: un video por clase de defecto, con clasificación multiclase (YOLO/clasificador) | Baja |

### 1.2 Requerimientos no funcionales

| ID | Requerimiento | Criterio de aceptación |
|----|---------------|------------------------|
| RNF-01 | Latencia de inferencia en vivo | ≥ 5 FPS de extremo a extremo en el hardware objetivo (definir: laptop con GPU, Jetson o Raspberry Pi + OpenVINO). Medirlo, no asumirlo |
| RNF-02 | Tiempo de entrenamiento | ≤ 30 min por proyecto en el hardware de desarrollo (PatchCore/PaDiM no requieren epochs largos) |
| RNF-03 | Usabilidad no-code | Un usuario sin conocimientos de ML completa el flujo grabar→entrenar→vigilar sin ayuda, validado con prueba de usabilidad con ≥ 3 personas |
| RNF-04 | Operación offline | La inferencia funciona sin conexión a internet (requisito del nicho PYME/edge) |
| RNF-05 | Validez científica | Las métricas reportadas provienen exclusivamente de splits anti-leakage; el informe incluye la comparación con split ingenuo como evidencia |
| RNF-06 | Calidad de código | Linters en CI (ruff+mypy para Python, ESLint para TS), cobertura de tests ≥ 60% en el núcleo de dominio |
| RNF-07 | Arquitectura | Monolito modular con puertos/adaptadores (hexagonal selectivo) en: motor ML, almacenamiento, fuente de video, runtime de inferencia |
| RNF-08 | Trazabilidad | Cada modelo entrenado queda versionado con su dataset, parámetros y métricas (MLflow o registro propio en Postgres) |

### 1.3 Requerimientos de hardware y entorno

- Cámara: webcam USB 1080p fija (mínimo); opcional cámara IP RTSP.
- Equipo de desarrollo/entrenamiento: PC con GPU NVIDIA (aunque PatchCore corre razonable en CPU).
- Equipo de inferencia (demo): la misma PC, o Jetson Orin Nano / Raspberry Pi 5 con OpenVINO si quieres demostrar edge (suma puntos, suma riesgo).
- Piezas físicas de prueba: elige 2-3 tipos de objetos con defectos reproducibles (ej. tapas plásticas rayadas, tuercas con rebabas, empaques mal impresos). Consíguelos en la semana 1, no en la 10.
- Dataset de referencia: MVTec AD para validar el pipeline contra un benchmark público, además de tu dataset propio en video.

### 1.4 Stack confirmado

- **Pipeline ML (Python):** OpenCV (extracción), imagehash/pHash + embeddings (dedup), anomalib (entrenamiento), ONNX Runtime (inferencia).
- **Backend (Python/FastAPI):** API REST + WebSocket para progreso y alertas en vivo. (Go queda como mejora futura para ingesta concurrente; para la tesis, un solo lenguaje de backend reduce riesgo.)
- **Cola de trabajos:** Redis + worker (RQ o Celery) para el pipeline asíncrono.
- **Frontend (TypeScript/React o Next.js):** UI no-code.
- **Datos:** PostgreSQL (metadatos), MinIO o disco local (videos/frames/modelos).
- **MLOps:** MLflow (experimentos), DVC opcional, GitHub Actions (CI).

---

## 2. Hoja de ruta por fases

**Fase 0 — Fundaciones (semanas 1-2).** Definición formal del alcance, matriz de trazabilidad con el PEA, repositorio, CI con linters, esqueleto hexagonal (dominio + puertos), adquisición de piezas físicas y cámara, y captura del primer lote de videos. Entregable: repositorio operativo + plan de tesis aprobado por el asesor.

**Fase 1 — Pipeline de datos (semanas 3-5).** El corazón investigable: extracción de frames, deduplicación, split anti-leakage. Aquí se produce la evidencia científica (comparación split ingenuo vs anti-leakage). Entregable: módulo de curación con tests + notebook de experimentos con resultados sobre MVTec AD y dataset propio.

**Fase 2 — Entrenamiento y evaluación (semanas 6-8).** Integración de anomalib, orquestación asíncrona del pipeline completo, registro de modelos y métricas, export a ONNX. Entregable: entrenar un modelo de punta a punta desde un video, por API.

**Fase 3 — Inferencia en vivo (semanas 9-10).** Servicio de inferencia sobre webcam/RTSP, umbral ajustable, eventos de defecto por WebSocket, medición formal de FPS (RNF-01). Entregable: demo en consola/API de vigilancia en vivo.

**Fase 4 — Frontend no-code (semanas 11-13).** UI completa del flujo: crear proyecto → grabar/subir → entrenar (con progreso) → ver métricas → vigilar en vivo → historial. Entregable: prototipo integrado usable por un no-técnico.

**Fase 5 — Validación y cierre (semanas 14-16).** Prueba de usabilidad con ≥ 3 usuarios, corrida final de experimentos, redacción de resultados, video de demo de respaldo (obligatorio: nunca dependas solo de la demo en vivo el día de la sustentación), y documentación de arquitectura (diagramas C4/UML — que además cubre "Modelado y Diseño del Software" del PEA). Entregable: prototipo congelado + capítulos de resultados.

---

## 3. Cronograma semanal

| Semana | Actividades clave | Hito / entregable |
|--------|-------------------|-------------------|
| 1 | Plan de tesis, alcance MVP, matriz PEA↔proyecto, compra de piezas y cámara | Plan aprobado por asesor |
| 2 | Repo, CI (ruff/mypy/ESLint), esqueleto hexagonal, modelo de datos en Postgres, primeros videos grabados | Esqueleto ejecutable + dataset v0 |
| 3 | Extracción de frames (OpenCV) con muestreo por diferencia perceptual | Módulo extracción + tests |
| 4 | Deduplicación (pHash + embeddings), métricas de redundancia eliminada | Módulo dedup + tests |
| 5 | Split anti-leakage (por video/cluster) y experimento comparativo vs split ingenuo en MVTec AD + dataset propio | **Evidencia científica clave** (tabla de deltas de métrica) |
| 6 | Integración anomalib (PatchCore/PaDiM) tras el puerto "MotorML" | Primer modelo entrenado |
| 7 | Orquestación asíncrona (Redis + worker), estados del pipeline, MLflow | Pipeline end-to-end por API |
| 8 | Evaluación automática, export ONNX, registro de versiones de modelo | Modelo versionado + métricas |
| 9 | Servicio de inferencia en vivo (webcam/RTSP), umbral ajustable | Vigilancia en vivo por API |
| 10 | Optimización de latencia (ONNX Runtime/OpenVINO), medición formal de FPS | Informe RNF-01 cumplido |
| 11 | Frontend: proyectos, captura/subida de video con guía, progreso del pipeline | Flujo entrenar completo en UI |
| 12 | Frontend: métricas comprensibles, ejemplos visuales, modo vigilancia con alertas | Flujo vigilar completo en UI |
| 13 | Historial de inspecciones, exportes (CSV/ONNX), pulido UX, colchón de retrasos | Prototipo integrado (feature freeze) |
| 14 | Prueba de usabilidad (≥3 usuarios no técnicos), corrección de hallazgos críticos | Informe de usabilidad |
| 15 | Corrida final de experimentos, video demo de respaldo, documentación C4/UML | Paquete de evidencias |
| 16 | Redacción de resultados/conclusiones, ensayo de sustentación | Prototipo congelado + tesis lista para revisión |

**Regla de gestión:** cada fase tiene un colchón implícito (semana 13 es amortiguador explícito). Si en la semana 8 el pipeline end-to-end no funciona, recorta: elimina RF-14, RTSP (deja solo webcam) y exportes; jamás recortes la semana 5, que es tu contribución.

---

## 4. Riesgos principales y mitigación

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Data leakage invalida métricas | Crítico (validez de tesis) | Split por video/cluster desde el día 1; documentar comparación (semana 5) |
| Latencia insuficiente en vivo | Alto | Medir FPS temprano (semana 9-10); fallback: bajar resolución, procesar 1 de cada N frames, o demo en laptop con GPU |
| Demo en vivo falla en sustentación | Alto | Video de demo grabado como respaldo (semana 15) |
| Alcance se infla | Alto | RF-14 y edge deployment son opcionales; feature freeze en semana 13 |
| Piezas/defectos físicos difíciles de conseguir | Medio | Elegir objetos cotidianos con defectos reproducibles en semana 1; MVTec AD como plan B experimental |
| anomalib/dependencias dan problemas de integración | Medio | Puerto "MotorML" permite cambiar PatchCore↔PaDiM↔clasificador simple sin tocar el dominio |

---

## 5. Criterios de éxito del prototipo (para la sustentación)

1. Un usuario no técnico entrena un modelo desde un video y activa la vigilancia sin ayuda.
2. AUROC ≥ 0.90 en el dataset propio con split anti-leakage (y resultados comparables en al menos 2-3 categorías de MVTec AD).
3. Delta demostrado entre split ingenuo y anti-leakage (la evidencia de que tu método importa).
4. ≥ 5 FPS de inferencia en vivo medidos en el hardware declarado.
5. Arquitectura documentada (hexagonal selectivo) con al menos un adaptador intercambiado en vivo o en tests (ej. PatchCore ↔ PaDiM) como demostración del diseño.
