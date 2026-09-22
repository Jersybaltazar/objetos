# Plataforma No-Code de Inspección Visual Industrial

Prototipo de tesis. Entrena modelos de detección de anomalías visuales a partir de
video grabado por el propio operario, sin conocimientos de machine learning.

**Autor:** Jersy Claudio Baltazar — Ingeniería de Software con Inteligencia Artificial

---

## Qué hay aquí

| Documento | Contenido |
|-----------|-----------|
| [hoja-ruta-tesis-inspeccion-visual.md](hoja-ruta-tesis-inspeccion-visual.md) | Requerimientos, fases, cronograma de 16 semanas, riesgos |
| [protocolo-evaluacion-experimental.md](protocolo-evaluacion-experimental.md) | **Diseño experimental congelado**: E1–E4, métricas, estadística, controles anti-fuga |
| [compass_artifact_wf-490f58d2...md](compass_artifact_wf-490f58d2-7e7a-595c-86b2-d9071b9a2ba0_text_markdown.md) | Plan de semana 1: marco teórico, referencias, hardware, protocolo de captura |
| [compass_artifact_wf-ed141eba...md](compass_artifact_wf-ed141eba-8d67-5f45-aa5f-66c7b9367c67_text_markdown.md) | Evaluación de arquitectura y estado del arte comercial |

## La contribución, en una frase

Los frames consecutivos de un video son casi idénticos. Si se reparten al azar
entre entrenamiento y test, el modelo evalúa sobre material que ya memorizó y las
métricas se inflan. Esta tesis diseña, implementa y mide el método de curación que
lo evita: muestreo por diferencia perceptual, deduplicación por hashing y
**partición por pieza física**, nunca por frame.

El módulo que lo implementa es [backend/src/inspeccion/application/curation/](backend/src/inspeccion/application/curation/).
No depende de PyTorch ni de anomalib a propósito: debe poder ejecutarse, testearse
y auditarse en cualquier máquina.

## Estructura

```
backend/            FastAPI + núcleo de dominio (hexagonal selectivo)
  src/inspeccion/
    domain/         Entidades, value objects y los puertos
    application/    Casos de uso — dependen solo de puertos
      curation/     <- contribución científica: sampling, dedup, splitting, pipeline
      evaluation/   métricas, calibración de umbral, pruebas estadísticas
    adapters/       Implementaciones concretas de los puertos
      hashing/        imagehash (pHash/dHash/aHash/wHash)  [Strategy]
      video_source/   OpenCV: archivo, webcam USB, RTSP
      storage/        disco local (MinIO/S3 en fase posterior)
      ml_engine/      anomalib: PatchCore / PaDiM  [Strategy]
    config/         Raíz de composición: Settings + Container  [Factory + DI]
  tests/            Incluye dobles en memoria de todos los puertos
ml/                 Experimentos, notebooks, manifiestos de partición
frontend/           React/Next.js (Fase 4)
infra/              docker-compose: PostgreSQL, MinIO, Redis
```

Los puertos se definen **solo** en las cuatro fronteras que van a variar (motor
ML, almacenamiento, fuente de video, runtime de inferencia). El resto es código
directo: poner puertos en todas partes sería sobreingeniería.

Patrones aplicados: **Strategy** (algoritmos de hashing y estrategias de
partición intercambiables tras su puerto), **Pipeline** (etapas encadenadas de
curación), **Factory + inyección de dependencias** (`config/container.py` es el
único módulo que conoce los adaptadores concretos) y **Repository** para el
acceso a artefactos.

## Puesta en marcha

Dos entornos, a propósito. El núcleo corre en **3.11+**; el stack de anomalib
exige **3.12**.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev,media]"   # 'media' = OpenCV + imagehash
pytest
```

Para los experimentos con modelos reales, un entorno aparte sobre Python 3.12:

```bash
py -3.12 -m venv .venv312
.venv312\Scripts\activate
pip install -e ".[dev,media,ml]"
pytest
```

Sin el extra `ml`, los tests que entrenan un modelo se saltan y todo lo demás se
verifica igual. Sin `media`, también se saltan los de los adaptadores y el núcleo
de curación se sigue verificando completo: esa es la propiedad que da la
arquitectura hexagonal.

Verificación completa, la misma que ejecuta el CI:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

Ganchos de pre-commit:

```bash
pip install pre-commit && pre-commit install
```

## Capturar el dataset propio (E3/E4)

[ml/capture/capturar.py](ml/capture/capturar.py) graba un clip por pieza física
según el protocolo de captura v2: automatismos de la cámara fijados, FFV1 sin
pérdida, ID de pieza en el nombre y control de calidad al terminar cada clip.

```bash
cd D:\tesis
backend\.venv\Scripts\activate

# 1. Encuadre, iluminación y qué controles aceptó el driver
python ml\capture\capturar.py --clase tapas --preview
#    -> revisa ml\data\propio\tapas\preview.png antes de seguir

# 2. Una pieza por invocación; el ID se asigna solo (ok-000, ok-001, ...)
python ml\capture\capturar.py --clase tapas --etiqueta normal
python ml\capture\capturar.py --clase tapas --etiqueta defecto

# 3. Progreso frente al objetivo del protocolo (30 normales, 15 con defecto)
python ml\capture\capturar.py --clase tapas --resumen
```

Si un clip termina con avisos (exposición oscilante, recorte, deriva de
iluminación o de enfoque respecto del primer clip), **repítelo en el momento**:
cuesta quince segundos. El aviso de exposición oscilante importa especialmente:
`VideoCapture.set()` puede aceptar la orden de fijar la exposición sin aplicarla,
y medirla sobre los frames es la única forma de saberlo.

## Reglas del proyecto que no se negocian

1. **El orden de operaciones de la curación es fijo:**
   `extraer → agrupar por pieza → particionar por grupo → deduplicar dentro del split`.
   Deduplicar antes de particionar elimina de test los frames que duplican a los de
   train: oculta la fuga en lugar de corregirla.

2. **El umbral de decisión se calibra en `val`, jamás en `test`.** Elegir el umbral
   mirando el test es ajustar un parámetro sobre el test. En una tesis sobre fuga de
   información sería un defecto fatal. Ver sección 7 del protocolo.

3. **`split_naive_by_frame` existe como baseline del experimento, no como opción.**
   Produce fuga por construcción. Nunca debe generar una métrica reportada como válida.

4. **Los manifiestos de partición se archivan como lista explícita de rutas**, no
   como una semilla. Reproducir un resultado no debe depender de que la
   implementación del generador aleatorio no haya cambiado.

5. **El dataset no se versiona; los manifiestos sí.**

6. **Los frames se guardan en PNG, sin pérdida.** Un artefacto de compresión JPEG
   alrededor de un borde es indistinguible de una rayadura o una mancha, que es
   justo lo que el modelo debe detectar.

7. **Las fuentes en vivo fijan autofoco, autoexposición y balance de blancos**, y
   `applied_controls` reporta cuáles aceptó el driver. `VideoCapture.set()` falla
   en silencio según backend y sistema operativo: dar por hecho que se aplicaron
   invalidaría la captura sin que nadie se entere.

8. **anomalib no decide nada sobre los datos ni sobre las métricas.** Sus
   datamodules reparticionan por su cuenta (`test_split_mode`, `val_split_mode`)
   y su umbral adaptativo puede acabar calculándose sobre el test. Aquí el
   manifiesto se inyecta directamente y las métricas se calculan en
   `application/evaluation/`, con AUROC propio validado contra scikit-learn.
   Los tests de `TestNoReparticiona` son la garantía y deben ejecutarse tras
   cada actualización del stack de ML.

9. **Los scores se emparejan con los frames por ruta, nunca por posición.**
   anomalib ordena las muestras alfabéticamente al asignarlas; emparejar por
   índice asignaría el score de un frame a otro y produciría un AUROC plausible
   y falso.

## Estado

- [x] Fase 0 — esqueleto, CI, puertos del dominio
- [x] Núcleo de curación: muestreo, deduplicación, partición anti-fuga, fuga controlada
- [x] Adaptadores: hashing (imagehash), video (OpenCV), almacenamiento, motor ML (anomalib)
- [x] Capa de evaluación: AUROC (validado contra scikit-learn), umbral en val, Wilcoxon, Holm, equivalencia
- [x] **E1 — validación de la integración: PASA.** PatchCore-1% reproduce el paper en 3/3 categorías dentro de ±0,32 pp
- [x] **E2 — dosis-respuesta de fuga, n=10 (primario):** Δ = +2,83 pp a λ=20 %, IC95 excluye el cero; p Holm = 0,053, subpotenciado
- [ ] E2 — réplica extendida (6 dosis × 20 semillas), en curso
- [x] Herramienta de captura con control de calidad por clip
- [ ] Captura del dataset propio (protocolo v2: ≥30 piezas normales y ≥15 con defecto por clase)
- [ ] E3, E4 — experimentos sobre video real
- [ ] Inferencia en vivo, API, frontend no-code
