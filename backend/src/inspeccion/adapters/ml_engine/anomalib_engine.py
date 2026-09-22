"""Adaptador del motor ML sobre anomalib.

Implementa el puerto `MLEngine` mediante el patron **Strategy**: PatchCore y
PaDiM se seleccionan por configuracion y el resto del sistema no distingue uno de
otro. Es el intercambio de adaptador que el criterio de exito del prototipo pide
poder demostrar.

Dos limites deliberados de este adaptador:

1. **No decide como se reparten los datos.** Recibe un `SplitManifest` ya
   construido y lo inyecta sin que anomalib pueda reasignar nada
   (ver `manifest_dataset`).
2. **No calcula las metricas que se reportan.** Devuelve scores crudos; el AUROC,
   el umbral y las pruebas estadisticas viven en `application.evaluation`, con
   codigo propio y validado contra scikit-learn. Asi las cifras de la tesis no
   dependen de los defaults de la libreria —en particular del umbral adaptativo,
   que segun como se configure el flujo puede acabar calculandose sobre el
   conjunto de test.

Se usa el `Trainer` de Lightning directamente y no el `Engine` de anomalib. Los
modelos de anomalib son `LightningModule`, asi que no se pierde nada, y el
`Engine` aporta dos cosas que aqui estorban: convenciones de workspace que crean
enlaces simbolicos —lo que falla en Windows sin privilegios de administrador y
haria el proyecto no reproducible en la maquina del tesista— y callbacks de
metricas, umbral y visualizacion que esta tesis calcula por su cuenta. anomalib
aporta el modelo y el dataset; la orquestacion es de Lightning.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from inspeccion.adapters.ml_engine.manifest_dataset import build_datamodule, build_dataset
from inspeccion.domain.models import FrameRef, Split, SplitManifest

CHECKPOINT_NAME = "model.ckpt"


class AnomalyModel(enum.StrEnum):
    """Estrategias de deteccion disponibles."""

    PATCHCORE = "patchcore"
    """Principal. Memory bank de parches nominales; resuelve el arranque en frio
    con solo imagenes buenas, que es el caso del MVP."""

    PADIM = "padim"
    """Baseline liviano. Gaussianas multivariadas por parche sobre ResNet18."""


class TrainingError(RuntimeError):
    """El entrenamiento no produjo un modelo utilizable."""


@dataclass(frozen=True, slots=True)
class AnomalibMLEngine:
    """Adaptador que satisface `MLEngine`.

    Args:
        model: estrategia de deteccion.
        backbone: extractor de caracteristicas. `None` deja el de anomalib
            (`wide_resnet50_2` para PatchCore, `resnet18` para PaDiM). Es
            configurable porque reproducir una cifra publicada exige replicar la
            configuracion del paper: PaDiM es muy sensible al backbone y sus
            resultados de referencia usan `wide_resnet50_2`, no `resnet18`.
        layers: capas de las que se extraen las caracteristicas. `None` deja las
            del modelo.
        n_features: dimensiones aleatorias que PaDiM retiene (550 en el paper).
            Se ignora para PatchCore.
        coreset_sampling_ratio: fraccion del memory bank que PatchCore conserva.
            El consumo de memoria escala con el numero de frames nominales, que es
            la razon operativa —ademas de la cientifica— para deduplicar antes de
            entrenar. Tambien cambia el resultado: el paper reporta 96.4 de AUROC
            en `screw` con 1% y 98.1 con 25%, asi que la cifra solo es comparable
            contra una referencia de la misma configuracion.
        num_workers: 0 en Windows. Cada worker relanza el proceso y para estos
            volumenes el arranque cuesta mas de lo que ahorra.
    """

    model: AnomalyModel = AnomalyModel.PATCHCORE
    backbone: str | None = None
    layers: tuple[str, ...] | None = None
    n_features: int | None = None
    coreset_sampling_ratio: float = 0.1
    batch_size: int = 32
    num_workers: int = 0
    accelerator: str = "auto"
    seed: int | None = None

    def train(self, manifest: SplitManifest, work_dir: Path) -> Path:
        """Entrena con el split TRAIN del manifiesto y devuelve el checkpoint.

        Nunca toca TEST: el datamodule lo expone, pero `fit` solo consume el
        dataloader de entrenamiento y el de validacion.
        """
        if not manifest.frames(Split.TRAIN):
            raise TrainingError("el split de entrenamiento del manifiesto esta vacio")

        work_dir.mkdir(parents=True, exist_ok=True)
        model = self._build_model()
        # E1 usa la particion oficial de MVTec AD, que no define validacion. El
        # memory bank de PatchCore se construye igual: `MemoryBankMixin` lo arma
        # en `on_train_epoch_end` ademas de en `on_validation_start`.
        trainer = self._build_trainer(work_dir, has_validation=bool(manifest.frames(Split.VAL)))

        trainer.fit(
            model=model,
            datamodule=build_datamodule(
                manifest,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                num_workers=self.num_workers,
            ),
        )

        checkpoint = work_dir / CHECKPOINT_NAME
        trainer.save_checkpoint(checkpoint)
        if not checkpoint.is_file():
            raise TrainingError(f"el entrenamiento no escribio el checkpoint en {checkpoint}")
        return checkpoint

    def scores(self, model_path: Path, frames: Sequence[FrameRef]) -> npt.NDArray[np.float64]:
        """Score de anomalia por frame, sin umbralizar.

        Devolver el score crudo y no una decision es lo que permite calcular el
        AUROC —independiente del umbral— y calibrar el umbral sobre validacion
        en un paso aparte y controlado.
        """
        if not frames:
            raise ValueError("no hay frames que puntuar")

        from torch.utils.data import DataLoader

        model = self._load(model_path)
        trainer = self._build_trainer(model_path.parent)

        dataset = build_dataset(frames, Split.TEST)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
        )

        predictions = trainer.predict(model=model, dataloaders=loader)
        if predictions is None:
            raise TrainingError("anomalib no devolvio predicciones")

        return self._align(predictions, frames)

    def evaluate(self, model_path: Path, frames: Sequence[FrameRef]) -> dict[str, float]:
        """AUROC a nivel de imagen sobre `frames`.

        Se calcula con la implementacion propia, validada contra scikit-learn, y
        no con el evaluador de anomalib: las cifras de la tesis no deben depender
        de los defaults de la libreria de entrenamiento.
        """
        from inspeccion.application.evaluation.metrics import auroc
        from inspeccion.domain.models import Label

        values = self.scores(model_path, frames)
        labels = np.array([frame.label is Label.DEFECT for frame in frames], dtype=np.bool_)
        return {"image_auroc": auroc(values, labels)}

    def export_onnx(
        self, model_path: Path, destination: Path, *, image_size: tuple[int, int] = (256, 256)
    ) -> Path:
        """Exporta a ONNX para la inferencia en vivo (RF-13, RNF-01).

        Se llama a `torch.onnx.export` directamente en vez de a
        `ExportMixin.to_onnx` de anomalib. Aquel pasa `dynamic_axes`, argumento
        que el exportador basado en dynamo —el activo por defecto desde torch
        2.14— rechaza, abortando la exportacion. Aqui se fuerza el exportador
        clasico con `dynamo=False`.

        `image_size` se fija en vez de dejar la forma dinamica: el runtime de
        inferencia se beneficia de una forma estatica y la captura tiene
        resolucion constante por protocolo.
        """
        import torch

        destination.parent.mkdir(parents=True, exist_ok=True)
        model = self._load(model_path).eval()
        sample = torch.zeros(1, 3, *image_size, device=model.device)

        # Nombres de salida tomados de una pasada real, como hace anomalib: el
        # modelo devuelve una namedtuple cuyos campos opcionales pueden venir a
        # None segun la configuracion del post-procesador.
        with torch.no_grad():
            output_names = [
                name for name, value in model(sample)._asdict().items() if value is not None
            ]

        # `dynamo=False` fuerza el exportador clasico. El de torch 2.14, activo
        # por defecto, rechaza el `dynamic_axes` que usa `ExportMixin.to_onnx` de
        # anomalib y aborta la exportacion.
        torch.onnx.export(
            model,
            (sample,),
            str(destination),
            opset_version=14,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes={"input": {0: "batch_size"}},
            dynamo=False,
        )

        if not destination.is_file():
            raise TrainingError(f"la exportacion no escribio {destination}")
        return destination

    # --- internos ---------------------------------------------------------

    @staticmethod
    def _align(predictions: Sequence[Any], frames: Sequence[FrameRef]) -> npt.NDArray[np.float64]:
        """Reordena los scores para que correspondan uno a uno con `frames`.

        No se puede concatenar los lotes y asumir que el orden coincide con el de
        entrada: el setter de `samples` de anomalib hace
        `sort_values(by="image_path")`, asi que el dataset se recorre en orden
        alfabetico de ruta, no en el orden en que se le paso la lista. Emparejar
        por posicion asignaria el score de un frame a otro y contaminaria en
        silencio todas las metricas de la tesis.
        """
        by_path: dict[str, float] = {}
        for batch in predictions:
            scores = np.asarray(batch.pred_score, dtype=np.float64).ravel()
            paths = list(batch.image_path)
            if len(paths) != scores.size:
                raise TrainingError(
                    f"el lote trae {len(paths)} rutas y {scores.size} scores; no se puede emparejar"
                )
            by_path.update(zip(paths, scores.tolist(), strict=True))

        missing = [frame.path for frame in frames if frame.path not in by_path]
        if missing:
            raise TrainingError(
                f"faltan predicciones para {len(missing)} frames, el primero es {missing[0]!r}"
            )
        return np.array([by_path[frame.path] for frame in frames], dtype=np.float64)

    def _build_model(self) -> Any:
        """Solo se pasan los argumentos declarados: lo que quede en `None` conserva
        el valor por defecto de anomalib, sin que este adaptador lo duplique."""
        from anomalib.models import Padim, Patchcore

        # Sin evaluador ni visualizador: las metricas se calculan en
        # `application.evaluation` con codigo propio, y el visualizador escribia
        # imagenes a disco en cada prediccion sin que nadie las usara.
        opciones: dict[str, Any] = {"evaluator": False, "visualizer": False}
        if self.backbone is not None:
            opciones["backbone"] = self.backbone
        if self.layers is not None:
            opciones["layers"] = list(self.layers)

        if self.model is AnomalyModel.PATCHCORE:
            return Patchcore(coreset_sampling_ratio=self.coreset_sampling_ratio, **opciones)
        if self.n_features is not None:
            opciones["n_features"] = self.n_features
        return Padim(**opciones)

    def _load(self, model_path: Path) -> Any:
        from anomalib.models import Padim, Patchcore

        if not model_path.is_file():
            raise FileNotFoundError(f"no existe el checkpoint: {model_path}")
        cls = Patchcore if self.model is AnomalyModel.PATCHCORE else Padim
        return cls.load_from_checkpoint(model_path)

    def _build_trainer(self, work_dir: Path, *, has_validation: bool = True) -> Any:
        from lightning.pytorch import Trainer, seed_everything

        if self.seed is not None:
            seed_everything(self.seed, workers=True)

        return Trainer(
            default_root_dir=work_dir,
            logger=False,
            accelerator=self.accelerator,
            devices=1,
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            limit_val_batches=1.0 if has_validation else 0,
        )
