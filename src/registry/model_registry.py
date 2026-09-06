"""The model registry: which model answers which task, and what is true of it now.

configs/model_registry.yaml declares only a model's *identity* -- name, version,
task, architecture, where its checkpoint lives, whether the serving path may use
it. This module hydrates that declaration against the filesystem on load, so
everything observable (does the checkpoint exist, what is its fingerprint, when
was it written, what classes does it predict, what did it score) is measured
rather than remembered. A written-down copy of any of that goes stale the moment
a model is retrained, which is precisely how /api/system-info came to advertise a
3-class dataset months after a 6-variety model replaced it.

Two properties are worth naming, because they are the point of the file:

`resolve(task)` is the only sanctioned way to choose a model. Callers ask for a
capability -- "who does quality?" -- and get either the one model declared to
serve it or TaskUnavailableError. They never name a checkpoint. Before this, the
answer to "which model grades quality?" was spelled out independently in the
pipeline, the backend and the frontend, and the three drifted apart.

`metrics_binding` reports whether a model's evaluation numbers can be proved to
belong to the checkpoint sitting on disk. Training writes the checkpoint's
SHA-256 into its metrics file; the registry recomputes it and compares. Retrain
without re-evaluating and the binding goes "stale" instead of the page quietly
reporting the old model's score for the new model. Metrics files written before
this existed report "unrecorded" -- not proof of agreement, and not a claim of
one.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.config import PROJECT_ROOT, load_config

REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"

# Fingerprinting a multi-hundred-MB checkpoint on every request would make the
# system-info page slow enough to notice, so hashes are cached against the file's
# identity (size + mtime). A rewritten checkpoint changes both and is re-hashed.
_HASH_CACHE: dict[tuple[str, int, float], str] = {}

# Same trick, same reason: reading a class list means torch.load-ing the whole
# checkpoint. Keyed on file identity so a refresh re-reads only what changed,
# which is what lets /api/system-info re-probe on every request without paying
# for it. The sentinel distinguishes "cached: this artefact has no class list"
# from "not cached yet".
_CLASSES_CACHE: dict[tuple[str, int, float], list[str] | None] = {}


class UnknownModelError(KeyError):
    """Asked for a model key the registry does not declare."""


class TaskUnavailableError(Exception):
    """No model is both declared to serve this task and present on this machine.

    Raised rather than falling back to a model trained for something else: a task
    with no model is answered "unavailable", never approximated.
    """


def _sha256(path: Path) -> str | None:
    try:
        st = path.stat()
    except OSError:
        return None
    cache_key = (str(path), st.st_size, st.st_mtime)
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    digest = h.hexdigest()
    _HASH_CACHE[cache_key] = digest
    return digest


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _classes_from_checkpoint(path: Path, task: str) -> list[str] | None:
    """Read the class list out of the checkpoint itself.

    The alternative -- listing classes in the registry YAML -- is the same
    mistake as hard-coding metrics: retrain with a different label set and the
    declared list silently misdescribes the model. Returns None when the artefact
    genuinely carries no class list (the contrastive encoder has no head at all,
    and the YOLO weights are read by ultralytics, not us).
    """
    if path.suffix != ".pt":
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    cache_key = (str(path), st.st_size, st.st_mtime)
    if cache_key in _CLASSES_CACHE:
        return _CLASSES_CACHE[cache_key]
    classes = _read_classes(path)
    _CLASSES_CACHE[cache_key] = classes
    return classes


def _read_classes(path: Path) -> list[str] | None:
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a checkpoint we cannot read is not fatal here
        return None
    if not isinstance(state, dict):
        return None
    if "variety_classes" in state and "quality_classes" in state:
        # the unified model answers two questions; report both label sets
        return [
            *(f"variety: {c}" for c in state["variety_classes"]),
            *(f"quality: {c}" for c in state["quality_classes"]),
        ]
    for key in ("classes", "channel_names"):
        value = state.get(key)
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    return None


@dataclass
class ModelRecord:
    """One registry entry: what was declared, plus what the disk actually says."""

    # ---- declared (configs/model_registry.yaml)
    key: str
    name: str
    version: str
    architecture: str
    task: str
    serves: list[str]
    role: str
    checkpoint: str  # absolute, resolved from the checkpoints root
    dataset: str | None
    trained_by: str | None
    input: dict
    requires_resolution: dict | None
    label_provenance: str | None
    metrics_file: str | None  # absolute, or None if the model has no evaluation
    # Classes the model holds but is never permitted to assert, mapped to the
    # measured reason. Declared rather than inferred, and published rather than
    # kept in the YAML, because a served model's silences are part of its
    # specification: a reader shown seven classes and no note has been told the
    # model can assert seven.
    withheld_classes: list[str] | None = None

    # ---- measured (filesystem, at load time)
    present: bool = False
    size_bytes: int | None = None
    trained_at: str | None = None
    fingerprint: str | None = None
    classes: list[str] | None = None
    metrics: dict | None = None
    metrics_binding: str = "missing"
    note: str | None = None

    @property
    def served(self) -> bool:
        """Declared usable by the serving path for at least one task.

        Independent of `present`: a synthetic-label model stays unserved even
        with its checkpoint sitting right there, which is the whole reason
        `serves` is declared instead of inferred from file existence.
        """
        return bool(self.serves)

    @property
    def status(self) -> str:
        if not self.present:
            return "missing"
        return "ready" if self.served else "available"

    @property
    def available(self) -> bool:
        """Usable right now: allowed to serve, and actually on this machine."""
        return self.served and self.present

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "task": self.task,
            "serves": list(self.serves),
            "role": self.role,
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "trained_by": self.trained_by,
            "input": dict(self.input),
            "requires_resolution": self.requires_resolution,
            "label_provenance": self.label_provenance,
            "status": self.status,
            "served": self.served,
            "present": self.present,
            "size_bytes": self.size_bytes,
            "trained_at": self.trained_at,
            "fingerprint": self.fingerprint,
            "classes": self.classes,
            "metrics_file": self.metrics_file,
            "metrics_binding": self.metrics_binding,
            "note": self.note,
            "withheld_classes": self.withheld_classes,
        }


@dataclass
class Registry:
    version: int
    tasks: dict[str, str]
    models: dict[str, ModelRecord] = field(default_factory=dict)

    # ---------------------------------------------------------------- lookups
    def get(self, key: str) -> ModelRecord:
        try:
            return self.models[key]
        except KeyError as e:
            raise UnknownModelError(
                f"'{key}' is not declared in configs/model_registry.yaml. "
                f"Known models: {sorted(self.models)}"
            ) from e

    def all(self) -> list[ModelRecord]:
        """Registry order, which is declaration order: served models first."""
        return list(self.models.values())

    def for_task(self, task: str) -> list[ModelRecord]:
        """Every model declared to serve this task, present or not."""
        self._check_task(task)
        return [m for m in self.models.values() if task in m.serves]

    def resolve(self, task: str) -> ModelRecord:
        """The model that answers `task`, or TaskUnavailableError.

        The only sanctioned way to pick a model. Never falls back to a model
        declared for a different task -- an unanswerable question is reported
        unanswerable.
        """
        self._check_task(task)
        candidates = self.for_task(task)
        if not candidates:
            raise TaskUnavailableError(
                f"No model in the registry serves '{task}'. This capability is "
                f"not available; it must not be approximated with another model."
            )
        ready = [m for m in candidates if m.present]
        if not ready:
            missing = ", ".join(f"{m.key} ({m.checkpoint})" for m in candidates)
            hint = candidates[0].trained_by
            raise TaskUnavailableError(
                f"'{task}' is served by {missing}, but no checkpoint is present on "
                f"this machine." + (f" Train it with: {hint}" if hint else "")
            )
        if len(ready) > 1:
            raise TaskUnavailableError(
                f"'{task}' is ambiguous: {[m.key for m in ready]} all claim to serve "
                f"it. Exactly one model must be declared per served task."
            )
        return ready[0]

    def can_serve(self, task: str) -> bool:
        try:
            self.resolve(task)
        except TaskUnavailableError:
            return False
        return True

    def served_tasks(self) -> list[str]:
        return sorted(t for t in self.tasks if self.can_serve(t))

    def _check_task(self, task: str) -> None:
        if task not in self.tasks:
            raise UnknownModelError(
                f"'{task}' is not a declared task. Known tasks: {sorted(self.tasks)}"
            )


def _hydrate(spec: dict, ckpt_root: Path, metrics_root: Path) -> ModelRecord:
    ckpt = ckpt_root / spec["checkpoint"]
    metrics_name = spec.get("metrics")
    metrics_path = metrics_root / f"{metrics_name}.json" if metrics_name else None

    record = ModelRecord(
        key=spec["key"],
        name=spec["name"],
        version=str(spec.get("version", "0.0.0")),
        architecture=spec["architecture"],
        task=spec["task"],
        serves=list(spec.get("serves") or []),
        role=spec["role"],
        checkpoint=str(ckpt),
        dataset=spec.get("dataset"),
        trained_by=spec.get("trained_by"),
        input=dict(spec.get("input") or {}),
        requires_resolution=spec.get("requires_resolution"),
        label_provenance=spec.get("label_provenance"),
        metrics_file=str(metrics_path) if metrics_path else None,
        withheld_classes=(list(spec["withheld_classes"])
                          if spec.get("withheld_classes") else None),
    )

    # A YOLO run directory counts as present only if the weights file is there.
    record.present = ckpt.exists()
    if record.present:
        st = ckpt.stat()
        record.size_bytes = st.st_size
        record.trained_at = (
            datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        )
        record.fingerprint = _sha256(ckpt)
        record.classes = _classes_from_checkpoint(ckpt, spec["task"])

    if metrics_path is not None:
        record.metrics = _read_json(metrics_path)
        if record.metrics:
            # the training script's own caveat, carried through verbatim rather
            # than being re-worded downstream
            record.note = record.metrics.get("label_note") or record.metrics.get("note")
            record.label_provenance = (
                record.metrics.get("label_provenance") or record.label_provenance
            )

    # Computed unconditionally: a model that declares no metrics file at all must
    # read "missing", not "unrecorded". "unrecorded" is a statement about an
    # evaluation that exists but cannot be tied to a checkpoint, and claiming it
    # where no evaluation exists would overstate what is on disk.
    record.metrics_binding = _binding(record)
    return record


def _binding(record: ModelRecord) -> str:
    """Do these metrics demonstrably belong to the checkpoint on disk?

    "verified"   - the metrics file records a checkpoint hash and it matches.
    "stale"      - it records one and it does NOT match: the model was retrained
                   without re-evaluating, so the numbers describe a different
                   artefact and must not be reported as this one's.
    "unrecorded" - no hash was written. Says nothing either way; it is not
                   evidence of agreement.
    "missing"    - no metrics file, or no checkpoint to compare against.
    """
    if not record.metrics or not record.present:
        return "missing"
    recorded = record.metrics.get("checkpoint_sha256")
    if not recorded:
        return "unrecorded"
    return "verified" if recorded == record.fingerprint else "stale"


def load_registry(config_path: str | None = None, registry_path: str | None = None) -> Registry:
    path = Path(registry_path) if registry_path else REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = load_config(config_path)
    ckpt_root = Path(cfg["paths"]["checkpoints"])
    metrics_root = Path(cfg["paths"]["metrics"])
    if not ckpt_root.is_absolute():
        ckpt_root = PROJECT_ROOT / ckpt_root
    if not metrics_root.is_absolute():
        metrics_root = PROJECT_ROOT / metrics_root

    tasks = dict(raw.get("tasks") or {})
    models: dict[str, ModelRecord] = {}
    for spec in raw.get("models") or []:
        key = spec["key"]
        if key in models:
            raise ValueError(f"Duplicate model key in the registry: '{key}'")
        undeclared = [t for t in (spec.get("serves") or []) if t not in tasks]
        if undeclared:
            raise ValueError(
                f"Model '{key}' claims to serve undeclared task(s) {undeclared}. "
                f"Add them to the `tasks:` block first."
            )
        if spec["task"] not in tasks:
            raise ValueError(f"Model '{key}' has undeclared task '{spec['task']}'.")
        models[key] = _hydrate(spec, ckpt_root, metrics_root)

    registry = Registry(version=int(raw.get("version", 1)), tasks=tasks, models=models)

    # Fail loudly at load, not at the first request: two models serving one task
    # means the choice would depend on iteration order.
    for task in tasks:
        serving = [m.key for m in registry.for_task(task)]
        if len(serving) > 1:
            raise ValueError(
                f"Task '{task}' is claimed by {serving}. Exactly one model may "
                f"serve a task; the others should declare serves: []."
            )
    return registry


@lru_cache(maxsize=1)
def _cached_registry() -> Registry:
    return load_registry()


def get_registry(refresh: bool = False) -> Registry:
    """Process-wide registry. `refresh=True` re-reads the filesystem, which is what
    a long-running server wants after a model is retrained underneath it."""
    if refresh:
        _cached_registry.cache_clear()
    return _cached_registry()


def checkpoint_sha256(path: str | os.PathLike) -> str | None:
    """Public helper so training scripts can stamp their metrics file with the
    fingerprint of the checkpoint they just evaluated."""
    return _sha256(Path(path))
