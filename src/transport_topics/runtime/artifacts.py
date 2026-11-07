"""Atomic records, immutable run directories and compact inspection archives."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
import zipfile

from .logging import now_local, now_utc, run_log_snapshot

OUTPUT_DIRECTORIES = (
    "data/raw/indian_transport", "data/raw/dabiri_traffic",
    "data/processed/indian_transport", "data/processed/dabiri_traffic",
    "data/annotations/indian_transport", "data/annotations/dabiri_traffic",
    "data/manifests/sources", "data/manifests/families", "data/manifests/folds",
    "data/manifests/samples", "data/manifests/queries", "assets/pretrained/encoder",
    "assets/pretrained/word_embeddings", "cache/embeddings", "cache/predictors",
    "cache/candidates", "cache/coherence", "outputs/runs", "outputs/checkpoints",
    "outputs/manifests", "outputs/verification", "outputs/inspection",
    "outputs/aggregated/topic_quality", "outputs/aggregated/transportation",
    "outputs/aggregated/ablations", "outputs/aggregated/sensitivity",
    "outputs/aggregated/corpus_size", "outputs/aggregated/predictive_audit",
    "outputs/aggregated/diagnostics", "outputs/paper/tables", "outputs/paper/figures",
    "outputs/paper/source_data", "outputs/supplement/tables", "outputs/supplement/figures",
    "outputs/supplement/source_data", "outputs/supplement/topic_reports", "logs",
)


def safe_path(root: Path, relative: str | Path) -> Path:
    base = root.resolve()
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Artifact paths must be project-relative")
    result = (base / path).resolve()
    if not result.is_relative_to(base):
        raise ValueError("Artifact path escapes its root: " + str(relative))
    return result


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Windows readers may briefly deny rename/delete sharing. Retry only those
# error codes; filesystem/permission failures on other platforms still fail fast.
_WINDOWS_REPLACE_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1., 1., 1., 1., 1., 1., 1.)
_WINDOWS_REPLACE_ERRORS = frozenset((5, 32, 33))


class ArtifactWriteBlockedError(RuntimeError):
    """A complete payload was preserved after a bounded Windows write retry."""
    def __init__(self, pending, target, error, attempts):
        self.pending_path = Path(pending)
        self.target_path = Path(target)
        self.winerror = getattr(error, 'winerror', None)
        self.attempts = attempts
        super().__init__(
            'Windows still refused the atomic update after %d attempts (WinError %s). '
            'Destination: %s. Any previous destination file was preserved. '
            'The complete new payload remains at: %s. '
            'Close any preview/editor holding this result file, then rerun main.py '
            'normally to reuse completed model/evaluation caches.'
            % (attempts, self.winerror, target, pending))


def _replace_with_retry(source, destination):
    for attempt in range(len(_WINDOWS_REPLACE_DELAYS) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in _WINDOWS_REPLACE_ERRORS:
                raise
            if attempt == len(_WINDOWS_REPLACE_DELAYS):
                raise ArtifactWriteBlockedError(source, destination, error, attempt + 1) from error
            time.sleep(_WINDOWS_REPLACE_DELAYS[attempt])


def atomic_write(path: Path, content: bytes, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite an existing run artifact: " + str(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".writing_", dir=str(path.parent))
    preserve_payload = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # The project execution lock serializes writers to the same output tree.
        _replace_with_retry(temporary, path)
    except ArtifactWriteBlockedError:
        # Bytes have been completely written, flushed, fsynced and closed.
        # Keep them recoverable; never delete/truncate the previous destination.
        preserve_payload = True
        raise
    finally:
        if not preserve_payload and os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value, overwrite: bool = False):
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_write(path, text.encode("utf-8"), overwrite)


def prepare_directories(root: Path) -> list[str]:
    for name in OUTPUT_DIRECTORIES:
        safe_path(root, name).mkdir(parents=True, exist_ok=True)
    return list(OUTPUT_DIRECTORIES)


def source_identity(root: Path) -> dict:
    names = [root / "main.py", root / "pyproject.toml", root / "requirements.txt"]
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("requirements", "*.txt"), ("configs", "*.yaml")):
        names.extend((root / folder).rglob(pattern))
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(set(names)) if p.is_file()}
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": aggregate, "files": files}


class RunStore:
    def __init__(self, root: Path, stage='execution_foundation'):
        self.root = root.resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = stamp + "_" + stage + "_" + uuid.uuid4().hex[:8]
        self.directory = safe_path(root, "outputs/runs/" + self.run_id)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started_utc, self.started_local = now_utc(), now_local()

    def json(self, name, value, overwrite=False):
        write_json(safe_path(self.directory, name), value, overwrite)

    def text(self, name, value, overwrite=False):
        atomic_write(safe_path(self.directory, name), value.encode("utf-8"), overwrite)

    def manifest(self, identity: dict):
        files = {}
        for path in sorted(self.directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Run artifacts cannot be symbolic links")
            if path.is_file() and path.name != "manifest.json":
                files[path.relative_to(self.directory).as_posix()] = {
                    "sha256": file_hash(path), "bytes": path.stat().st_size}
        self.json("manifest.json", {"schema_version": 1, "run_id": self.run_id,
                  "identity": identity, "artifacts": files}, overwrite=True)


def make_inspection(root: Path, run_id: str) -> Path:
    from .checkpointing import validate_run_id
    directory = safe_path(root, "outputs/runs/" + validate_run_id(run_id))
    if not (directory / "status.json").exists():
        raise ValueError("Inspection target has no run status")
    payloads = {}
    allowed = {"status.json", "environment.json", "manifest.json", "config.yaml",
               "effective_settings.json", "source_fingerprints.json", "configuration_sources.json",
               "requirements_check.json", "directory_check.json", "process_settings.json",
               "readiness.json", "resume_check.json", "preflight_plan.json", "verification.json",
               "timing.json", "summary.txt", "report_target.json", "corpus_preparation.json",
               "encoder_audit.json", "corpus_requirements.json", "source_acquisition.json"}
    allowed.update({'preparation_dependency.json', 'predictor_audit.json', 'selected_actions_index.json',
                    'borrowing_rule.json', 'outer_prediction_audit.json'})
    allowed.update({'model_requirements.json','models_index.json','sharing_controls_index.json','model_integration.json'})
    allowed.update({'evaluation_requirements.json','study_plan.json','dataset_evaluation_profile.json',
                    'coherence_references.json','evaluation_progress.json','size_conditions_index.json',
                    'study_models_index.json','study_evaluations_index.json','study_controls_index.json',
                    'evaluation_outputs.json','evaluation_summary.json','size_encoder_audit.json'})
    allowed.update({'reporting_requirements.json','phase5_acceptance.json','method_source_check.json',
                    'topic_report_summary.json','publication_outputs.json','publication_summary.json','results_readout.md'})
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Inspection refuses symbolic links")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if (relative not in allowed and not (relative.startswith('results/') and path.suffix=='.csv')) or path.stat().st_size > 5 * 1024 * 1024:
                continue
            payloads["run/" + relative] = path.read_bytes()
    publication = directory / 'publication_summary.json'
    if publication.is_file():
        record = json.loads(publication.read_text(encoding='utf-8'))
        if record.get('complete'):
            for area, name in (('paper', 'paper_directory'), ('supplement', 'supplement_directory')):
                base = safe_path(root, record[name])
                for folder, suffix in (('tables', '.tex'), ('figures', '.png')):
                    for path in sorted((base / folder).glob('*' + suffix)):
                        if path.is_symlink():
                            raise ValueError('Publication inspection refuses symbolic links')
                        if path.is_file() and path.stat().st_size <= 5 * 1024 * 1024:
                            payloads['publication/' + area + '/' + folder + '/' + path.name] = path.read_bytes()
    payloads["logs/run_log.txt"] = run_log_snapshot(root / "logs/run_log.txt", run_id)
    payloads["README.txt"] = (
        "Proposed transport project inspection bundle\n"
        "No datasets, model weights or environment variables are included.\n"
        "Borrowing-stage bundles include numerical diagnostics and protected predictive audits when available.\n"
        "run/status.json states success/failure; run/readiness.json describes future-stage readiness.\n"
        "Implementation checks and prediction audits do not establish final topic or retrieval superiority.\n").encode()
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    payloads["inspection_manifest.json"] = json.dumps(checksums, indent=2, sort_keys=True).encode()
    target = safe_path(root, "outputs/inspection/" + run_id + "_inspection.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target = target.with_name(target.stem + "_" + uuid.uuid4().hex[:8] + ".zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Inspection archive integrity check failed")
    return target
