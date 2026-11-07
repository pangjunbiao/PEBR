"""Metadata-only checks and declared result coverage before corpus experiments."""
from __future__ import annotations

from pathlib import Path

from ..config import load_configuration, read_yaml, within_project
from ..runtime.artifacts import file_hash
from ..schemas import ConfigurationError

RESULTS_MAP = (
    {"id": "table_1", "name": "Dataset and evaluation scope", "area": "data"},
    {"id": "table_2", "name": "Two-corpus topic quality across K", "area": "topic_quality"},
    {"id": "table_3", "name": "Transportation complaint retrieval", "area": "transportation"},
    {"id": "table_4", "name": "Controlled borrowing comparisons", "area": "ablations"},
    {"id": "figure_1", "name": "Corpus-size effects", "area": "corpus_size"},
    {"id": "figure_2", "name": "Service-category retrieval differences", "area": "transportation"},
)


def validate_profiles(root: Path) -> dict:
    overlays = []
    for folder in ("runtime", "methods", "models", "evaluation", "studies", "experiments", "outputs"):
        for path in sorted((root / "configs" / folder).glob("*.yaml")):
            relative = path.relative_to(root).as_posix()
            loaded = load_configuration(root, relative)
            overlays.append({"path": relative, "configuration_sha256": loaded.sha256,
                             "executable_in_current_phase": loaded.values["experiment"]["stage"] in ('execution_foundation','corpus_preparation','borrowing_rule','borrowing_audit','model_smoke','topic_benchmark','full_study','transportation_retrieval')})
    data_descriptors = []
    expected = {"dataset_id", "role", "source_url", "raw_file", "annotation_use", "download_stage"}
    for name in ("indian_transport", "dabiri_traffic"):
        path = root / "configs/data" / (name + ".yaml")
        descriptor = read_yaml(path)
        if set(descriptor) != expected or any(not isinstance(v, str) for v in descriptor.values()):
            raise ConfigurationError("Unexpected or missing dataset descriptor field: " + str(path))
        if descriptor["dataset_id"] != name or descriptor["download_stage"] != "corpus_preparation":
            raise ConfigurationError("Dataset identity/stage mismatch in " + str(path))
        raw = within_project(root, descriptor["raw_file"])
        present = raw.is_file() and raw.stat().st_size > 0
        descriptor.update(present=present, bytes=raw.stat().st_size if present else None,
                          sha256=file_hash(raw) if present else None)
        data_descriptors.append(descriptor)
    return {"validated_overlay_count": len(overlays), "overlays": overlays,
            "datasets": data_descriptors, "planned_main_results": list(RESULTS_MAP),
            "model_performance_computed": False}


def readiness(environment: dict) -> dict:
    packages = environment["watched_packages"]
    required_next = ("numpy", "scipy", "scikit-learn", "tokenizers", "huggingface-hub")
    missing = [name for name in required_next if packages.get(name) is None]
    if not packages.get("onnxruntime") and not packages.get("onnxruntime-gpu"):
        missing.append("onnxruntime or onnxruntime-gpu (one backend, selected after CUDA audit)")
    warnings = []
    if packages.get("onnxruntime") and packages.get("onnxruntime-gpu"):
        warnings.append("Both ONNX Runtime distributions are installed; shared-module ownership needs review before encoder setup")
    probe = environment.get("probe", {})
    if "probe_error" in probe:
        warnings.append(probe["probe_error"])
    if probe.get("torch", {}).get("cuda_smoke", {}).get("status") == "failed":
        warnings.append("PyTorch CUDA smoke check failed; inspect environment.json")
    if packages.get("torch") and not probe.get("torch", {}).get("import_ok"):
        warnings.append("Installed PyTorch did not import successfully; inspect its recorded error before GPU use")
    if (packages.get("onnxruntime") or packages.get("onnxruntime-gpu")) and not probe.get("onnxruntime", {}).get("import_ok"):
        warnings.append("Installed ONNX Runtime did not import successfully; inspect its recorded error before encoder use")
    for name, result in probe.get("imports", {}).items():
        if not result.get("ok"):
            warnings.append(name + " import is missing or broken; check before the corpus phase")
    if not environment.get("device_selection", {}).get("torch_cuda_execution_verified", False):
        warnings.append("No PyTorch GPU execution was verified in this run; the foundation can still be checked on CPU")
    return {"foundation_scope": "Execution/configuration verification only",
            "next_phase": "Corpus and Evidence Preparation", "missing_next_phase_packages": missing,
            "warnings": warnings, "heavy_packages_modified": False,
            "encoder_session_verified": False, "datasets_downloaded": False,
            "research_performance_evaluated": False}
