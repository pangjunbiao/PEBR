"""Safe YAML overlays with strict field validation and stable fingerprints."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .schemas import ConfigurationError, LoadedConfiguration, validate_config


def within_project(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigurationError("Path must stay inside the project: " + str(value))
    return resolved


def deep_merge(base: dict, overlay: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def canonical_hash(value) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ConfigurationError("Configuration is not finite, acyclic JSON-compatible data") from error
    return hashlib.sha256(encoded).hexdigest()


def read_yaml(path: Path) -> dict:
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigurationError("YAML mapping keys must be strings: " + str(path))
            if key in result:
                raise ConfigurationError("Duplicate YAML key '%s' in %s" % (key, path))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = yaml.load(text, Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("Cannot read YAML %s: %s" % (path, error)) from error
    if not isinstance(data, dict) or not data:
        raise ConfigurationError("YAML file is empty or is not a mapping: " + str(path))
    canonical_hash(data)
    return data


def load_configuration(root: Path, overlay: str | None = None,
                       device: str | None = None, mode: str | None = None) -> LoadedConfiguration:
    root = root.resolve()
    sources = []
    def read(relative, stack):
        path = within_project(root, relative)
        if path in stack:
            raise ConfigurationError("Cyclic YAML extends: " + " -> ".join(str(p) for p in (*stack, path)))
        data = read_yaml(path)
        parents = data.pop("extends", [])
        if not isinstance(parents, list) or any(not isinstance(x, str) for x in parents):
            raise ConfigurationError("extends must be a list of project-relative YAML paths")
        result = {}
        for parent in parents:
            result = deep_merge(result, read(parent, (*stack, path)))
        relative_name = path.relative_to(root).as_posix()
        if relative_name not in sources:
            sources.append(relative_name)
        return deep_merge(result, data)
    default = within_project(root, "configs/project.yaml")
    values = read(default, ())
    if overlay and within_project(root, overlay) != default:
        values = deep_merge(values, read(overlay, ()))
    if device is not None:
        values["runtime"]["device"] = device
    if mode is not None:
        values["experiment"]["mode"] = mode
    effective = validate_config(values)
    return LoadedConfiguration(values, effective, tuple(sources), canonical_hash(values))
