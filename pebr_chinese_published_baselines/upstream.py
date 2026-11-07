"""Fetch only the pinned model files and isolate their Python namespaces."""
from hashlib import sha256
import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from urllib.request import urlopen
from urllib.error import URLError

MODEL_ENTRY = {"encot": ("NewMethod", "NewMethod"), "fastopic": ("_fastopic", "fastopic"),
               "neuromax": ("NeuroMax", "NeuroMax")}


def load_upstream(cache, model, spec, local_root=None):
    raw_dir = cache / model / spec["commit"] / "original"
    run_dir = raw_dir.parent / "runtime"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for remote_path, expected in spec["files"].items():
        name = Path(remote_path).name
        path = raw_dir / name
        if path.exists():
            content = path.read_bytes()
        elif local_root is not None:
            content = (local_root / model / remote_path).read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/{spec["repository"]}/{spec["commit"]}/{remote_path}'
            try:
                with urlopen(url, timeout=30) as response:
                    content = response.read(1_000_001)
            except (URLError, TimeoutError) as exc:
                raise OSError(f"Could not fetch {model} source file {name}. Check internet access "
                              "or use --source-root with the pinned repository checkouts.") from exc
        if sha256(content).hexdigest() != expected:
            raise ValueError(f"Upstream source hash mismatch for {model}/{name}: {path}")
        if not path.exists():
            with TemporaryDirectory(dir=raw_dir) as stage:
                saved = Path(stage) / name
                saved.write_bytes(content)
                saved.replace(path)
        source = content.decode("utf-8").replace("\r\n", "\n")
        if model == "neuromax" and name == "NeuroMax.py":
            # The paper uses precomputed MiniLM vectors. The model's ST import is unused.
            # The tensor device change preserves the GR computation on CPU and GPU.
            old = "torch.ones(self.num_topics, self.num_topics).cuda()"
            if source.count(old) != 1 or source.count("import sentence_transformers\n") != 1:
                raise ValueError("The NeuroMax compatibility patch no longer matches")
            source = source.replace("import sentence_transformers\n", "")
            source = source.replace(old, "torch.ones(self.num_topics, self.num_topics, device=self.topic_embeddings.device)")
        target = run_dir / name
        with TemporaryDirectory(dir=run_dir) as stage:
            saved = Path(stage) / name
            saved.write_text(source, encoding="utf-8", newline="\n")
            saved.replace(target)
    namespace = f"_pebr_benchmark_{model}_{spec['commit'][:10]}"
    package = ModuleType(namespace)
    package.__path__ = [str(run_dir.resolve())]
    sys.modules[namespace] = package
    module, entry = MODEL_ENTRY[model]
    return getattr(importlib.import_module(f"{namespace}.{module}"), entry)
