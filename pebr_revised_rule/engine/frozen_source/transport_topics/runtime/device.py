"""Isolated package/GPU diagnostics; no model download and no heavy installation."""
from __future__ import annotations

from importlib import metadata
import json
import platform
import shutil
import subprocess
import sys
import time

PROBE_CODE = r'''
import importlib, importlib.util, json, sys
options=json.loads(sys.argv[1])
report={"imports":{}}
for name in ("numpy","scipy","sklearn"):
    try:
        module=importlib.import_module(name)
        report["imports"][name]={"ok":True,"version":getattr(module,"__version__",None)}
    except Exception as error:
        report["imports"][name]={"ok":False,"error":str(error)[:2000]}
torch_report={"import_ok":False,"cuda_available":False,"devices":[],"cuda_smoke":{"status":"not_run"},
              "cpu_smoke":{"status":"not_run"}}
try:
    import torch
    torch_report.update(import_ok=True,version=torch.__version__,build_cuda=torch.version.cuda)
    torch.set_num_threads(options["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cpu=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float32)
    cpu_result=(cpu@cpu).tolist()
    torch_report["cpu_smoke"]={"status":"passed" if cpu_result==[[7.,10.],[15.,22.]] else "failed"}
    available=bool(torch.cuda.is_available())
    torch_report["cuda_available"]=available
    if available:
        for index in range(torch.cuda.device_count()):
            properties=torch.cuda.get_device_properties(index)
            torch_report["devices"].append({"index":index,"name":properties.name,
                "total_memory_bytes":properties.total_memory,"capability":[properties.major,properties.minor]})
        if options["device"]=="cpu":
            torch_report["cuda_smoke"]={"status":"skipped_cpu_policy"}
        elif options["gpu_index"]>=len(torch_report["devices"]):
            torch_report["cuda_smoke"]={"status":"failed","error":"Configured GPU index is not available"}
        else:
            device="cuda:"+str(options["gpu_index"])
            a=torch.tensor([[1.,2.],[3.,4.]],device=device,dtype=torch.float32)
            result=(a@a).cpu().tolist()
            torch.cuda.synchronize(options["gpu_index"])
            ok=result==[[7.,10.],[15.,22.]]
            torch_report["cuda_smoke"]={"status":"passed" if ok else "failed","device":device,"result":result}
            del a
except Exception as error:
    torch_report["error"]=str(error)[:4000]
    if torch_report["import_ok"] and torch_report["cuda_available"]:
        torch_report["cuda_smoke"]={"status":"failed","error":str(error)[:4000]}
report["torch"]=torch_report
ort_report={"import_ok":False,"providers":[],"cuda_provider_advertised":False,"provider_execution_verified":False}
try:
    import onnxruntime as ort
    providers=ort.get_available_providers()
    ort_report.update(import_ok=True,version=ort.__version__,providers=providers,
                      cuda_provider_advertised="CUDAExecutionProvider" in providers)
except Exception as error:
    ort_report["error"]=str(error)[:4000]
report["onnxruntime"]=ort_report
print("FOUNDATION_PROBE="+json.dumps(report,sort_keys=True),flush=True)
'''

WATCHED_PACKAGES = ("PyYAML", "packaging", "numpy", "scipy", "scikit-learn", "torch",
                    "onnxruntime", "onnxruntime-gpu", "tokenizers", "huggingface-hub",
                    "sentence-transformers", "pandas", "matplotlib", "gensim", "pytest")


def installed_packages() -> dict:
    result = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            result[name] = distribution.version
    return dict(sorted(result.items(), key=lambda item: item[0].lower()))


def select_devices(runtime: dict, probe: dict) -> dict:
    torch = probe.get("torch", {})
    smoke = torch.get("cuda_smoke", {})
    valid_cuda = bool(torch.get("import_ok")) and bool(torch.get("cuda_available")) and smoke.get("status") == "passed"
    if (runtime["device"] != "cpu" and torch.get("cuda_available")
            and smoke.get("error") == "Configured GPU index is not available"):
        raise RuntimeError("runtime.gpu_index is outside the detected CUDA device list")
    if runtime["device"] == "cuda" and not valid_cuda:
        raise RuntimeError("CUDA was explicitly requested, but the PyTorch CUDA execution check did not pass. "
                           "Existing packages have been preserved; inspect environment.json.")
    gpu = runtime["device"] != "cpu" and valid_cuda
    ort_candidate = (runtime["device"] != "cpu"
                     and probe.get("onnxruntime", {}).get("cuda_provider_advertised", False))
    selected = "cuda:%d" % runtime["gpu_index"] if gpu else ("cpu" if torch.get("import_ok") else None)
    return {"requested_policy": runtime["device"],
            "torch_device_selected": selected,
            "torch_cpu_execution_verified": torch.get("cpu_smoke", {}).get("status") == "passed",
            "torch_cuda_execution_verified": gpu,
            "encoder_provider_candidate": "CUDAExecutionProvider" if ort_candidate else "CPUExecutionProvider",
            "encoder_provider_execution_verified": False,
            "note": "The actual frozen-encoder ONNX session is verified in Corpus and Evidence Preparation"}


def collect_environment(runtime: dict, logger) -> dict:
    logger.info("Inspecting installed numerical packages, PyTorch and ONNX Runtime in an isolated process.")
    started = time.monotonic()
    try:
        child = subprocess.run([sys.executable, "-c", PROBE_CODE, json.dumps(runtime)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=runtime["probe_timeout_seconds"], shell=False)
        lines = [line for line in child.stdout.splitlines() if line.startswith("FOUNDATION_PROBE=")]
        if child.returncode or not lines:
            probe = {"probe_error": "Environment probe exited %s without a valid report" % child.returncode,
                     "stderr_tail": child.stderr[-4000:]}
        else:
            probe = json.loads(lines[-1].split("=", 1)[1])
    except subprocess.TimeoutExpired:
        probe = {"probe_error": "Environment probe exceeded the configured timeout"}
    packages = installed_packages()
    selected = {}
    for name in WATCHED_PACKAGES:
        try:
            selected[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            selected[name] = None
    smi = {"available": False}
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            process = subprocess.run([binary, "--query-gpu=index,name,driver_version,memory.total",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=10)
            smi = {"available": process.returncode == 0, "gpu_rows": process.stdout.strip().splitlines(),
                   "error": process.stderr[-2000:] if process.returncode else None}
        except (OSError, subprocess.TimeoutExpired) as error:
            smi["error"] = str(error)
    result = {"python": {"version": platform.python_version(), "executable": sys.executable,
                         "implementation": platform.python_implementation(), "bits": 64 if sys.maxsize > 2**32 else 32},
              "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
              "watched_packages": selected, "installed_packages": packages,
              "probe": probe, "nvidia_smi": smi, "probe_elapsed_seconds": time.monotonic() - started}
    # Record diagnostics before the caller enforces explicit CUDA requirements.
    return result
