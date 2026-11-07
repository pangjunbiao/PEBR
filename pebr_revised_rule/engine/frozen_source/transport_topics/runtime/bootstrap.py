"""Check this interpreter; install only genuinely missing foundation packages."""
from __future__ import annotations

import importlib
from importlib import metadata
import subprocess
import sys
import time

FOUNDATION_REQUIREMENTS = (
    ("packaging", "packaging", ">=23,<27"),
    ("PyYAML", "yaml", ">=6,<7"),
)


def _install(requirement: str, logger, timeout: float = 180.0) -> None:
    command = [sys.executable, "-m", "pip", "install", "--no-input",
               "--disable-pip-version-check", "--no-deps", requirement]
    logger.info("Installing missing foundation requirement in the CURRENT interpreter: %s", requirement)
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", shell=False)
    try:
        while True:
            try:
                output, _ = process.communicate(timeout=min(10.0, timeout))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Package installation timed out after %.0f seconds" % timeout)
                logger.info("Installation is still running (%.0fs).", time.monotonic() - started)
        # Pip masks index credentials; avoid storing unrelated index/configuration lines.
        for line in output.splitlines():
            if line.startswith(("Successfully installed", "Requirement already satisfied", "ERROR:", "WARNING:")) or "No module named pip" in line:
                logger.info("pip: %s", line)
        if process.returncode:
            raise RuntimeError("pip failed for %s (exit %s). Run this requirement with the current interpreter."
                               % (requirement, process.returncode))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    importlib.invalidate_caches()


def ensure_foundation(logger, install_missing: bool = True) -> list[dict]:
    results = []
    for distribution, module, specifier in FOUNDATION_REQUIREMENTS:
        installed_now = False
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            if not install_missing:
                raise RuntimeError("Missing %s%s. Run main.py --install-missing in this interpreter."
                                   % (distribution, specifier))
            _install(distribution + specifier, logger)
            version = metadata.version(distribution)
            installed_now = True
        # packaging is the first requirement, so version checking is available here.
        from packaging.specifiers import SpecifierSet
        if not SpecifierSet(specifier).contains(version, prereleases=True):
            raise RuntimeError("Existing %s==%s is outside the phase's checked range %s. "
                               "It has been preserved; send the inspection ZIP before changing it."
                               % (distribution, version, specifier))
        try:
            importlib.import_module(module)
        except Exception as error:
            raise RuntimeError("%s is installed but cannot import: %s. It was not reinstalled."
                               % (distribution, error)) from error
        logger.info("Requirement OK: %s==%s (%s).", distribution, version,
                    "installed missing package" if installed_now else "existing installation preserved")
        results.append({"distribution": distribution, "version": version,
                        "required_range": specifier, "installed_now": installed_now})
    return results
