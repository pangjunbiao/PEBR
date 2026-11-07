"""Backend installation policy, tested without modifying the test environment."""
from importlib import metadata
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from packaging.requirements import Requirement

from transport_topics.runtime.corpus_requirements import (
    MISSING_ONNX_REQUIREMENTS, ensure_corpus_requirements,
)


class CorpusRequirementTests(unittest.TestCase):
    def resolve(self, gpu, backend_packages=None, install_missing=True,
                extra_versions=None, dependencies=None):
        versions = {
            'numpy': '1.26.4', 'scipy': '1.13.1',
            'scikit-learn': '1.5.2', 'tokenizers': '0.19.1',
        }
        versions.update(backend_packages or {})
        versions.update(extra_versions or {})
        environment = {
            'watched_packages': dict(backend_packages or {}),
            'device_selection': {'torch_cuda_execution_verified': gpu},
        }
        requested = []

        def version(name):
            if name not in versions:
                raise metadata.PackageNotFoundError(name)
            return versions[name]

        def install(text, logger, timeout):
            requirement = Requirement(text)
            self.assertEqual(timeout, 900)
            self.assertNotIn(requirement.name, versions)
            requested.append(text)
            candidates = {'onnxruntime-gpu': '1.22.0',
                          'onnxruntime': '1.22.1', 'protobuf': '4.23.3'}
            selected = candidates[requirement.name]
            self.assertTrue(requirement.specifier.contains(selected))
            versions[requirement.name] = selected

        with patch('transport_topics.runtime.corpus_requirements.metadata.version',
                   side_effect=version), \
             patch('transport_topics.runtime.corpus_requirements.metadata.requires',
                   side_effect=lambda name: (dependencies or {}).get(name, [])), \
             patch('transport_topics.runtime.corpus_requirements._install',
                   side_effect=install) as installer, \
             patch('transport_topics.runtime.corpus_requirements.import_module'):
            try:
                records = ensure_corpus_requirements(
                    environment, Mock(), install_missing=install_missing)
            except RuntimeError:
                installer.assert_not_called()
                raise
        return records, requested

    def test_missing_gpu_installs_published_gpu_pin_only(self):
        records, requested = self.resolve(gpu=True)
        self.assertEqual(requested, ['onnxruntime-gpu==1.22.0'])
        self.assertEqual([r['distribution'] for r in records if r['installed_now']],
                         ['onnxruntime-gpu'])

    def test_missing_cpu_retains_separate_cpu_pin(self):
        _, requested = self.resolve(gpu=False)
        self.assertEqual(requested, ['onnxruntime==1.22.1'])

    def test_compatible_existing_gpu_is_preserved(self):
        records, requested = self.resolve(
            gpu=True, backend_packages={'onnxruntime-gpu': '1.23.2'})
        self.assertEqual(requested, [])
        self.assertTrue(all(not r['installed_now'] for r in records))

    def test_cpu_mode_reuses_existing_gpu_distribution(self):
        _, requested = self.resolve(
            gpu=False, backend_packages={'onnxruntime-gpu': '1.22.0'})
        self.assertEqual(requested, [])

    def test_two_backends_are_rejected_before_installation(self):
        with self.assertRaisesRegex(RuntimeError, 'Both ONNX Runtime'):
            self.resolve(gpu=True, backend_packages={
                'onnxruntime': '1.22.1', 'onnxruntime-gpu': '1.22.0'})

    def test_cpu_backend_is_preserved_when_gpu_is_requested(self):
        with self.assertRaisesRegex(RuntimeError, 'CPU ONNX Runtime is installed'):
            self.resolve(gpu=True, backend_packages={'onnxruntime': '1.22.1'})

    def test_no_install_option_rejects_missing_backend(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing corpus requirement'):
            self.resolve(gpu=True, install_missing=False)

    def test_incompatible_existing_backend_is_not_replaced(self):
        with self.assertRaisesRegex(RuntimeError, 'preserved rather than upgraded'):
            self.resolve(gpu=True, backend_packages={'onnxruntime-gpu': '1.27.0'})

    def test_only_missing_transitive_dependency_is_installed(self):
        records, requested = self.resolve(
            gpu=True, backend_packages={'onnxruntime-gpu': '1.22.0'},
            dependencies={'onnxruntime-gpu': ['protobuf>=3.20.2', 'numpy>=1.21.6']})
        self.assertEqual(requested, ['protobuf>=3.20.2'])
        self.assertEqual([r['distribution'] for r in records if r['installed_now']],
                         ['protobuf'])

    def test_gpu_requirements_file_agrees_with_runtime_pin(self):
        root = Path(__file__).resolve().parents[2]
        requirements = [line.strip() for line in
                        (root / 'requirements/gpu.txt').read_text(encoding='utf-8').splitlines()
                        if line.strip() and not line.lstrip().startswith('#')]
        self.assertEqual(requirements, [MISSING_ONNX_REQUIREMENTS['onnxruntime-gpu']])
