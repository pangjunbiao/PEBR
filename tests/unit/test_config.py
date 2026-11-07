from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from transport_topics.config import canonical_hash, deep_merge, load_configuration, read_yaml
from transport_topics.schemas import ConfigurationError, validate_config

ROOT = Path(__file__).resolve().parents[2]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_configuration(ROOT)

    def rejected(self, modify):
        values = deepcopy(self.config.values)
        modify(values)
        with self.assertRaises(ConfigurationError):
            validate_config(values)

    def test_default_is_proposed_and_five_seeds(self):
        self.assertEqual(self.config.values["project"]["method_label"], "proposed")
        self.assertEqual(self.config.values["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(self.config.values["experiment"]["stage"], "full_study")

    def test_default_borrowing_grid(self):
        for actual, expected in zip(self.config.effective["borrowing_grid"], [0, 1/12, 1/6, 1/4, 1/3]):
            self.assertAlmostEqual(actual, expected)

    def test_unknown_nested_field_is_rejected(self):
        self.rejected(lambda c: c["method"]["selection"].update({"downside_weigth": 2}))

    def test_unknown_root_field_is_rejected(self):
        self.rejected(lambda c: c.update({"silent_extra": True}))

    def test_sixth_seed_is_rejected(self):
        self.rejected(lambda c: c.update(seeds=list(range(6))))

    def test_boolean_seed_is_rejected(self):
        self.rejected(lambda c: c.update(seeds=[True]))

    def test_duplicate_seed_is_rejected(self):
        self.rejected(lambda c: c.update(seeds=[0, 0]))

    def test_seed_outside_declared_list_is_rejected(self):
        self.rejected(lambda c: c.update(seeds=[7]))

    def test_cap_one_is_rejected(self):
        self.rejected(lambda c: c["method"]["transfer"].update(max_weight=1))

    def test_negative_downside_is_rejected(self):
        self.rejected(lambda c: c["method"]["selection"]["downside"].update(weight=-1))

    def test_nonfinite_number_is_rejected(self):
        self.rejected(lambda c: c["method"]["selection"]["downside"].update(weight=float("nan")))

    def test_grid_must_include_cap(self):
        self.rejected(lambda c: c["method"]["transfer"].update(grid_fractions=[0.5]))

    def test_downside_switch_resolves_to_zero(self):
        values = deepcopy(self.config.values)
        values["method"]["selection"]["downside"]["enabled"] = False
        self.assertEqual(validate_config(values)["effective_downside_weight"], 0)

    def test_borrowing_switch_resolves_to_none(self):
        values = deepcopy(self.config.values)
        values["method"]["borrowing"]["enabled"] = False
        self.assertEqual(validate_config(values)["selection_strategy"], "none")

    def test_original_nmf_overlay(self):
        config = load_configuration(ROOT, "configs/methods/nmf_original.yaml")
        self.assertFalse(config.values["method"]["borrowing"]["enabled"])
        self.assertEqual(config.values["method"]["name"], "nmf_original")

    def test_device_override_is_fingerprinted(self):
        target = "cpu" if self.config.values["runtime"]["device"] != "cpu" else "auto"
        modified = load_configuration(ROOT, device=target)
        self.assertEqual(modified.values["runtime"]["device"], target)
        self.assertNotEqual(modified.sha256, self.config.sha256)

    def test_hash_ignores_mapping_order(self):
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))

    def test_merge_does_not_mutate_inputs(self):
        original = {"a": {"x": 1, "y": [2]}}
        merged = deep_merge(original, {"a": {"x": 4}})
        merged["a"]["y"].append(3)
        self.assertEqual(original, {"a": {"x": 1, "y": [2]}})

    def test_yaml_duplicate_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("x: 1\nx: 2\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                read_yaml(path)

    def test_empty_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                read_yaml(path)

    def test_unsafe_yaml_tag_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("x: !!python/object/apply:os.getcwd []", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                read_yaml(path)

    def test_yaml_extends_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/project.yaml").write_text("extends: [configs/project.yaml]\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_configuration(root)

    def test_yaml_parent_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/project.yaml").write_text("extends: [../outside.yaml]\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_configuration(root)

    def test_zero_budget_rejected(self):
        self.rejected(lambda c: c["retrieval"].update(budgets=[0, 10]))

    def test_evaluation_reference_cannot_be_augmented(self):
        self.rejected(lambda c: c["evaluation"]["topic"].update(original_text_reference=False))

    def test_tf32_cannot_be_silently_enabled(self):
        self.rejected(lambda c: c["runtime"].update(allow_tf32=True))
