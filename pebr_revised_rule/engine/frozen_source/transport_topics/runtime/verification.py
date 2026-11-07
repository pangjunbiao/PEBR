"""Run implemented foundation tests without installing a test framework."""
from __future__ import annotations

import io
import time
import unittest

TEST_MODULES = (
    "tests.unit.test_config", "tests.unit.test_logging", "tests.unit.test_runtime",
    "tests.integration.test_pipeline_smoke", "tests.integration.test_resume",
    "tests.unit.test_artifact_recovery",
)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._started = {}

    def startTest(self, test):
        self._started[test.id()] = time.monotonic()
        super().startTest(test)

    def _record(self, test, state, detail=None):
        self.records.append({"test": test.id(), "state": state, "detail": detail,
                             "seconds": time.monotonic() - self._started.get(test.id(), time.monotonic())})

    def addSuccess(self, test):
        self._record(test, "passed")
        super().addSuccess(test)

    def addFailure(self, test, error):
        self._record(test, "failed", self._exc_info_to_string(error, test))
        super().addFailure(test, error)

    def addError(self, test, error):
        self._record(test, "error", self._exc_info_to_string(error, test))
        super().addError(test, error)

    def addSkip(self, test, reason):
        self._record(test, "skipped", reason)
        super().addSkip(test, reason)


def run_verification(logger,include_corpus=False,include_rule=False,include_models=False,include_evaluation=False,include_reports=False) -> dict:
    names=TEST_MODULES+(("tests.unit.test_data","tests.unit.test_families_and_folds", "tests.unit.test_masks", "tests.unit.test_information_boundaries", "tests.unit.test_corpus_requirements") if include_corpus else ())
    if include_rule:
        names += ('tests.unit.test_nnls', 'tests.unit.test_reference', 'tests.unit.test_transfer',
                  'tests.unit.test_action_scoring', 'tests.unit.test_search', 'tests.integration.test_borrowing')
    if include_models:
        names += ('tests.unit.test_models',)
    if include_evaluation:
        names += ('tests.unit.test_metrics','tests.unit.test_retrieval','tests.unit.test_study_plans',
                  'tests.integration.test_evaluation')
    if include_reports:
        names += ('tests.unit.test_paper_results', 'tests.integration.test_paper_exports')
    suite = unittest.TestLoader().loadTestsFromNames(names)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    report = {"passed": passed, "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped), "tests": result.records,
              "runner_output": stream.getvalue(), "scope": "Infrastructure, prepared evidence, numerical rule and registered model adapters; comparative superiority is not an implementation check"}
    logger.info("Implementation checks: %d tests, %d failures, %d errors, %d skipped.",
                result.testsRun, len(result.failures), len(result.errors), len(result.skipped))
    if not passed:
        for record in result.records:
            if record["state"] in ("failed", "error"):
                logger.error("Check failed: %s\n%s", record["test"], record["detail"])
    return report
