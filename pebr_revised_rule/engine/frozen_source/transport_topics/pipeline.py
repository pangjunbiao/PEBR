"""Stage orchestration with a single log and auditable run artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time

from .config import canonical_hash, load_configuration
from .experiments.preflight import readiness, validate_profiles
from .runtime.artifacts import (RunStore, make_inspection, prepare_directories,
                                safe_path, source_identity, write_json)
from .runtime.bootstrap import ensure_foundation
from .runtime.checkpointing import ProjectLock, verify_resume
from .runtime.device import collect_environment, select_devices, installed_packages
from .runtime.logging import RunLogger, now_local, now_utc
from .runtime.reproducibility import configure_process
from .runtime.verification import run_verification

EVALUATION_STAGES=('full_study','transportation_retrieval')
MODEL_STAGES=('model_smoke','topic_benchmark')+EVALUATION_STAGES
RULE_STAGES=('borrowing_rule','borrowing_audit')+MODEL_STAGES
CORPUS_STAGES=('corpus_preparation',)+RULE_STAGES
IMPLEMENTED_STAGES=('execution_foundation','reports')+CORPUS_STAGES


def _environment_identity(environment: dict) -> str:
    return canonical_hash({"python": environment["python"], "platform": environment["platform"],
                           "packages": environment["installed_packages"],
                           "selection": environment.get("device_selection"),
                           "driver": environment.get("nvidia_smi")})


def execute(root: Path, args) -> int:
    root = root.resolve()
    # Explicit opt-in extensions; the existing English workflow is unchanged.
    requested = Path(args.config).resolve() if args.config and Path(args.config).is_absolute() else root / (args.config or '')
    if requested.resolve() == (root / 'configs/reporting/corpus_size_figure.yaml').resolve():
        from .reporting.corpus_size import execute_report
        return execute_report(root, args)
    routes = {root / 'configs/chinese/full_study.yaml': 'chinese',
              root / 'configs/reporting/requested_tables.yaml': 'saved_tables'}
    for config_path, extension in routes.items():
        if requested.resolve() == config_path.resolve():
            from .chinese.entry import execute_extension
            return execute_extension(root, args, extension)
    lock = None
    logger = None
    store = None
    loaded = None
    identity = {}
    final_state = "failed"
    operation = args.mode or "check"
    error_text = None
    inspection = None
    verification = None
    started = time.monotonic()
    stages = {}
    stage = 'project_run'
    try:
        lock = ProjectLock(safe_path(root, "outputs/manifests/execution.lock"))
        lock.acquire()
        store = RunStore(root,stage='project_run')
        logger = RunLogger(safe_path(root, "logs/run_log.txt"), store.run_id)
        logger.info("PROPOSED TRANSPORT PROJECT | implemented through Paper Results and Reproducibility")
        logger.info("Project root: %s", root)
        logger.info("Interpreter: %s", sys.executable)
        logger.info("Python: %s | local time: %s", platform.python_version(), now_local())
        logger.info("Full-study settings are fixed before evaluation. Smoke outputs remain development-only; every declared outcome is retained.")
        store.json("status.json", {"state": "running", "stage": "execution_foundation",
                   "operation": operation, "started_utc": store.started_utc,
                   "started_local": store.started_local})
        tick = time.monotonic()
        bootstrap = ensure_foundation(logger, args.install_missing)
        store.json("requirements_check.json", bootstrap)
        stages["requirements"] = time.monotonic() - tick

        loaded = load_configuration(root, args.config, args.device, args.mode)
        operation = loaded.values["experiment"]["mode"]
        stage = loaded.values['experiment']['stage']
        import yaml
        store.text("config.yaml", yaml.safe_dump(loaded.values, sort_keys=False, allow_unicode=True))
        store.json("effective_settings.json", loaded.effective)
        sources = source_identity(root)
        store.json("source_fingerprints.json", sources)
        store.json("configuration_sources.json", {"files": loaded.source_files, "sha256": loaded.sha256})
        identity.update(configuration_sha256=loaded.sha256, source_sha256=sources["sha256"],
                        stage=stage, input_sha256="no_corpus_inputs")
        logger.info("Configuration valid: %s | hash %s", loaded.values["experiment"]["name"], loaded.sha256[:12])
        logger.info("Model seed list: %s (maximum five).", loaded.values["seeds"])
        if stage not in IMPLEMENTED_STAGES:
            raise RuntimeError("Requested stage '%s' is not implemented yet. "
                               "Publication formatting follows in Paper Results and Reproducibility. No substitute stage was run."
                               % loaded.values["experiment"]["stage"])
        if stage == 'reports' and operation != 'report':
            raise ValueError('The reports stage requires --mode report; it reads one completed full study without model fitting.')
        prepared = prepare_directories(root)
        store.json("directory_check.json", {"validated_directories": prepared})

        if operation == "report":
            if stage == 'reports':
                from .reporting.validation import resolve_study
                from .runtime.reporting_requirements import ensure_reporting_requirements
                # Reject a missing/failed source before verification or output generation.
                target = resolve_study(root, args.run_id).name
                store.json('reporting_requirements.json', ensure_reporting_requirements(logger, args.install_missing))
                verification = run_verification(logger, include_reports=True)
                store.json('verification.json', verification)
                if not verification['passed']:
                    raise RuntimeError('Publication implementation verification failed; inspect the generated bundle')
                store.json('environment.json', {'operation': 'saved_result_publication',
                    'python': platform.python_version(), 'executable': sys.executable,
                    'hardware_probe_repeated': False, 'gpu_required': False})
                from .reporting.run import run as publish
                tick = time.monotonic()
                publication = publish(root, target, logger, store)
                identity['input_sha256'] = publication['source_study_identity']
                stages['paper_results'] = time.monotonic() - tick
                store.json('readiness.json', {'paper_results_complete': True,
                    'source_study_complete': True, 'models_retrained': False, 'next_phase': 'Inspect results and write evidence-supported conclusions'})
            else:
                if args.run_id:
                    target = args.run_id
                else:
                    latest = safe_path(root, "outputs/manifests/latest_"+stage+".json")
                    if not latest.is_file():
                        raise ValueError("There is no previous run to inspect; run main.py first")
                    target = json.loads(latest.read_text(encoding="utf-8"))["run_id"]
                inspection = make_inspection(root, target)
                store.json("report_target.json", {"target_run_id": target,
                           "inspection_archive": inspection.relative_to(root).as_posix()})
                store.json("environment.json", {"operation": "report_only", "python": platform.python_version(),
                           "executable": sys.executable, "hardware_probe_repeated": False})
                logger.info("Inspection archive regenerated for run %s.", target)
        else:
            store.json("process_settings.json", configure_process(loaded.values["runtime"], loaded.values["seeds"]))
            tick = time.monotonic()
            environment = collect_environment(loaded.values["runtime"], logger)
            store.json("environment.json", environment)
            environment["device_selection"] = select_devices(loaded.values["runtime"], environment["probe"])
            store.json("environment.json", environment, overwrite=True)
            if stage in CORPUS_STAGES:
                from .runtime.corpus_requirements import ensure_corpus_requirements
                from importlib import metadata
                requirements=ensure_corpus_requirements(environment,logger,args.install_missing)
                store.json('corpus_requirements.json',requirements)
                environment['installed_packages']=installed_packages()
                for package in environment['watched_packages']:
                    try: environment['watched_packages'][package]=metadata.version(package)
                    except metadata.PackageNotFoundError: pass
                import onnxruntime as ort
                providers=ort.get_available_providers()
                environment['probe']['onnxruntime']={'import_ok':True,'providers':providers,
                    'cuda_provider_advertised':'CUDAExecutionProvider' in providers,'provider_execution_verified':False}
                environment['device_selection']=select_devices(loaded.values['runtime'],environment['probe'])
                store.json('environment.json',environment,overwrite=True)
            if stage in MODEL_STAGES:
                from .baselines.registry import MAIN_METHODS
                from .runtime.model_requirements import ensure_model_requirements
                model_requirements=ensure_model_requirements(MAIN_METHODS,logger,args.install_missing)
                store.json('model_requirements.json',model_requirements)
                environment['installed_packages']=installed_packages()
                store.json('environment.json',environment,overwrite=True)
            if stage in EVALUATION_STAGES:
                from .runtime.evaluation_requirements import ensure_evaluation_requirements
                store.json('evaluation_requirements.json',ensure_evaluation_requirements(logger,args.install_missing))
                environment['installed_packages']=installed_packages()
                for package in environment['watched_packages']:
                    try:environment['watched_packages'][package]=metadata.version(package)
                    except metadata.PackageNotFoundError:environment['watched_packages'][package]=None
                store.json('environment.json',environment,overwrite=True)
            identity["environment_sha256"] = _environment_identity(environment)
            stages["environment"] = time.monotonic() - tick
            for name, version in environment["watched_packages"].items():
                logger.info("Installed %-22s %s", name, version if version is not None else "not installed (inventory only)")
            device = environment["device_selection"]
            logger.info("PyTorch device selected: %s | CUDA execution verified: %s",
                        device["torch_device_selected"] or "not available", device["torch_cuda_execution_verified"])
            logger.info("Encoder provider candidate: %s (actual encoder execution is verified in corpus preparation).",
                        device["encoder_provider_candidate"])
            state = readiness(environment)
            store.json("readiness.json", state)
            for warning in state["warnings"]:
                logger.warning(warning)
            if state["missing_next_phase_packages"]:
                logger.info("Missing FUTURE corpus-phase packages: %s. They were not installed in this phase.",
                            ", ".join(state["missing_next_phase_packages"]))
            if args.resume and not (stage in CORPUS_STAGES and operation=='run'):
                if not loaded.values["output"]["resume"]:
                    raise ValueError("Resume is disabled by the active configuration")
                result = verify_resume(root, args.resume, identity)
                store.json("resume_check.json", result)
                logger.info("Resume compatibility confirmed for %s; current environment checks are refreshed.", args.resume)
            tick = time.monotonic()
            plan = validate_profiles(root)
            store.json("preflight_plan.json", plan)
            stages["profiles"] = time.monotonic() - tick
            logger.info("Validated %d configuration overlays and both dataset descriptors.", plan["validated_overlay_count"])
            logger.info("Transportation outputs reserved: main retrieval table and service-category figure.")
            tick = time.monotonic()
            verification = run_verification(logger,include_corpus=stage!='execution_foundation',
                                            include_rule=stage in RULE_STAGES,
                                            include_models=stage in MODEL_STAGES,
                                            include_evaluation=stage in EVALUATION_STAGES)
            store.json("verification.json", verification)
            stages["verification"] = time.monotonic() - tick
            if not verification["passed"]:
                raise RuntimeError("Implementation verification failed; send the generated inspection ZIP")
            if stage=='corpus_preparation' and operation=='run':
                from .experiments.corpus_preparation import run
                tick=time.monotonic()
                prepared=run(root,loaded.values,environment,logger,store)
                identity['input_sha256']=prepared['sample_manifest_sha256']
                if args.resume:
                    if not loaded.values['output']['resume']: raise ValueError('Resume is disabled')
                    store.json('resume_check.json',verify_resume(root,args.resume,identity))
                state.update(datasets_downloaded=True,encoder_session_verified=True,
                             registered_preparation_complete=prepared['complete_registered_preparation'],
                             next_phase='Proposed Borrowing Rule' if prepared['complete_registered_preparation'] else 'Complete full corpus preparation')
                store.json('readiness.json',state,overwrite=True)
                environment['device_selection']['encoder_provider_execution_verified']=True
                store.json('environment.json',environment,overwrite=True)
                stages['corpus_preparation']=time.monotonic()-tick
            elif stage in ('borrowing_rule','borrowing_audit') and operation=='run':
                from .experiments.borrowing import run
                tick = time.monotonic()
                rule_result, dependency = run(root, loaded.values, environment, logger, store)
                identity['input_sha256'] = dependency['preparation_sha256']
                if args.resume:
                    if not loaded.values['output']['resume']: raise ValueError('Resume is disabled')
                    store.json('resume_check.json', verify_resume(root, args.resume, identity))
                state.update(datasets_downloaded=True, encoder_session_verified=True,
                             prepared_evidence_reused=True,
                             registered_rule_complete=rule_result['complete_registered_proposed_scope'],
                             all_reference_fits_converged=rule_result['all_reference_fits_converged'],
                             research_performance_evaluated=rule_result['outer_prediction_audited'],
                             implemented_scope='Proposed action selection, evidence transfer and protected prediction audit',
                             final_topic_performance_evaluated=False,
                             next_phase='Topic Models and Baseline Integration')
                store.json('readiness.json', state, overwrite=True)
                stages['borrowing_rule'] = time.monotonic()-tick
            elif stage in ('model_smoke','topic_benchmark') and operation=='run':
                from .experiments.models import run
                tick=time.monotonic()
                model_result,dependency=run(root,loaded.values,environment,logger,store)
                identity['input_sha256']=dependency['preparation_sha256']
                if args.resume:
                    if not loaded.values['output']['resume']:raise ValueError('Resume is disabled')
                    store.json('resume_check.json',verify_resume(root,args.resume,identity))
                state.update(datasets_downloaded=True,encoder_session_verified=True,
                             prepared_evidence_reused=True,models_integrated=True,
                             development_only=model_result['development_only'],
                             performance_comparisons_evaluated=False,
                             next_phase='Transportation Evaluation and Experimental Studies')
                store.json('readiness.json',state,overwrite=True)
                stages['model_integration']=time.monotonic()-tick
            elif stage in EVALUATION_STAGES and operation=='run':
                from .data.prepared import preparation_dependency
                from .experiments.studies import run as run_studies
                _,dependency=preparation_dependency(root,loaded.values)
                identity['input_sha256']=canonical_hash(dependency)
                if args.resume:
                    if not loaded.values['output']['resume']:raise ValueError('Resume is disabled')
                    store.json('resume_check.json',verify_resume(root,args.resume,identity))
                tick=time.monotonic()
                result,dependency=run_studies(root,loaded.values,environment,logger,store)
                state.update(prepared_evidence_reused=True,models_integrated=True,
                             development_only=result['development_only'],performance_comparisons_evaluated=True,
                             next_phase='Paper Results and Reproducibility')
                store.json('readiness.json',state,overwrite=True)
                stages['scientific_evaluation']=time.monotonic()-tick
        final_state = "completed"
    except KeyboardInterrupt:
        final_state = "interrupted"
        error_text = "Interrupted by user"
        if logger:
            logger.warning(error_text)
        else:
            print(error_text, file=sys.stderr)
    except Exception as error:
        error_text = str(error)
        if logger:
            logger.exception("RUN FAILED: " + error_text)
        else:
            print("ERROR: " + error_text, file=sys.stderr)
    finally:
        if store is not None and logger is not None:
            try:
                prediction_path = store.directory / 'outer_prediction_audit.json'
                prediction_computed = (prediction_path.is_file() and
                                       bool(json.loads(prediction_path.read_text(encoding='utf-8')).get('results')))
                control_index=store.directory/'sharing_controls_index.json'
                if control_index.is_file():
                    prediction_computed = prediction_computed or any(
                        bool(bank.get('outer_prediction')) for bank in json.loads(control_index.read_text(encoding='utf-8'))['banks'])
                model_index=store.directory/'models_index.json'
                model_fits = model_index.is_file() and bool(json.loads(model_index.read_text(encoding='utf-8')).get('models'))
                evaluation_path=store.directory/'evaluation_summary.json'
                evaluation_complete=evaluation_path.is_file() and json.loads(evaluation_path.read_text()).get('complete_requested_study',False)
                progress_path=store.directory/'evaluation_progress.json'
                evaluation_partial=progress_path.is_file() and json.loads(progress_path.read_text()).get('logical_evaluations_completed',0)>0
                publication_path=store.directory/'publication_summary.json'
                publication_complete=publication_path.is_file() and json.loads(publication_path.read_text(encoding='utf-8')).get('complete',False)
                store.json("status.json", {"state": final_state, "stage": stage,
                           "operation": operation, "started_utc": store.started_utc,
                           "started_local": store.started_local, "finished_utc": now_utc(),
                           "finished_local": now_local(), "error": error_text,
                           "research_performance_computed": bool(prediction_computed or evaluation_partial),
                           "final_topic_performance_computed": bool(evaluation_partial),
                           "requested_evaluation_complete":bool(evaluation_complete),
                           "requested_publication_complete":bool(publication_complete),
                           "source_evaluation_reused":bool(publication_complete),
                           "development_only":bool(loaded and stage in MODEL_STAGES and loaded.values['integration']['scope']=='smoke'),
                           "model_outputs_fitted":bool(model_fits)}, overwrite=True)
                store.json("timing.json", {"total_seconds": time.monotonic() - started, "stages": stages})
                store.text("summary.txt", "Project run: %s\nRun ID: %s\nOperation: %s\n"
                           "Tests: %s\nProtected predictive audit computed: %s\nModel outputs fitted or reused: %s\nComparative evaluations complete: %s\nPaper results complete: %s\nError: %s\n"
                           % (final_state, store.run_id, operation,
                              verification["tests_run"] if verification else "not run", bool(prediction_computed),bool(model_fits),bool(evaluation_complete),bool(publication_complete), error_text or "none"))
                store.manifest(identity)
                logger.info("RUN %s | ID: %s | elapsed %.2fs", final_state.upper(), store.run_id, time.monotonic() - started)
                if inspection is None and (loaded is None or loaded.values["output"]["inspection_zip"]):
                    inspection = make_inspection(root, store.run_id)
                pointer = {"run_id": store.run_id, "state": final_state,
                           "run_directory": store.directory.relative_to(root).as_posix(),
                           "inspection_zip": inspection.relative_to(root).as_posix() if inspection else None}
                pointer_name = ('latest_'+stage+'_report.json') if operation=='report' else ('latest_'+stage+'.json')
                write_json(safe_path(root, "outputs/manifests/" + pointer_name), pointer, overwrite=True)
                logger.info("Saved run artifacts: %s", store.directory)
                logger.info("Single append-only log: %s", root / "logs/run_log.txt")
                if inspection:
                    logger.info("SEND THIS INSPECTION ZIP: %s", inspection)
                if final_state == "completed" and operation != "report":
                    logger.info("Implemented stage completed: %s (%s). Send the inspection bundle before the next phase.",stage,operation)
            except Exception as error:
                final_state = "failed"
                logger.exception("Could not finish saving the run: " + str(error))
                try:
                    store.json("status.json", {"state": "failed", "stage": "execution_foundation",
                               "operation": operation, "finished_utc": now_utc(),
                               "error": "Run finalization failed: " + str(error),
                               "research_performance_computed": False}, overwrite=True)
                    store.manifest(identity)
                except Exception:
                    logger.error("Status writeback also failed; rely on the console error and nonzero exit code.")
            finally:
                logger.close()
        if lock:
            lock.release()
    return 0 if final_state == "completed" else (130 if final_state == "interrupted" else 1)
