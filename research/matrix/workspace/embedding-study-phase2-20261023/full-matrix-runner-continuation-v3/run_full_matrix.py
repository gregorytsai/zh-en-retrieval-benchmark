#!/usr/bin/env python3
"""Planner/validator and explicit guarded entry point for the continuation matrix.

Without --execute this program is offline-only. An explicit pilot is isolated
from official output; official runs require unchanged, approved pilot evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
from pathlib import Path
from typing import Any

from matrix_spec import (
    BATCHES, RUN_BATCHES, BATCH_DATASETS, DATASET_COUNTS, EMBEDDERS, MODEL_PATHS, MODEL_REGISTRY, QUERY_ROWS,
    ROWS_PER_EMBEDDER, SUMMARIES_PER_EMBEDDER, TOTAL_ROWS, TOTAL_SUMMARIES,
    MatrixValidationError, enumerate_configs, validate_and_merge_manifests,
)
from pilot_forecast import (
    PilotGateError, audit_and_forecast, length_quantile_selection_issues,
    verify_official_gate, write_decision,
)

PINNED_DATASET_SHA256 = {
    "first_batch": "bc0657a77090f86b081b378e3f369653dc08f6049da78487b7dc0c134e7adad1",
    "miracl_batch": "fdf802cc03e9059783e5a2cba0785437253f07841ec6956246d5884461e7ba3d",
}


class PreparationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path, sidecar: Path | None = None) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise PreparationError(f"Dataset manifest not found: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise PreparationError(f"{path}: expected a top-level datasets mapping")
    digest = hashlib.sha256(raw).hexdigest()
    if sidecar is None or not sidecar.is_file():
        raise PreparationError(f"Required validation evidence is missing: {sidecar}")
    if sidecar.is_file():
        try:
            evidence = json.loads(sidecar.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreparationError(f"Invalid validation evidence {sidecar}: {exc}") from exc
        expected = evidence.get("datasets_sha256")
        if expected and expected != digest:
            raise PreparationError(f"{sidecar} does not match {path} SHA-256")
    return payload, digest


def _asset_report(workspace: Path, verify_hashes: bool) -> dict[str, Any]:
    rows = []
    for name, spec in MODEL_REGISTRY.items():
        path = workspace / spec.path
        row: dict[str, Any] = {
            "id": name, "kind": spec.kind, "path": spec.path,
            "expected_bytes": spec.bytes, "expected_sha256": spec.sha256,
            "exists": path.is_file(),
        }
        if path.is_file():
            row["actual_bytes"] = path.stat().st_size
            if verify_hashes:
                row["actual_sha256"] = sha256_file(path)
                row["pin_match"] = row["actual_bytes"] == spec.bytes and row["actual_sha256"] == spec.sha256
        if verify_hashes and not path.is_file():
            row["pin_match"] = False
        rows.append(row)
    binary_rel = MODEL_PATHS["llama-server"]
    binary = workspace / binary_rel
    result: dict[str, Any] = {
        "models": rows,
        "binary": {"path": binary_rel, "exists": binary.is_file()},
        "hash_verification_requested": verify_hashes,
    }
    if binary.is_file() and verify_hashes:
        result["binary"]["sha256"] = sha256_file(binary)
    if verify_hashes:
        result["model_pins_pass"] = len(rows) == 10 and all(row.get("pin_match") is True for row in rows)
        result["binary_pin"] = "not pinned in this registry; requires separately verified binary manifest"
    return result


def build_plan(workspace: Path, datasets_path: Path, miracl_path: Path,
               embedder_ids: list[str] | None = None, verify_assets: bool = False,
               selected_batch: str | None = None) -> dict[str, Any]:
    first, first_sha = _read_manifest(datasets_path, datasets_path.with_name("data-validation.json"))
    second, second_sha = _read_manifest(miracl_path, miracl_path.with_name("batch2-miracl-validation.json"))
    if first_sha != PINNED_DATASET_SHA256["first_batch"]:
        raise PreparationError("First-batch datasets.json SHA-256 differs from the frozen manifest")
    if second_sha != PINNED_DATASET_SHA256["miracl_batch"]:
        raise PreparationError("MIRACL datasets-batch2.json SHA-256 differs from the frozen manifest")
    datasets = validate_and_merge_manifests(first, second)
    batch_map = dict(RUN_BATCHES)
    if selected_batch is not None:
        if selected_batch not in batch_map:
            raise MatrixValidationError(f"Unknown batch {selected_batch}")
        selected_embedders = [batch_map[selected_batch]]
        selected_dataset_ids = list(BATCH_DATASETS[selected_batch])
    else:
        selected_embedders = list(embedder_ids) if embedder_ids is not None else list(EMBEDDERS)
        selected_dataset_ids = list(DATASET_COUNTS)
    configs = enumerate_configs(selected_embedders)
    batch_rows = QUERY_ROWS * 4
    return {
        "status": "offline-validation",
        "execution_enabled": True,
        "scope": (f"{selected_batch}: one embedder × four configurations × {len(selected_dataset_ids)} fixed datasets"
                  if selected_batch else "seven embedders × standalone plus three rerankers × six datasets"),
        "workspace": str(workspace),
        "output_path": None,
        "output_created": False,
        "input_manifests": {
            str(datasets_path): {"sha256": first_sha, "datasets": sorted(first["datasets"])},
            str(miracl_path): {"sha256": second_sha, "datasets": sorted(second["datasets"])},
        },
        "datasets": {
            name: {
                "corpus": len(datasets[name]["corpus"]),
                "queries": len(datasets[name]["queries"]),
                "qrels": sum(len(judgments) for judgments in datasets[name]["qrels"].values()),
                "grade_mode": datasets[name]["grade_mode"],
            }
            for name in DATASET_COUNTS
        },
        "matrix": {
            "selected_embedders": selected_embedders,
            "selected_batch": selected_batch,
            "selected_dataset_ids": selected_dataset_ids,
            "batch_schedule": [
                {"batch": batch, "embedder": embedder,
                 "datasets": list(BATCH_DATASETS[batch]),
                 "summaries": len(BATCH_DATASETS[batch]) * 4,
                 "query_rows": sum(DATASET_COUNTS[name]["queries"] for name in BATCH_DATASETS[batch]) * 4}
                for batch, embedder in RUN_BATCHES
            ],
            "configurations": configs,
            "config_count": len(configs),
            "summaries": len(configs) * len(selected_dataset_ids),
            "query_rows": len(configs) * sum(DATASET_COUNTS[name]["queries"] for name in selected_dataset_ids),
            "all_embedder_summaries": TOTAL_SUMMARIES,
            "all_embedder_rows": TOTAL_ROWS,
            "per_embedder_summaries": SUMMARIES_PER_EMBEDDER,
            "per_embedder_rows": ROWS_PER_EMBEDDER,
            "per_embedder_rows_check": batch_rows,
            "candidate_k": 50,
            "standalone_final_k": 50,
            "reranker_candidate_k": 50,
            "reranker_cache_rule": "same embedder+dataset candidate IDs and order across all three rerankers",
            "quality_macro": "equal-weight mean of six dataset nDCG@10; eligible only when all six are complete",
        },
        "assets": _asset_report(workspace, verify_assets),
        "runtime_policy": {
            "max_wall_hours_per_embedder_batch": 12,
            "execute_requires_explicit_flag": True,
            "execute_status": "guarded execution requires explicit --execute, one fixed --batch, and approved pilot evidence for official mode",
            "offline_plan_starts_no_model_or_http": True,
            "no_output_write": True,
        },
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace root containing model/binary paths")
    parser.add_argument("--datasets", type=Path, default=None, help="First-batch datasets.json")
    parser.add_argument("--datasets-batch2", type=Path, default=None, help="MIRACL datasets-batch2.json")
    parser.add_argument("--output", type=Path, default=None, help="Reserved output root (read-only modes never create it)")
    parser.add_argument("--plan-path", type=Path, default=None, help="Progress plan updated only after audited official completion")
    parser.add_argument("--embedder", action="append", choices=EMBEDDERS, help="Plan one embedder batch; repeatable")
    parser.add_argument("--batch", choices=[batch for batch, _ in RUN_BATCHES], help="Select a fixed continuation batch ID")
    parser.add_argument("--validate", action="store_true", help="Validate full six-dataset schema and hashes")
    parser.add_argument("--plan", action="store_true", help="Print the frozen matrix plan (default)")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --plan; no files are written")
    parser.add_argument("--verify-assets", action="store_true", help="Read all model files and verify pinned bytes/SHA-256")
    parser.add_argument("--wall-hours", type=float, default=None, help="Optional assertion only: pilot is fixed at 2h and official at 12h")
    parser.add_argument("--execute", action="store_true", help="Explicitly enter the guarded runner; requires one --batch")
    parser.add_argument("--pilot", action="store_true", help="Run the isolated 2h, 72-row pilot for the selected batch")
    parser.add_argument("--resume", action="store_true", help="Resume the same pilot/official output without resetting its original deadline")
    parser.add_argument("--pilot-evidence", type=Path, default=None, help="Approved pilot decision JSON (official mode; defaults under output/Bxx/pilot)")
    parser.add_argument("--segment", default="01", help="Immutable run segment label; a new label creates a new fixed deadline")
    parser.add_argument("--continue-compatible-from", type=Path, default=None,
                        help="Start a new capped official segment by importing only strictly verified commits from a prior stopped segment")
    return parser


def build_engine_namespace(*, workspace: Path, datasets_path: Path, miracl_path: Path,
                           output_root: Path, selected_batch: str, pilot: bool,
                           resume: bool, plan_path: Path, segment: str = "01",
                           continue_compatible_from: Path | None = None) -> argparse.Namespace:
    """Map public CLI options to the guarded engine's intentionally small API."""
    embedder = dict(RUN_BATCHES)[selected_batch]
    return argparse.Namespace(
        execute=True,
        pilot=pilot,
        datasets_file=datasets_path,
        datasets_batch2=miracl_path,
        output=output_root / selected_batch / f"segment-{segment}",
        workspace=workspace,
        batch_id=selected_batch,
        embedder=embedder,
        wall_hours=2.0 if pilot else 12.0,
        deadline=None,
        resume=resume,
        pilot_queries=3,
        pilot_datasets=list(BATCH_DATASETS[selected_batch]),
        validate_data_only=False,
        plan_path=plan_path,
        dataset_ids=list(BATCH_DATASETS[selected_batch]),
        continue_compatible_from=continue_compatible_from,
    )


def _combined_dataset_sha256(datasets_path: Path, miracl_path: Path) -> str:
    from matrix_spec import sha256_json
    observed = {"first": sha256_file(datasets_path), "second": sha256_file(miracl_path)}
    return sha256_json(observed)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.wall_hours is not None and (not math.isfinite(args.wall_hours) or args.wall_hours <= 0 or args.wall_hours > 12):
        parser.error("--wall-hours must be >0 and cannot exceed the 12-hour hard cap")
    if args.embedder and len(args.embedder) != len(set(args.embedder)):
        parser.error("--embedder values must be unique")
    if args.batch and args.embedder:
        parser.error("Use --batch or --embedder, not both")
    if args.resume and not args.execute:
        parser.error("--resume is valid only with --execute")
    if args.pilot_evidence and (not args.execute or args.pilot):
        parser.error("--pilot-evidence is valid only for an official --execute run")
    if not args.segment or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in args.segment):
        parser.error("--segment must be a nonempty filesystem-safe label")
    if args.continue_compatible_from and (not args.execute or args.pilot):
        parser.error("--continue-compatible-from is valid only for an official --execute segment")

    workspace = args.workspace.expanduser().resolve()
    project = workspace / "embedding-study-phase2-20261023"
    datasets_path = (args.datasets or project / "datasets.json").expanduser().resolve()
    miracl_path = (args.datasets_batch2 or project / "datasets-batch2.json").expanduser().resolve()
    # Offline modes include output as metadata only and never create it.
    output = (args.output or project / "run-full-matrix").expanduser()
    if not output.is_absolute():
        output = workspace / output
    plan_path = (args.plan_path or project / "FULL_MATRIX_PLAN_ZH_TW.md").expanduser()
    if not plan_path.is_absolute():
        plan_path = workspace / plan_path
    plan_path = plan_path.resolve()

    if args.execute:
        if not args.batch:
            parser.error("--execute requires exactly one continuation --batch")
        if args.embedder:
            parser.error("--execute uses the fixed --batch registry; do not pass --embedder")
        if args.validate or args.plan or args.dry_run:
            parser.error("--execute cannot be combined with offline --validate/--plan/--dry-run")
        fixed_hours = 2.0 if args.pilot else 12.0
        if args.wall_hours is not None and args.wall_hours != fixed_hours:
            parser.error(f"This mode has a fixed {fixed_hours:g}-hour wall deadline; --wall-hours cannot change it")
        try:
            # Keep the same full data-schema and frozen-SHA checks as offline validation.
            build_plan(workspace, datasets_path, miracl_path, selected_batch=args.batch)
            from matrix_spec import RUN_BATCHES as BATCH_REGISTRY
            from guarded_engine import core as engine_core
            import guarded_engine
            engine_core.configure(workspace)
            engine_args = build_engine_namespace(
                workspace=workspace,
                datasets_path=datasets_path,
                miracl_path=miracl_path,
                output_root=output.resolve(),
                selected_batch=args.batch,
                pilot=args.pilot,
                resume=args.resume,
                plan_path=plan_path,
                segment=args.segment,
                continue_compatible_from=args.continue_compatible_from,
            )
            embedder = dict(BATCH_REGISTRY)[args.batch]
            if args.pilot:
                pilot_datasets, _ = guarded_engine.load_matrix_inputs(engine_args)
                pilot_selection = guarded_engine.build_selection(
                    pilot_datasets, list(engine_args.dataset_ids), True, 3,
                )
                selection_issues = length_quantile_selection_issues(
                    pilot_datasets, pilot_selection, list(engine_args.dataset_ids))
                if selection_issues:
                    raise PilotGateError("Pilot selection is not fixed short/mid/long quantiles: " + "; ".join(selection_issues))
            if not args.pilot:
                default_pilot_home = output.resolve() / args.batch / f"segment-{args.segment}" / "pilot"
                decision_path = (args.pilot_evidence or default_pilot_home / "pilot-decision.json").expanduser().resolve()
                pilot_home = decision_path.parent
                env = guarded_engine.environment_hashes(
                    datasets_path,
                    _combined_dataset_sha256(datasets_path, miracl_path),
                )
                verify_official_gate(decision_path, pilot_home, args.batch, embedder, env,
                                     list(engine_args.dataset_ids))
            if args.pilot:
                pilot_home = engine_args.output / "pilot"
                try:
                    guarded_engine.run(engine_args)
                except BaseException:
                    if pilot_home.is_dir():
                        try:
                            decision = audit_and_forecast(pilot_home, args.batch, embedder)
                            write_decision(pilot_home, decision)
                        except Exception:
                            pass
                    raise
                decision = audit_and_forecast(pilot_home, args.batch, embedder)
                write_decision(pilot_home, decision)
                print(json.dumps(decision, ensure_ascii=False, indent=2))
                return 0 if decision.get("evidence_audit_passed") is True else 3
            guarded_engine.run(engine_args)
            return 0
        except (OSError, PreparationError, MatrixValidationError, PilotGateError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    try:
        result = build_plan(workspace, datasets_path, miracl_path, args.embedder, args.verify_assets, args.batch)
    except (OSError, PreparationError, MatrixValidationError) as exc:
        parser.error(str(exc))
    result["output_path"] = str(output)
    result["mode"] = "validate" if args.validate and not (args.plan or args.dry_run) else "plan"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _handle_termination(signum: int, _frame: Any) -> None:
    """Raise through the engine stack so its finally block stops owned services."""
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)
    sys.exit(main())
