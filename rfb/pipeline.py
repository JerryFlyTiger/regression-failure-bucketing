"""Parsing orchestration and top-level run() entry point.

Parsing (M1), deviation derivation (M2, rfb.diverge.derive) and signature /
clustering (M3, rfb.signature / rfb.cluster) are wired together here. The
log parsers and rfb.diverge are imported lazily inside the functions that use
them (see _derive_deviation below).
"""

import csv
import multiprocessing
import os


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def classify_family(regr, sim) -> str:
    """Classify a case into MISMATCH / FATAL / ASSERT / OTHER.

    See docs/ARCHITECTURE.md section 5.1. Order matters: MISMATCH is
    checked first, then FATAL, then ASSERT, everything else is OTHER.
    """
    if regr.kind == "MISMATCH":
        return "MISMATCH"
    if sim.fatals:
        return "FATAL"
    if sim.error_templates:
        return "ASSERT"
    return "OTHER"


def _fallback_deviation():
    from rfb.record import Deviation

    return Deviation(kind="NONE", mnemonic=None, mclass=None, detail={}, source="none", confidence=0.0)


def _derive_deviation(record):
    """Call rfb.diverge.derive(record). Any exception from derive() is caught
    and falls back to a NONE deviation, recording the error instead of
    aborting the whole case."""
    from rfb.diverge import derive

    try:
        return derive(record), []
    except Exception as exc:  # noqa: BLE001 - collected, not raised
        return _fallback_deviation(), [f"diverge.derive: {exc}"]


def _parse_one_case(case_id: int, regr_path: str, sim_path: str, trace_path: str):
    """Parse one case's three logs into a FailureRecord, with its deviation
    already filled in. Runs inside a multiprocessing worker (must be a
    top-level, picklable function); every exception is caught per-file so a
    single bad case never aborts the whole batch."""
    from rfb.record import FailureRecord, RegrInfo, SimInfo, TraceInfo
    from rfb.parse_regr import parse_regr
    from rfb.parse_sim import parse_sim
    from rfb.parse_trace import parse_trace

    parse_errors: list[str] = []
    try:
        regr_info = parse_regr(regr_path)
    except Exception as exc:  # noqa: BLE001 - collected, not raised
        parse_errors.append(f"parse_regr: {exc}")
        regr_info = RegrInfo.empty()
    try:
        sim_info = parse_sim(sim_path)
    except Exception as exc:  # noqa: BLE001
        parse_errors.append(f"parse_sim: {exc}")
        sim_info = SimInfo.empty()
    try:
        trace_info = parse_trace(trace_path, regr_info)
        parse_errors.extend(trace_info.parse_errors)
    except Exception as exc:  # noqa: BLE001
        parse_errors.append(f"parse_trace: {exc}")
        trace_info = TraceInfo.empty()

    family = classify_family(regr_info, sim_info)

    record = FailureRecord(
        case_id=case_id,
        family=family,
        regr=regr_info,
        sim=sim_info,
        trace=trace_info,
        deviation=None,
        parse_errors=parse_errors,
    )

    deviation, dev_errors = _derive_deviation(record)
    record.deviation = deviation
    if dev_errors:
        record.parse_errors.extend(dev_errors)

    return record


def parse_all(input_csv: str) -> list:
    """Parse every case referenced by input_csv into a list of FailureRecord
    (in the same order as the CSV rows), with deviation already filled in.

    Paths for the three logs are relative to the directory containing
    input_csv and are resolved to absolute paths here. Cases are parsed in
    parallel, one job per case, via multiprocessing.Pool.
    """
    base_dir = os.path.dirname(os.path.abspath(input_csv))

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        header_map = {_normalize_header(h): h for h in reader.fieldnames}

        def col(row, *names):
            for n in names:
                key = header_map.get(n)
                if key is not None:
                    return row[key]
            raise KeyError(f"none of {names} found in header {reader.fieldnames}")

        rows = list(reader)

    jobs = []
    for row in rows:
        case_id = int(str(col(row, "case")).strip())
        regr_path = os.path.join(base_dir, str(col(row, "regr log")).strip())
        sim_path = os.path.join(base_dir, str(col(row, "sim log")).strip())
        trace_path = os.path.join(base_dir, str(col(row, "trace log")).strip())
        jobs.append((case_id, regr_path, sim_path, trace_path))

    if not jobs:
        return []

    n_procs = min(8, os.cpu_count() or 1, len(jobs))
    if n_procs <= 1:
        records = [_parse_one_case(*job) for job in jobs]
    else:
        with multiprocessing.Pool(processes=n_procs) as pool:
            records = pool.starmap(_parse_one_case, jobs)

    return records


def run(input_csv: str, output_csv: str, k: int) -> None:
    """Parse -> signature -> cluster -> write output.csv (Case, bucket)."""
    from rfb.signature import build_signatures
    from rfb.cluster import cluster as cluster_signatures

    records = parse_all(input_csv)
    signatures = build_signatures(records)
    labels = cluster_signatures(signatures, k)

    records_sorted = sorted(records, key=lambda r: r.case_id)

    out_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "bucket"])
        for record in records_sorted:
            writer.writerow([record.case_id, labels[record.case_id]])
