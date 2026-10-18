"""sim.log(.gz) parser. See docs/ARCHITECTURE.md section 4.5."""

from __future__ import annotations

import os
import re

from rfb.io_utils import open_text, normalize
from rfb.record import SimInfo, UvmMsg

_UVM_LINE_RE = re.compile(
    r"^(UVM_\w+)\s+(\S+)\((\d+)\)\s+@\s+\d+:\s+(\S+)\s+\[([^\]]*)\]\s?(.*)$"
)
_TESTNAME_RE = re.compile(r"\+UVM_TESTNAME=(\S+)")
_VERDICT_RE = re.compile(r"^--- RISC-V UVM TEST (PASSED|FAILED) ---\s*$")
_SEVERITY_HDR = "** Report counts by severity"
_ID_HDR = "** Report counts by id"
_SEVERITY_ROW_RE = re.compile(r"^(UVM_\w+)\s*:\s*(\d+)\s*$")
_ID_ROW_RE = re.compile(r"^\[([^\]]*)\]\s+(\d+)\s*$")

_ASSERT_PREFIX = "core_ibex_tb_top.dut."
_ASSERT_BRACKET_RE = re.compile(r"^\[([^\]]*)\]")

_FIRST_ERRORS_LIMIT = 5


def _build_msg(severity: str, file_field: str, line_field: str, scope: str, id_: str, message: str, raw: str) -> UvmMsg:
    file_base = os.path.basename(file_field)
    line_no = int(line_field) if line_field else None
    if id_ == "ASSERT FAILED":
        m = _ASSERT_BRACKET_RE.match(message)
        if m:
            name = m.group(1)
            if name.startswith(_ASSERT_PREFIX):
                name = name[len(_ASSERT_PREFIX) :]
            template = name
        else:
            template = normalize(message)
    else:
        template = normalize(message)
    return UvmMsg(
        severity=severity,
        file=file_base,
        line=line_no,
        scope=scope,
        id=id_,
        template=template,
        raw=raw,
    )


def parse_sim(path: str) -> SimInfo:
    test_name: str | None = None
    verdict: str | None = None
    fatals: list[UvmMsg] = []
    first_errors: list[UvmMsg] = []
    error_templates: dict[str, int] = {}
    actual_counts: dict[str, int] = {}
    report_counts: dict[str, int] = {}
    report_ids: dict[str, int] = {}
    line_count = 0

    section = None  # None | "severity" | "id"

    with open_text(path) as f:
        for line in f:
            line_count += 1
            line = line.rstrip("\n").rstrip("\r")

            if test_name is None:
                m = _TESTNAME_RE.search(line)
                if m:
                    test_name = m.group(1)

            m = _VERDICT_RE.match(line)
            if m:
                verdict = m.group(1)
                continue

            stripped = line.strip()
            if stripped == _SEVERITY_HDR:
                section = "severity"
                continue
            if stripped == _ID_HDR:
                section = "id"
                continue
            if section == "severity":
                m = _SEVERITY_ROW_RE.match(stripped)
                if m:
                    report_counts[m.group(1)] = int(m.group(2))
                    continue
                else:
                    section = None
            elif section == "id":
                m = _ID_ROW_RE.match(stripped)
                if m:
                    report_ids[m.group(1)] = int(m.group(2))
                    continue
                else:
                    section = None

            if line.startswith("UVM_WARNING"):
                actual_counts["UVM_WARNING"] = actual_counts.get("UVM_WARNING", 0) + 1
                continue

            if line.startswith("UVM_FATAL") or line.startswith("UVM_ERROR"):
                severity_guess = "UVM_FATAL" if line.startswith("UVM_FATAL") else "UVM_ERROR"
                m = _UVM_LINE_RE.match(line)
                if not m:
                    actual_counts[severity_guess] = actual_counts.get(severity_guess, 0) + 1
                    continue
                severity, file_field, line_field, scope, id_, message = m.groups()
                actual_counts[severity] = actual_counts.get(severity, 0) + 1
                msg = _build_msg(severity, file_field, line_field, scope, id_, message, line)
                if severity == "UVM_FATAL":
                    fatals.append(msg)
                else:  # UVM_ERROR
                    error_templates[msg.template] = error_templates.get(msg.template, 0) + 1
                    if len(first_errors) < _FIRST_ERRORS_LIMIT:
                        first_errors.append(msg)

    counts = report_counts if report_counts else actual_counts

    return SimInfo(
        test_name=test_name,
        verdict=verdict,
        fatals=fatals,
        first_errors=first_errors,
        error_templates=error_templates,
        counts=counts,
        report_ids=report_ids,
        line_count=line_count,
    )
