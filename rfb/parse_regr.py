"""regr.log parser. See docs/ARCHITECTURE.md section 4.1."""

from __future__ import annotations

import re

from rfb.io_utils import open_text
from rfb.record import Mismatch, RegrInfo

_MISMATCH_HDR_RE = re.compile(r"^Mismatch\[(\d+)\]:$")
_IBEX_RE = re.compile(r"^ibex\[(\d+)\]\s*:\s*pc\[([0-9a-fA-F]+)\]\s*(.*)$")
_SPIKE_RE = re.compile(r"^spike\[(\d+)\]\s*:\s*pc\[([0-9a-fA-F]+)\]\s*(.*)$")
_FAILED_COUNT_RE = re.compile(r"^\[FAILED\]:\s*(\d+)\s*matched,\s*(\d+)\s*mismatch\s*$")
_TEST_FAILED_RE = re.compile(r"^(\S+)\.0\s*:\s*\[FAILED\]\s*$")
_TRAILING_RD_RE = re.compile(r":\s*(\w+):([0-9a-fA-F]+)\s*$")


def _split_rd_val(text: str) -> tuple[str | None, int | None]:
    m = _TRAILING_RD_RE.search(text)
    if not m:
        return None, None
    return m.group(1), int(m.group(2), 16)


def _spike_pc(hex_str: str) -> int:
    if hex_str.startswith("ffffffff"):
        hex_str = hex_str[len("ffffffff") :]
    return int(hex_str, 16)


def parse_regr(path: str) -> RegrInfo:
    raw: list[str] = []
    mismatches: list[Mismatch] = []
    matched: int | None = None
    mismatch_count: int | None = None
    test_name: str | None = None
    kind = "UNKNOWN"

    pending_ibex: tuple[int, int, str] | None = None
    cur_n: int | None = None

    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            raw.append(line.rstrip())
            stripped = line.rstrip()

            m = _MISMATCH_HDR_RE.match(stripped)
            if m:
                cur_n = int(m.group(1))
                pending_ibex = None
                kind = "MISMATCH"
                continue

            m = _IBEX_RE.match(stripped)
            if m:
                ibex_idx = int(m.group(1))
                ibex_pc = int(m.group(2), 16)
                ibex_text = m.group(3).strip()
                pending_ibex = (ibex_idx, ibex_pc, ibex_text)
                continue

            m = _SPIKE_RE.match(stripped)
            if m and pending_ibex is not None:
                spike_idx = int(m.group(1))
                spike_pc = _spike_pc(m.group(2))
                spike_text = m.group(3).strip()
                ibex_idx, ibex_pc, ibex_text = pending_ibex
                ibex_rd, ibex_val = _split_rd_val(ibex_text)
                spike_rd, spike_val = _split_rd_val(spike_text)
                mismatches.append(
                    Mismatch(
                        n=cur_n if cur_n is not None else len(mismatches) + 1,
                        ibex_idx=ibex_idx,
                        ibex_pc=ibex_pc,
                        ibex_text=ibex_text,
                        ibex_rd=ibex_rd,
                        ibex_val=ibex_val,
                        spike_idx=spike_idx,
                        spike_pc=spike_pc,
                        spike_text=spike_text,
                        spike_rd=spike_rd,
                        spike_val=spike_val,
                    )
                )
                pending_ibex = None
                continue

            m = _FAILED_COUNT_RE.match(stripped)
            if m:
                matched = int(m.group(1))
                mismatch_count = int(m.group(2))
                continue

            m = _TEST_FAILED_RE.match(stripped)
            if m:
                test_name = m.group(1)
                if kind == "UNKNOWN":
                    kind = "TEST_FAILED"
                continue

    return RegrInfo(
        kind=kind,
        test_name=test_name,
        mismatches=mismatches,
        matched=matched,
        mismatch_count=mismatch_count,
        raw=raw,
    )
