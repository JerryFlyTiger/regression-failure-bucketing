"""parse_trace must give identical results whether a trace is held in memory
once or re-streamed per pass (files above _IN_MEMORY_MAX_BYTES)."""

import gzip

import pytest

from rfb import parse_trace as pt
from rfb.parse_regr import parse_regr

SET2 = "testcase/2026_08_20_samples/problem/benchmark_set_2"


@pytest.mark.parametrize("case", ["case_1", "case_10", "case_14"])
def test_streaming_path_matches_in_memory_path(monkeypatch, case):
    regr = parse_regr(f"{SET2}/{case}/regr.log")
    trace = f"{SET2}/{case}/trace.log.gz"
    in_memory = pt.parse_trace(trace, regr)
    monkeypatch.setattr(pt, "_IN_MEMORY_MAX_BYTES", 0)
    calls = []
    real = pt._stream_lines
    monkeypatch.setattr(pt, "_stream_lines", lambda p: calls.append(p) or real(p))
    streamed = pt.parse_trace(trace, regr)
    assert len(calls) == 2
    assert streamed.to_dict() == in_memory.to_dict()


def test_decompressed_size_uses_gzip_trailer(tmp_path):
    p = tmp_path / "t.log.gz"
    payload = b"x" * 100_000
    with gzip.open(p, "wb") as f:
        f.write(payload)
    assert pt._decompressed_size(str(p)) == len(payload)
    plain = tmp_path / "t.log"
    plain.write_bytes(payload)
    assert pt._decompressed_size(str(plain)) == len(payload)


def test_stream_lines_matches_iter_lines_without_trailing_newline(tmp_path):
    p = tmp_path / "t.log"
    p.write_text("a\nb\r\nc")
    assert list(pt._stream_lines(str(p))) == list(pt._iter_lines(pt._read_all_text(str(p))))


def _trace_payload(n_lines: int) -> bytes:
    """A small, highly-repetitive (so it compresses very well) but valid
    trace.log payload: header + n_lines identical data lines."""
    header = "Time\tCycle\tPC\tInsn\tDecoded instruction\tRegister and memory contents\n"
    line = "0\t0\t1000\t13\taddi\tx1,x0,1\tx1=0x00000001\n"
    return (header + line * n_lines).encode("utf-8")


def test_isize_wraparound_forces_streaming_with_correct_result(tmp_path, monkeypatch):
    # A genuine >4 GiB decompressed stream isn't practical to build in a
    # test, but its effect on ``_line_source`` -- ``_decompressed_size``
    # (the ISIZE pre-filter) reporting a small, *trusted-looking* size for
    # a decompressed stream that is actually much larger -- is exactly
    # reproduced by monkeypatching ``_decompressed_size`` itself to lie
    # about a real, larger gz payload. (Corrupting the gzip ISIZE trailer
    # bytes directly isn't viable: Python's gzip reader independently
    # verifies the true stream length against the trailer at EOF and
    # raises BadGzipFile on any mismatch, so it can't be fooled without
    # actually producing >4 GiB of data.)
    payload = _trace_payload(5000)
    p = tmp_path / "wrap.log.gz"
    with gzip.open(p, "wb") as f:
        f.write(payload)
    path = str(p)
    lying_isize = 200
    assert lying_isize < len(payload)

    monkeypatch.setattr(pt, "_decompressed_size", lambda _p: lying_isize)
    monkeypatch.setattr(pt, "_IN_MEMORY_MAX_BYTES", lying_isize + 10)
    calls = []
    real = pt._stream_lines
    monkeypatch.setattr(pt, "_stream_lines", lambda p: calls.append(p) or real(p))

    result = pt.parse_trace(path, None)
    assert len(calls) == 2  # the bounded read catches it despite the lying pre-filter
    assert result.n_instr == 5000

    monkeypatch.undo()
    reference = pt.parse_trace(path, None)
    assert result.to_dict() == reference.to_dict()


def test_bounded_read_over_threshold_discards_and_streams(tmp_path, monkeypatch):
    payload = _trace_payload(200)
    path = tmp_path / "plain.log"
    path.write_bytes(payload)
    monkeypatch.setattr(pt, "_IN_MEMORY_MAX_BYTES", 10)  # far below the real size
    calls = []
    real = pt._stream_lines
    monkeypatch.setattr(pt, "_stream_lines", lambda p: calls.append(p) or real(p))

    result = pt.parse_trace(str(path), None)
    assert len(calls) == 2
    assert result.n_instr == 200
