"""Comm metrics: the exchange must report real bytes and wall time."""
import json, logging, re
from panoengine.decentralized.async_diloco import AsyncDiLoCo

def test_emit_comm_metrics_shape(caplog):
    w = AsyncDiLoCo.__new__(AsyncDiLoCo)
    w._sync_every = 500
    w._exchange_count = 0
    w._comm_seconds_total = 0.0
    w._comm_bytes_up_total = 0
    w._comm_bytes_down_total = 0
    with caplog.at_level(logging.INFO):
        w._emit_comm_metrics(2.0, 600_000_000, 1_200_000_000, None)
        w._emit_comm_metrics(3.0, 600_000_000, 1_200_000_000, None)
    lines = [r.getMessage() for r in caplog.records if "PFMETRICS" in r.getMessage()]
    assert len(lines) == 2
    first = json.loads(re.search(r"PFMETRICS (\{.*\})", lines[0]).group(1))
    assert first["step"] == 500 and first["comm/exchange"] == 1
    assert first["comm/bytes_total"] == 1_800_000_000
    assert abs(first["comm/mbps"] - 1_800_000_000 * 8 / 2.0 / 1e6) < 1
    second = json.loads(re.search(r"PFMETRICS (\{.*\})", lines[1]).group(1))
    assert second["step"] == 1000
    assert second["comm/seconds_cumulative"] == 5.0        # accumulates
    assert second["comm/gb_cumulative"] == 3.6             # 2 x 1.8 GB
