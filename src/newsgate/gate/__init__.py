"""The incremental-value gate: choose price-only, news-enhanced, or abstain."""

from newsgate.gate.evidence import evidence_label
from newsgate.gate.gate import GATE_FEATURES, IncrementalValueGate
from newsgate.gate.runner import run_gate, run_gate_holdout
from newsgate.gate.selective import selective_risk_table

__all__ = ["GATE_FEATURES", "IncrementalValueGate", "evidence_label", "run_gate", "run_gate_holdout", "selective_risk_table"]
