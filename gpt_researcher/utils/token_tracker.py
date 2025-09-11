"""
Global token tracking for async operations.
Aggregates tokens and cost across the entire process without per-researcher context.
"""
from typing import Dict


class TokenTracker:
    """Global token tracker that aggregates by model and overall totals."""

    # Overall totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    # Per-model totals: { model: {"input": int, "output": int, "cost": float} }
    per_model_totals: Dict[str, Dict[str, float | int]] = {}

    @staticmethod
    def track_tokens(model: str, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Track tokens and cost globally (and per model)."""
        # Update global totals
        TokenTracker.total_input_tokens += int(input_tokens or 0)
        TokenTracker.total_output_tokens += int(output_tokens or 0)
        TokenTracker.total_cost += float(cost or 0.0)

        # Update per-model totals
        if model not in TokenTracker.per_model_totals:
            TokenTracker.per_model_totals[model] = {"input": 0, "output": 0, "cost": 0.0}
        TokenTracker.per_model_totals[model]["input"] += int(input_tokens or 0)
        TokenTracker.per_model_totals[model]["output"] += int(output_tokens or 0)
        TokenTracker.per_model_totals[model]["cost"] += float(cost or 0.0)

    @staticmethod
    def get_totals() -> Dict[str, float | int]:
        return {
            "input_tokens": TokenTracker.total_input_tokens,
            "output_tokens": TokenTracker.total_output_tokens,
            "cost": TokenTracker.total_cost,
        }

    @staticmethod
    def get_per_model_totals() -> Dict[str, Dict[str, float | int]]:
        return TokenTracker.per_model_totals


