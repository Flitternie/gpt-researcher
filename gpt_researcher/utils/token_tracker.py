"""
Global token tracking for async operations.
Aggregates tokens and cost across the entire process without per-researcher context.
Adds optional usage tagging (e.g., "research", "monitor") to differentiate sources.
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

    # Per-usage totals: { usage_tag: {"input": int, "output": int, "cost": float} }
    per_usage_totals: Dict[str, Dict[str, float | int]] = {}

    @staticmethod
    def track_tokens(model: str, input_tokens: int, output_tokens: int, cost: float = 0.0, usage_tag: str | None = None):
        """Track tokens and cost globally (and per model/usage).

        Args:
            model: LLM model identifier
            input_tokens: Count of input tokens
            output_tokens: Count of output tokens
            cost: Monetary cost for the call
            usage_tag: Optional usage classification (e.g., "research", "monitor", "planning").
        """
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

        # Update per-usage totals
        tag = usage_tag or "unspecified"
        if tag not in TokenTracker.per_usage_totals:
            TokenTracker.per_usage_totals[tag] = {"input": 0, "output": 0, "cost": 0.0}
        TokenTracker.per_usage_totals[tag]["input"] += int(input_tokens or 0)
        TokenTracker.per_usage_totals[tag]["output"] += int(output_tokens or 0)
        TokenTracker.per_usage_totals[tag]["cost"] += float(cost or 0.0)

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

    @staticmethod
    def get_per_usage_totals() -> Dict[str, Dict[str, float | int]]:
        return TokenTracker.per_usage_totals


