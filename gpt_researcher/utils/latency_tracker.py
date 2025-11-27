"""
Lightweight latency tracker for API calls (LLM and Search).
Tracks call latency and counts for performance profiling.
"""
from typing import Dict, List
import time


class LatencyTracker:
    """Global latency tracker for profiling API latency."""

    # Per-type totals: { "llm" | "search": {"count": int, "total_latency": float, "calls": List[float]} }
    per_type_latencies: Dict[str, Dict] = {}

    # Per-model/retriever totals: { model_or_retriever: {"count": int, "total_latency": float, "calls": List[float]} }
    per_source_latencies: Dict[str, Dict] = {}

    @staticmethod
    def track_latency(call_type: str, latency: float, source: str = None):
        """Track API call latency globally.

        Args:
            call_type: Type of call ("llm" or "search")
            latency: Latency in seconds
            source: Optional model name (for LLM) or retriever name (for search)
        """
        # Update per-type totals
        if call_type not in LatencyTracker.per_type_latencies:
            LatencyTracker.per_type_latencies[call_type] = {
                "count": 0,
                "total_latency": 0.0,
                "calls": []
            }
        LatencyTracker.per_type_latencies[call_type]["count"] += 1
        LatencyTracker.per_type_latencies[call_type]["total_latency"] += latency
        LatencyTracker.per_type_latencies[call_type]["calls"].append(latency)

        # Update per-source totals if source is provided
        if source:
            if source not in LatencyTracker.per_source_latencies:
                LatencyTracker.per_source_latencies[source] = {
                    "count": 0,
                    "total_latency": 0.0,
                    "calls": []
                }
            LatencyTracker.per_source_latencies[source]["count"] += 1
            LatencyTracker.per_source_latencies[source]["total_latency"] += latency
            LatencyTracker.per_source_latencies[source]["calls"].append(latency)

    @staticmethod
    def get_summary() -> Dict:
        """Get a summary of all latency data."""
        summary = {}
        
        # Add per-type summaries
        for call_type, data in LatencyTracker.per_type_latencies.items():
            if data["count"] > 0:
                calls = data["calls"]
                summary[call_type] = {
                    "count": data["count"],
                    "total_latency": data["total_latency"],
                    "avg_latency": data["total_latency"] / data["count"],
                    "min_latency": min(calls) if calls else 0,
                    "max_latency": max(calls) if calls else 0,
                }
        
        # Add per-source summaries
        by_source = {}
        for source, data in LatencyTracker.per_source_latencies.items():
            if data["count"] > 0:
                calls = data["calls"]
                by_source[source] = {
                    "count": data["count"],
                    "total_latency": data["total_latency"],
                    "avg_latency": data["total_latency"] / data["count"],
                    "min_latency": min(calls) if calls else 0,
                    "max_latency": max(calls) if calls else 0,
                }
        
        if by_source:
            summary["by_source"] = by_source
        
        return summary

    @staticmethod
    def get_formatted_summary() -> str:
        """Get a human-readable formatted summary of latency data."""
        summary = LatencyTracker.get_summary()
        
        if not summary:
            return "No latency data recorded."
        
        lines = ["API Latency Profile:"]
        lines.append("")
        
        # Overall summary by type
        for call_type in ["llm", "search"]:
            if call_type in summary:
                data = summary[call_type]
                lines.append(f"{call_type.upper()} Calls:")
                lines.append(f"  Total calls: {data['count']}")
                lines.append(f"  Total time:  {data['total_latency']:.2f}s")
                lines.append(f"  Avg latency: {data['avg_latency']:.3f}s")
                lines.append(f"  Min latency: {data['min_latency']:.3f}s")
                lines.append(f"  Max latency: {data['max_latency']:.3f}s")
                lines.append("")
        
        # Per-source breakdown
        if summary.get("by_source"):
            lines.append("By Source:")
            for source, data in sorted(summary["by_source"].items()):
                lines.append(f"  {source}:")
                lines.append(f"    Calls: {data['count']}, "
                           f"Avg: {data['avg_latency']:.3f}s, "
                           f"Min: {data['min_latency']:.3f}s, "
                           f"Max: {data['max_latency']:.3f}s")
        
        return "\n".join(lines)

    @staticmethod
    def reset():
        """Reset all tracked latency data."""
        LatencyTracker.per_type_latencies.clear()
        LatencyTracker.per_source_latencies.clear()

