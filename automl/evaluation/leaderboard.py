"""
Leaderboard: Tracks and ranks models by evaluation score.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class Leaderboard:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(
        self,
        model_name: str,
        metrics: Dict[str, float],
        primary_score: float,
        model_obj: Any = None,
        extra_info: Optional[Dict] = None,
    ):
        entry = {
            "model_name": model_name,
            "primary_score": primary_score,
            **metrics,
            "_model": model_obj,
        }
        if extra_info:
            entry.update(extra_info)
        self.entries.append(entry)

    def rank(self) -> "Leaderboard":
        self.entries.sort(key=lambda e: e["primary_score"], reverse=True)
        return self

    def best(self) -> Optional[Dict[str, Any]]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e["primary_score"])

    def to_dataframe(self) -> pd.DataFrame:
        display_entries = []
        for e in self.entries:
            row = {k: v for k, v in e.items() if k != "_model"}
            display_entries.append(row)
        df = pd.DataFrame(display_entries)
        if not df.empty:
            df = df.sort_values("primary_score", ascending=False).reset_index(drop=True)
            df.index += 1
        return df

    def print(self):
        df = self.to_dataframe()
        print("\n" + "=" * 70)
        print("                        LEADERBOARD")
        print("=" * 70)
        print(df.drop(columns=["primary_score"], errors="ignore").to_string())
        print("=" * 70 + "\n")
