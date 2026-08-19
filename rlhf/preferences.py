from dataclasses import dataclass, field


@dataclass
class Preference:
    """One annotator's judgment: between rollout A and B, which is better."""
    task_id: str
    rollout_a: str
    rollout_b: str
    annotator: str
    winner: str                      # the rollout_id the annotator preferred


class PreferenceLog:
    """All preference judgments collected so far."""

    def __init__(self) -> None:
        self.prefs: list[Preference] = []

    def record(self, pref: Preference) -> None:
        if pref.winner not in (pref.rollout_a, pref.rollout_b):
            raise ValueError("winner must be one of the two compared rollouts")
        self.prefs.append(pref)

    def for_pair(self, a: str, b: str) -> list[Preference]:
        """Every judgment about this exact unordered pair of rollouts."""
        pair = {a, b}
        return [p for p in self.prefs if {p.rollout_a, p.rollout_b} == pair]
