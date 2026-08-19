from rlhf.preferences import PreferenceLog


def pair_agreement(log: PreferenceLog, a: str, b: str) -> float | None:
    """Fraction of annotators on this pair who sided with the majority.

    1.0 = unanimous, 0.5 = an even split (maximum disagreement on two options).
    Returns None when fewer than two judgments exist.
    """
    prefs = log.for_pair(a, b)
    if len(prefs) < 2:
        return None
    votes_a = sum(1 for p in prefs if p.winner == a)
    votes_b = len(prefs) - votes_a
    majority = max(votes_a, votes_b)
    return majority / len(prefs)


def mean_agreement(log: PreferenceLog, pairs: list[tuple[str, str]]) -> float:
    """Average agreement across many pairs — a dataset-wide quality signal."""
    scores = [pair_agreement(log, a, b) for a, b in pairs]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else 0.0
