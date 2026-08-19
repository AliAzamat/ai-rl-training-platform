from rlhf.preferences import PreferenceLog


def aggregate_pair(log: PreferenceLog, a: str, b: str) -> dict | None:
    """Combine all judgments about pair (a, b) into one label with confidence.

    Returns {winner, loser, confidence, n} or None when there's a tie/no data.
    """
    prefs = log.for_pair(a, b)
    if not prefs:
        return None
    votes_a = sum(1 for p in prefs if p.winner == a)
    votes_b = len(prefs) - votes_a
    if votes_a == votes_b:
        return None                  # a tie carries no training signal — discard
    winner, loser, win_votes = (a, b, votes_a) if votes_a > votes_b else (b, a, votes_b)
    return {
        "winner": winner,
        "loser": loser,
        "confidence": win_votes / len(prefs),    # fraction who agreed
        "n": len(prefs),
    }
