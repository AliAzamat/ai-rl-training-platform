from dataclasses import dataclass
from rlhf.preferences import PreferenceLog
from rlhf.agreement import mean_agreement


@dataclass
class QualityThresholds:
    """The lines below which feedback quality is considered unhealthy."""
    min_agreement: float = 0.65
    min_prefs_per_pair: float = 2.0


def quality_report(log: PreferenceLog, pairs: list[tuple[str, str]],
                   th: QualityThresholds = QualityThresholds()) -> dict:
    """A health snapshot of the current preference data, with flags."""
    agreement = mean_agreement(log, pairs)
    counts = [len(log.for_pair(a, b)) for a, b in pairs]
    avg_per_pair = sum(counts) / len(counts) if counts else 0.0
    flags = []
    if agreement < th.min_agreement:
        flags.append("LOW_AGREEMENT")
    if avg_per_pair < th.min_prefs_per_pair:
        flags.append("UNDER_SAMPLED")
    return {
        "mean_agreement": round(agreement, 3),
        "avg_prefs_per_pair": round(avg_per_pair, 2),
        "flags": flags,
        "healthy": not flags,
    }
