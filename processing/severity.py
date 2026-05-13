from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class DefectTier(str, Enum):
    noise = "noise"
    negligible = "negligible"
    low = "low"
    moderate = "moderate"
    severe = "severe"
    critical = "critical"


_NOISE_FLOOR_G = 1.50   # calibrated from rad.txt: observed max at rest = 1.44G
_TIERS = [
    (7.0, DefectTier.critical),
    (5.0, DefectTier.severe),
    (3.5, DefectTier.moderate),
    (2.5, DefectTier.low),
    (1.5, DefectTier.negligible),
]


@dataclass(frozen=True)
class SeverityResult:
    tier: DefectTier
    defect_type: str
    severity_score: float   # 0.0–10.0, matches defects table scale
    should_ingest: bool
    peak_g: float
    duration_ms: float


def score_event(
    peak_g: float,
    duration_us: float,
    speed_mps: float | None = None,
) -> SeverityResult:
    """Score an IMU event.

    Args:
        peak_g:      Peak G-force magnitude (already absolute value).
        duration_us: Event duration in **microseconds** as sent by the device.
        speed_mps:   Vehicle speed in m/s (optional, used for type hinting).

    Returns:
        SeverityResult with tier, type, score, and ingest flag.
    """
    duration_ms = duration_us / 1_000.0

    if peak_g < _NOISE_FLOOR_G:
        return SeverityResult(
            tier=DefectTier.noise,
            defect_type="normal",
            severity_score=0.0,
            should_ingest=False,
            peak_g=peak_g,
            duration_ms=duration_ms,
        )

    tier = DefectTier.negligible
    for threshold, t in _TIERS:
        if peak_g >= threshold:
            tier = t
            break

    defect_type = _classify_type(peak_g, duration_ms, speed_mps)
    score = _compute_score(peak_g, duration_ms, tier)

    return SeverityResult(
        tier=tier,
        defect_type=defect_type,
        severity_score=score,
        should_ingest=(tier != DefectTier.negligible),
        peak_g=peak_g,
        duration_ms=duration_ms,
    )


def should_ingest(result: SeverityResult) -> bool:
    return result.should_ingest


def _classify_type(peak_g: float, duration_ms: float, speed_mps: float | None) -> str:
    if duration_ms < 5.0:
        return "crack"
    if duration_ms > 150.0:
        return "rough_pavement"
    if 5.0 <= duration_ms <= 30.0 and peak_g >= 5.0:
        return "pothole"
    if 8.0 <= duration_ms <= 25.0 and peak_g < 3.0:
        return "expansion_joint"
    return "pothole"


def _compute_score(peak_g: float, duration_ms: float, tier: DefectTier) -> float:
    _tier_base = {
        DefectTier.critical: 8.0,
        DefectTier.severe: 6.0,
        DefectTier.moderate: 4.0,
        DefectTier.low: 2.0,
        DefectTier.negligible: 0.5,
    }
    base = _tier_base.get(tier, 0.0)
    # Duration bonus: longer events indicate road surface problems, not point impacts
    duration_bonus = min(1.5, duration_ms / 200.0)
    return round(min(10.0, base + duration_bonus), 2)
