"""EER and min t-DCF, ported from the official ASVspoof2019 CM scoring toolkit
(``calculate_tDCF_EER.py`` / ``eval_metrics.py``, ASVspoof consortium) so results are
comparable to published baselines rather than reimplemented from scratch.
"""

from __future__ import annotations

import numpy as np

# Standard ASVspoof2019 LA evaluation-plan cost model.
ASVSPOOF19_LA_COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": 0.95 * 0.99,
    "Pnon": 0.95 * 0.01,
    "Cmiss": 1,
    "Cfa": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frr, far, thresholds) arrays tracing the detection error trade-off curve."""
    target_scores = np.asarray(target_scores, dtype=float)
    nontarget_scores = np.asarray(nontarget_scores, dtype=float)

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> tuple[float, float]:
    """Return (eer, threshold) — the point where false-reject rate == false-accept rate."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])


def obtain_asv_error_rates(
    tar_asv: np.ndarray, non_asv: np.ndarray, spoof_asv: np.ndarray, asv_threshold: float
) -> tuple[float, float, float | None]:
    """ASV operating-point error rates at its own EER threshold, needed for min t-DCF."""
    tar_asv = np.asarray(tar_asv, dtype=float)
    non_asv = np.asarray(non_asv, dtype=float)
    spoof_asv = np.asarray(spoof_asv, dtype=float)

    pfa_asv = float(np.sum(non_asv >= asv_threshold) / non_asv.size)
    pmiss_asv = float(np.sum(tar_asv < asv_threshold) / tar_asv.size)

    if spoof_asv.size == 0:
        pmiss_spoof_asv = None
    else:
        pfa_spoof_asv = float(np.sum(spoof_asv >= asv_threshold) / spoof_asv.size)
        pmiss_spoof_asv = 1 - pfa_spoof_asv

    return pfa_asv, pmiss_asv, pmiss_spoof_asv


def compute_tdcf(
    bonafide_score_cm: np.ndarray,
    spoof_score_cm: np.ndarray,
    pfa_asv: float,
    pmiss_asv: float,
    pmiss_spoof_asv: float | None,
    cost_model: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized t-DCF curve for a countermeasure (CM), given the ASV system's operating
    point. Returns (tdcf_norm, cm_thresholds) — take min(tdcf_norm) for the min t-DCF.
    """
    cost_model = cost_model or ASVSPOOF19_LA_COST_MODEL

    if pmiss_spoof_asv is None:
        raise ValueError(
            "pmiss_spoof_asv is required (min t-DCF needs the ASV system's error rate "
            "against spoofed trials, not just bona fide vs. nontarget)."
        )

    pmiss_cm, pfa_cm, cm_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    c1 = cost_model["Ptar"] * (cost_model["Cmiss_cm"] - cost_model["Cmiss"] * pmiss_asv) - (
        cost_model["Pnon"] * cost_model["Cfa"] * pfa_asv
    )
    c2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - pmiss_spoof_asv)

    if c1 < 0 or c2 < 0:
        raise ValueError(
            "Negative t-DCF weights (C1/C2) — check the ASV operating point and cost model; "
            "this should not happen with the standard ASVspoof2019 LA cost model."
        )

    tdcf = c1 * pmiss_cm + c2 * pfa_cm
    tdcf_norm = tdcf / min(c1, c2)

    return tdcf_norm, cm_thresholds


def compute_min_tdcf(
    bonafide_score_cm: np.ndarray,
    spoof_score_cm: np.ndarray,
    pfa_asv: float,
    pmiss_asv: float,
    pmiss_spoof_asv: float,
    cost_model: dict | None = None,
) -> float:
    """Convenience wrapper: min t-DCF as a single number."""
    tdcf_norm, _ = compute_tdcf(
        bonafide_score_cm, spoof_score_cm, pfa_asv, pmiss_asv, pmiss_spoof_asv, cost_model
    )
    return float(np.min(tdcf_norm))
