import numpy as np
import pytest

from muffle.metrics import (
    ASVSPOOF19_LA_COST_MODEL,
    compute_eer,
    compute_min_tdcf,
    obtain_asv_error_rates,
)


def test_eer_perfect_separation_is_zero():
    target = np.array([3.0, 4.0, 5.0])
    nontarget = np.array([-1.0, 0.0, 1.0])
    eer, _ = compute_eer(target, nontarget)
    assert eer == pytest.approx(0.0, abs=1e-9)


def test_eer_identical_distributions_is_half():
    scores = np.array([0.1, 0.4, 0.6, 0.9, 1.2, 1.5])
    eer, _ = compute_eer(scores, scores)
    assert eer == pytest.approx(0.5, abs=1e-9)


def test_eer_invariant_to_monotonic_score_transform():
    rng = np.random.default_rng(0)
    target = rng.normal(loc=1.0, scale=1.0, size=200)
    nontarget = rng.normal(loc=-1.0, scale=1.0, size=200)

    eer_raw, _ = compute_eer(target, nontarget)
    eer_transformed, _ = compute_eer(np.exp(target), np.exp(nontarget))  # monotonic increasing

    assert eer_raw == pytest.approx(eer_transformed, abs=1e-6)


def test_eer_known_hand_computed_case():
    # Interleaved scores {1(non), 2(tar), 3(non), 4(tar)}: hand-tracing compute_det_curve's
    # sort/cumsum gives frr=[0,0,.5,.5,1], far=[1,.5,.5,0,0] over thresholds
    # [0.999,1,2,3,4], so |frr-far| is minimized (=0) at threshold=2 with frr=far=0.5.
    target = np.array([2.0, 4.0])
    nontarget = np.array([1.0, 3.0])
    eer, threshold = compute_eer(target, nontarget)
    assert eer == pytest.approx(0.5, abs=1e-9)
    assert threshold == pytest.approx(2.0, abs=1e-9)


def test_min_tdcf_lower_for_better_countermeasure():
    rng = np.random.default_rng(1)
    bona = rng.normal(loc=2.0, scale=1.0, size=500)

    good_spoof = rng.normal(loc=-2.0, scale=1.0, size=500)  # well separated from bona fide
    bad_spoof = rng.normal(loc=1.5, scale=1.0, size=500)  # heavily overlapping

    tar_asv = rng.normal(loc=2.0, scale=1.0, size=500)
    non_asv = rng.normal(loc=-2.0, scale=1.0, size=500)
    spoof_asv = rng.normal(loc=0.0, scale=1.0, size=500)

    _, asv_threshold = compute_eer(tar_asv, non_asv)
    pfa_asv, pmiss_asv, pmiss_spoof_asv = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    min_tdcf_good = compute_min_tdcf(
        bona, good_spoof, pfa_asv, pmiss_asv, pmiss_spoof_asv, ASVSPOOF19_LA_COST_MODEL
    )
    min_tdcf_bad = compute_min_tdcf(
        bona, bad_spoof, pfa_asv, pmiss_asv, pmiss_spoof_asv, ASVSPOOF19_LA_COST_MODEL
    )

    assert min_tdcf_good < min_tdcf_bad
    assert min_tdcf_good >= 0.0
