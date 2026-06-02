from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from woodelf.core.cube_metric import BanzhafValues, CubeMetric
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.personalized_woodelf import personalized_baseline_woodelf
from woodelf.woodelf_sparse import woodelf_sparse


def _linreg_coeffs(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Return (const, coef) for the OLS line y = const + coef * x."""
    r = x.corr(y)
    if np.isnan(r) or x.std() == 0:
        return float(y.mean()), 0.0
    coef = r * y.std() / x.std()
    const = y.mean() - coef * x.mean()
    return float(const), float(coef)


def _best_anchor(f: str, candidates: List[str], consumer_data: pd.DataFrame) -> Tuple[str, float, float, float]:
    """Find the candidate with the highest |Pearson r| to feature f. Returns (anchor, abs_corr, const, coef)."""
    best_abs_corr = -1.0
    best_f_sel = candidates[0]
    for f_sel in candidates:
        abs_corr = abs(consumer_data[f].corr(consumer_data[f_sel]))
        if np.isnan(abs_corr):
            abs_corr = 0.0
        if abs_corr > best_abs_corr:
            best_abs_corr = abs_corr
            best_f_sel = f_sel
    const, coef = _linreg_coeffs(consumer_data[best_f_sel], consumer_data[f])
    return best_f_sel, best_abs_corr, const, coef


def _mean_abs(values: Dict[str, np.ndarray], features: List[str], n: int) -> Dict[str, float]:
    return {f: float(np.mean(np.abs(values.get(f, np.zeros(n))))) for f in features}


def feature_selection_ranking(
    model,
    consumer_data: pd.DataFrame,
    initial_selection: Optional[List[str]] = None,
    metric: Optional[CubeMetric] = None,
    GPU: bool = False,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Ranks features by their marginal contribution under personalized baselines.

    Starting from an initial selection of "anchor" features (whose background equals their consumer
    values), each remaining feature's background is set to a linear approximation via its most
    correlated anchor. The feature with the highest mean absolute personalized Banzhaf value is
    then selected, added to the ranking, becomes an anchor itself, and the process repeats.

    Returns:
        ranking: all feature names in selection order. Initial selection features are prepended —
            in their original order if provided, or sorted by path-dependent value if auto-computed.
        values_at_selection: maps each feature to its metric values at selection time. For
            auto-computed initial selection this is the path-dependent value; for user-provided
            initial selection these entries are absent.
    """
    if metric is None:
        metric = BanzhafValues()

    all_features = list(consumer_data.columns)
    n = len(consumer_data)

    model_obj = load_decision_tree_ensemble_model(model, all_features)

    # --- Determine initial selection ---
    initial_selection_values: Optional[Dict[str, np.ndarray]] = None
    if initial_selection is None:
        values = woodelf_sparse(model_obj, consumer_data, None, metric, GPU=GPU, model_was_loaded=True)
        mean_abs_values = _mean_abs(values, all_features, n)
        top_k = 3 if len(all_features) <= 30 else 5
        initial_selection = sorted(mean_abs_values, key=mean_abs_values.get, reverse=True)[:top_k]
        initial_selection_values = {f: values[f].copy() for f in initial_selection if f in values}

    selected = list(initial_selection)
    remaining = [f for f in all_features if f not in selected]

    if not remaining:
        return list(initial_selection), initial_selection_values or {}

    # --- Build initial background B ---
    # selected features: B[f] = C[f]  (already true in a copy of consumer_data)
    # remaining features: B[f] = const + coef * C[best_anchor]
    B = consumer_data.copy()

    # For each remaining feature track (anchor, abs_corr, const, coef)
    anchor_info: Dict[str, Tuple[str, float, float, float]] = {}
    for f in remaining:
        anchor, abs_corr, const, coef = _best_anchor(f, selected, consumer_data)
        anchor_info[f] = (anchor, abs_corr, const, coef)
        B[f] = const + coef * consumer_data[anchor]

    # --- Initial personalized baseline run ---
    result = personalized_baseline_woodelf(
        model_obj, consumer_data, B, metric, GPU=GPU, model_was_loaded=True, verbose=False
    )

    scores = _mean_abs(result, remaining, n)

    zero_contrib_features: List[str] = []
    for f in list(remaining):
        if scores[f] == 0.0:
            zero_contrib_features.append(f)
            remaining.remove(f)

    ranking: List[str] = list(initial_selection)
    ranking_values: Dict[str, np.ndarray] = initial_selection_values or {}

    if not remaining:
        for f in reversed(zero_contrib_features):
            ranking.append(f)
            ranking_values[f] = np.zeros(n)
        return ranking, ranking_values

    top_f = max(remaining, key=scores.get)

    # --- Selection loop ---
    while remaining:
        ranking.append(top_f)
        ranking_values[top_f] = result.get(top_f, np.zeros(n)).copy()
        remaining.remove(top_f)

        for f in list(remaining):
            if scores[f] == 0.0:
                zero_contrib_features.append(f)
                remaining.remove(f)

        if not remaining:
            break

        # Update B for remaining features where top_f is a better anchor
        for f in remaining:
            _, curr_best_corr, _, _ = anchor_info[f]
            abs_corr_with_top_f = abs(consumer_data[f].corr(consumer_data[top_f]))
            if np.isnan(abs_corr_with_top_f):
                abs_corr_with_top_f = 0.0
            if abs_corr_with_top_f > curr_best_corr:
                const, coef = _linreg_coeffs(consumer_data[top_f], consumer_data[f])
                anchor_info[f] = (top_f, abs_corr_with_top_f, const, coef)
                B[f] = const + coef * consumer_data[top_f]

        # top_f is now selected: neutralize its background
        B[top_f] = consumer_data[top_f]
        selected.append(top_f)

        result = personalized_baseline_woodelf(
            model_obj, consumer_data, B, metric, GPU=GPU, model_was_loaded=True, verbose=False
        )

        scores = _mean_abs(result, remaining, n)
        top_f = max(remaining, key=scores.get)

    for f in reversed(zero_contrib_features):
        ranking.append(f)
        ranking_values[f] = np.zeros(n)

    return ranking, ranking_values
