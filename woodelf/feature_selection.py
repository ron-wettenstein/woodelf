from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from woodelf.core.cube_metric import BanzhafValues, CubeMetric
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.always_participating_woodelf import always_participating_delta_update, path_dependent_under_always_participating_features
from woodelf.personalized_woodelf import personalized_baseline_delta_update, personalized_baseline_woodelf
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


def _init_background(
    consumer_data: pd.DataFrame,
    remaining: List[str],
    baseline_init_scheme: str,
    anchor_info: Dict[str, Tuple[str, float, float, float]],
) -> pd.DataFrame:
    """
    Build the initial background B for remaining features.

    Selected features: B[f] = C[f] (copy of consumer, already set).
    Remaining features depend on the scheme:
      - pearson_correlation: B[f] = const + coef * C[best_anchor], taken from anchor_info.
      - median: B[f] = median(C[f]) — same constant for all rows.
      - mean:   B[f] = mean(C[f])  — same constant for all rows.
    """
    B = consumer_data.copy()
    if baseline_init_scheme == "pearson_correlation":
        for f in remaining:
            anchor, _, const, coef = anchor_info[f]
            B[f] = const + coef * consumer_data[anchor]
    elif baseline_init_scheme == "median":
        for f in remaining:
            B[f] = consumer_data[f].median()
    elif baseline_init_scheme == "mean":
        for f in remaining:
            B[f] = consumer_data[f].mean()
    return B


def feature_selection_ranking(
    model,
    consumer_data: pd.DataFrame,
    initial_selection: Optional[List[str]] = None,
    metric: Optional[CubeMetric] = None,
    GPU: bool = False,
    baseline_updating_scheme: str = "pearson_correlation",
    baseline_init_scheme: str = "pearson_correlation",
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Ranks features by their marginal contribution under personalized baselines.

    Starting from an initial selection of "anchor" features (whose background equals their consumer
    values), each remaining feature's background is set according to baseline_init_scheme. The feature
    with the highest mean absolute personalized Banzhaf value is then selected, added to the ranking,
    becomes an anchor itself, and the process repeats.

    Returns:
        ranking: all feature names in selection order. Initial selection features are prepended —
            in their original order if provided, or sorted by path-dependent value if auto-computed.
        values_at_selection: maps each feature to its metric values at selection time. For
            auto-computed initial selection this is the path-dependent value; for user-provided
            initial selection these entries are absent.

    @param baseline_init_scheme: How to set the initial background B for remaining features:
        - "pearson_correlation": B[f] = const + coef * C[best_anchor] via OLS (default).
        - "median": B[f] = median(C[f]) — same constant for all rows.
        - "mean":   B[f] = mean(C[f])  — same constant for all rows.
    @param baseline_updating_scheme: How to update B each time a new feature is selected:
        - "pearson_correlation": re-anchor each remaining feature to the best correlated selected
          feature using OLS (default).
        - "do_nothing": only neutralize B[top_f] = C[top_f]; leave all other baselines unchanged.
        - "median_correlation": not yet implemented.
    """
    _VALID_INIT_SCHEMES = {"pearson_correlation", "median", "mean"}
    if baseline_init_scheme not in _VALID_INIT_SCHEMES:
        raise ValueError(f"baseline_init_scheme must be one of {_VALID_INIT_SCHEMES}, got {baseline_init_scheme!r}")

    _VALID_UPDATE_SCHEMES = {"pearson_correlation", "do_nothing", "median_correlation"}
    if baseline_updating_scheme not in _VALID_UPDATE_SCHEMES:
        raise ValueError(f"baseline_updating_scheme must be one of {_VALID_UPDATE_SCHEMES}, got {baseline_updating_scheme!r}")
    if baseline_updating_scheme == "median_correlation":
        raise NotImplementedError("baseline_updating_scheme='median_correlation' is not yet implemented")

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
    anchor_info: Dict[str, Tuple[str, float, float, float]] = {}
    if baseline_init_scheme == "pearson_correlation" or baseline_updating_scheme == "pearson_correlation":
        for f in remaining:
            anchor_info[f] = _best_anchor(f, selected, consumer_data)

    B = _init_background(consumer_data, remaining, baseline_init_scheme, anchor_info)

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
    with tqdm(total=len(all_features), desc="Selecting features", initial=len(all_features) - len(remaining)) as pbar:
        pbar.refresh()
        while remaining:
            prev_remaining = len(remaining)

            ranking.append(top_f)
            ranking_values[top_f] = result.pop(top_f, np.zeros(n)).copy()
            remaining.remove(top_f)

            for f in list(remaining):
                if scores[f] == 0.0:
                    zero_contrib_features.append(f)
                    remaining.remove(f)

            pbar.update(prev_remaining - len(remaining))

            if not remaining:
                break

            # Determine which features will switch anchor — pre-compute before touching B
            changed_features: List[str] = []
            new_anchor_data: Dict[str, Tuple] = {}
            if baseline_updating_scheme == "pearson_correlation":
                for f in remaining:
                    _, curr_best_corr, _, _ = anchor_info[f]
                    abs_corr_with_top_f = abs(consumer_data[f].corr(consumer_data[top_f]))
                    if np.isnan(abs_corr_with_top_f):
                        abs_corr_with_top_f = 0.0
                    if abs_corr_with_top_f > curr_best_corr:
                        const, coef = _linreg_coeffs(consumer_data[top_f], consumer_data[f])
                        new_anchor_data[f] = (top_f, abs_corr_with_top_f, const, coef)
                        changed_features.append(f)

            # top_f included because B[top_f] is also changing (neutralized to consumer)
            features_subset_delta = [top_f] + changed_features

            # Build new B (apply anchor switches and neutralize top_f)
            new_B = B.copy()
            for f, (anchor, abs_corr, const, coef) in new_anchor_data.items():
                new_B[f] = const + coef * consumer_data[top_f]
            new_B[top_f] = consumer_data[top_f]

            result = personalized_baseline_delta_update(
                model_obj, consumer_data, B, new_B, features_subset_delta, result, metric,
                GPU=GPU, model_was_loaded=True,
            )

            for f, (anchor, abs_corr, const, coef) in new_anchor_data.items():
                anchor_info[f] = (anchor, abs_corr, const, coef)
            B = new_B
            selected.append(top_f)

            scores = _mean_abs(result, remaining, n)
            top_f = max(remaining, key=scores.get)

    for f in reversed(zero_contrib_features):
        ranking.append(f)
        ranking_values[f] = np.zeros(n)

    return ranking, ranking_values


def _get_tree_features(model) -> List[str]:
    seen = set()
    for tree in model.trees:
        for node in tree.bfs(including_leaves=False):
            seen.add(node.feature_name)
    return list(seen)


def path_dependent_feature_selection_ranking(
    model,
    consumer_data: pd.DataFrame,
    metric: Optional[CubeMetric] = None,
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Ranks features by their marginal contribution in the path-dependent restricted game.

    Starting from the standard path-dependent Banzhaf game (no always-participating features),
    at each step the most significant remaining feature is selected, added to always_participating,
    and values are updated exactly via always_participating_delta_update. The process repeats
    until all features that appear in any tree split are ranked.

    Returns:
        ranking: feature names in selection order.
        ranking_values: maps each feature to its metric values at the time it was selected.

    @param model: A fitted decision-tree ensemble.
    @param consumer_data: The data to explain.
    @param metric: The metric to compute. Defaults to BanzhafValues().
    @param verbose: Whether to show tqdm progress bars.
    """
    if metric is None:
        metric = BanzhafValues()

    n = len(consumer_data)
    model_obj = load_decision_tree_ensemble_model(model, list(consumer_data.columns))

    remaining = _get_tree_features(model_obj)

    result = path_dependent_under_always_participating_features(
        model_obj, consumer_data, metric, always_participating_features=[],
        model_was_loaded=True, verbose=verbose,
    )

    always_participating: List[str] = []
    ranking: List[str] = []
    ranking_values: Dict[str, float] = {}

    scores = _mean_abs(result, remaining, n)

    with tqdm(total=len(remaining), desc="Selecting features", disable=not verbose) as pbar:
        while remaining:
            top_f = max(remaining, key=lambda f: scores[f])
            ranking.append(top_f)
            ranking_values[top_f] = scores[top_f]
            remaining.remove(top_f)

            if not remaining:
                pbar.update(1)
                break

            prev_always_participating = list(always_participating)
            always_participating.append(top_f)

            result = always_participating_delta_update(
                model_obj, consumer_data,
                prev_always_participating, always_participating,
                [top_f], result, metric, model_was_loaded=True,
            )
            result.pop(top_f, None)

            scores = _mean_abs(result, remaining, n)
            pbar.update(1)

    ranked_set = set(ranking)
    for f in consumer_data.columns:
        if f not in ranked_set:
            ranking.append(f)
            ranking_values[f] = 0.0

    return ranking, ranking_values
