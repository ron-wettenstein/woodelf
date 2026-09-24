from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from woodelf.core.cube_metric import BanzhafValues, CubeMetric, ShapleyValues
from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
from woodelf.always_participating_woodelf import always_participating_delta_update, path_dependent_under_always_participating_features
from woodelf.personalized_woodelf import personalized_baseline_delta_update, personalized_baseline_woodelf
from woodelf.woodelf_sparse import woodelf_sparse


# Correlation-based baseline schemes (impute a remaining feature from its most-correlated anchor).
_CORRELATION_SCHEMES = ("pearson_correlation", "monoton_xgboost")

# XGBoost hyper-parameters for the monotone single-feature imputation (the "monoton_xgboost" schemes).
# Depth/trees/lr are deliberately small so the fit estimates E[f1|f2] (the signal) rather than memorising
# f1; the min_child_weight floor (computed per-n at fit time) keeps every monotone bin backed by enough rows.
_MONOTONE_XGB_PARAMS = dict(
    n_estimators=10,
    learning_rate=0.3,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    random_state=0,
    objective="reg:squarederror",
)


def _linreg_coeffs(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Return (const, coef) for the OLS line y = const + coef * x."""
    r = x.corr(y)
    if np.isnan(r) or x.std() == 0:
        return float(y.mean()), 0.0
    coef = r * y.std() / x.std()
    const = y.mean() - coef * x.mean()
    return float(const), float(coef)


def _monotone_xgboost_impute(x: pd.Series, y: pd.Series, n: int) -> np.ndarray:
    """
    Impute y from a single feature x with a monotone XGBoost (the "monoton_xgboost" baseline).

    The monotone direction follows the sign of Spearman's rho, so the imputation is a robust, possibly
    non-linear, monotone function of x — more robust to outliers and ordinal/non-linear relationships
    than the OLS a + b*x baseline. Degenerate cases (constant x, or zero/NaN correlation) fall back to a
    constant baseline (the mean of y), mirroring _linreg_coeffs. Rows where y is missing are left out of the fit
    (XGBoost rejects NaN labels); every row still gets a prediction.
    """
    ok = y.notna().to_numpy()
    return _monotone_xgboost_fit_predict(x[ok], y[ok], x, n)


def _monotone_xgboost_fit_predict(x: pd.Series, y: pd.Series, x_pred: pd.Series, n: int) -> np.ndarray:
    """Fit the monotone single-feature XGBoost of _monotone_xgboost_impute on (x, y) and predict at x_pred."""
    if x.std() == 0:  # constant anchor: correlation is undefined, fall back without computing it
        return np.full(len(x_pred), float(y.mean()))
    rho = x.corr(y, method="spearman")
    if np.isnan(rho) or rho == 0.0:
        return np.full(len(x_pred), float(y.mean()))
    sign = 1 if rho > 0 else -1
    model = xgb.XGBRegressor(
        monotone_constraints=(sign,),
        min_child_weight=max(20, int(0.02 * n)),
        **_MONOTONE_XGB_PARAMS,
    )
    model.fit(x.to_numpy().reshape(-1, 1), y.to_numpy())
    return model.predict(x_pred.to_numpy().reshape(-1, 1))


def _impute_baseline(target: str, anchor: str, consumer_data: pd.DataFrame, scheme: str, n: int):
    """Impute the background values B[target] from C[anchor] for a correlation-based scheme."""
    x, y = consumer_data[anchor], consumer_data[target]
    if scheme == "pearson_correlation":
        const, coef = _linreg_coeffs(x, y)
        return const + coef * x
    if scheme == "monoton_xgboost":
        return _monotone_xgboost_impute(x, y, n)
    raise ValueError(f"scheme {scheme!r} is not a correlation-based imputation scheme")


def _best_anchor(f: str, candidates: List[str], consumer_data: pd.DataFrame, corr_method: str) -> Tuple[str, float]:
    """Find the candidate with the highest |corr| (corr_method) to feature f. Returns (anchor, abs_corr)."""
    best_abs_corr = -1.0
    best_f_sel = candidates[0]
    for f_sel in candidates:
        abs_corr = abs(consumer_data[f].corr(consumer_data[f_sel], method=corr_method))
        if np.isnan(abs_corr):
            abs_corr = 0.0
        if abs_corr > best_abs_corr:
            best_abs_corr = abs_corr
            best_f_sel = f_sel
    return best_f_sel, best_abs_corr


def _mean_abs(values: Dict[str, np.ndarray], features: List[str], n: int) -> Dict[str, float]:
    return {f: float(np.mean(np.abs(values.get(f, np.zeros(n))))) for f in features}


def _init_background(
    consumer_data: pd.DataFrame,
    remaining: List[str],
    baseline_init_scheme: str,
    anchor_info: Dict[str, Tuple[str, float]],
    n: int,
    min_corr: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build the initial background B for remaining features.

    Selected features: B[f] = C[f] (copy of consumer, already set).
    Remaining features depend on the scheme:
      - pearson_correlation: B[f] = const + coef * C[best_anchor] via OLS.
      - monoton_xgboost:     B[f] = monotone XGBoost prediction from C[best_anchor].
      - median: B[f] = median(C[f]) — same constant for all rows.
      - mean:   B[f] = mean(C[f])  — same constant for all rows.

    For the correlation-based schemes, a feature whose best-anchor |corr| < min_corr falls back to the
    mean baseline (too weak an anchor to impute from reliably).
    """
    B = consumer_data.copy()
    if baseline_init_scheme in _CORRELATION_SCHEMES:
        for f in remaining:
            anchor, abs_corr = anchor_info[f]
            if min_corr is not None and abs_corr < min_corr:
                B[f] = consumer_data[f].mean()
            else:
                B[f] = _impute_baseline(f, anchor, consumer_data, baseline_init_scheme, n)
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
    min_corr: Optional[float] = None,
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
        - "monoton_xgboost": B[f] = monotone XGBoost prediction from C[best_anchor]. The anchor is chosen
          by Spearman correlation and the model is constrained monotone in the Spearman-sign direction.
          More robust to outliers and ordinal/non-linear relationships than the OLS line.
        - "median": B[f] = median(C[f]) — same constant for all rows.
        - "mean":   B[f] = mean(C[f])  — same constant for all rows.
        The "mean"/"median" inits currently require baseline_updating_scheme="do_nothing".
    @param baseline_updating_scheme: How to update B each time a new feature is selected:
        - "pearson_correlation": re-anchor each remaining feature to the best correlated selected
          feature using OLS (default).
        - "monoton_xgboost": re-anchor (by Spearman correlation) using the monotone XGBoost imputation.
        - "do_nothing": only neutralize B[top_f] = C[top_f]; leave all other baselines unchanged.
        - "median_correlation": not yet implemented.

    Correlation method: Spearman is used whenever either scheme is "monoton_xgboost", otherwise Pearson.

    @param min_corr: Minimum |correlation| with the best anchor required to impute a feature from it.
        When provided, any remaining feature whose best-anchor |corr| < min_corr is given a constant
        mean baseline (B[f] = mean(C[f])) instead of a correlation-based imputation — a weak anchor
        would fit noise, so the mean is the safer baseline. None (default) disables the threshold and
        every feature is imputed from its best anchor. Only affects the correlation-based schemes
        ("pearson_correlation", "monoton_xgboost"); a no-op for the "mean"/"median"/"do_nothing" schemes.
    """
    if min_corr is not None and not 0.0 <= min_corr <= 1.0:
        raise ValueError(f"min_corr must be in [0, 1] or None, got {min_corr!r}")
    _VALID_INIT_SCHEMES = {"pearson_correlation", "monoton_xgboost", "median", "mean"}
    if baseline_init_scheme not in _VALID_INIT_SCHEMES:
        raise ValueError(f"baseline_init_scheme must be one of {_VALID_INIT_SCHEMES}, got {baseline_init_scheme!r}")

    _VALID_UPDATE_SCHEMES = {"pearson_correlation", "monoton_xgboost", "do_nothing", "median_correlation"}
    if baseline_updating_scheme not in _VALID_UPDATE_SCHEMES:
        raise ValueError(f"baseline_updating_scheme must be one of {_VALID_UPDATE_SCHEMES}, got {baseline_updating_scheme!r}")
    if baseline_updating_scheme == "median_correlation":
        raise NotImplementedError("baseline_updating_scheme='median_correlation' is not yet implemented")
    # mean/median init records no anchor correlations, so a correlation-based updating scheme would never
    # re-anchor features that are correlated with the initial selection (they would stay on the constant
    # baseline). Until that is handled, only the "do_nothing" updating scheme is supported with these inits.
    if baseline_init_scheme in {"mean", "median"} and baseline_updating_scheme != "do_nothing":
        raise ValueError(
            f"baseline_init_scheme={baseline_init_scheme!r} only supports baseline_updating_scheme='do_nothing', "
            f"got baseline_updating_scheme={baseline_updating_scheme!r}"
        )

    corr_method = "spearman" if "monoton_xgboost" in (baseline_init_scheme, baseline_updating_scheme) else "pearson"

    if metric is None:
        metric = BanzhafValues()

    all_features = list(consumer_data.columns)
    n = len(consumer_data)

    model_obj = load_decision_tree_ensemble_model(model, all_features)

    # --- Determine initial selection ---
    if initial_selection is not None and len(initial_selection) == 0:
        initial_selection = None  # an empty list means "no anchors provided" -> auto-select below
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
    anchor_info: Dict[str, Tuple[str, float]] = {}
    if baseline_init_scheme in _CORRELATION_SCHEMES or baseline_updating_scheme in _CORRELATION_SCHEMES:
        for f in remaining:
            anchor_info[f] = _best_anchor(f, selected, consumer_data, corr_method)

    B = _init_background(consumer_data, remaining, baseline_init_scheme, anchor_info, n, min_corr)

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
            new_anchor_data: Dict[str, Tuple[str, float]] = {}
            if baseline_updating_scheme in _CORRELATION_SCHEMES:
                for f in remaining:
                    _, curr_best_corr = anchor_info[f]
                    abs_corr_with_top_f = abs(consumer_data[f].corr(consumer_data[top_f], method=corr_method))
                    if np.isnan(abs_corr_with_top_f):
                        abs_corr_with_top_f = 0.0
                    if abs_corr_with_top_f > curr_best_corr:
                        new_anchor_data[f] = (top_f, abs_corr_with_top_f)
                        if min_corr is None or abs_corr_with_top_f >= min_corr:
                            changed_features.append(f)

            # top_f included because B[top_f] is also changing (neutralized to consumer)
            features_subset_delta = [top_f] + changed_features

            # Build new B (impute the features whose baseline changed and neutralize top_f)
            new_B = B.copy()
            for f in changed_features:
                new_B[f] = _impute_baseline(f, top_f, consumer_data, baseline_updating_scheme, n)
            new_B[top_f] = consumer_data[top_f]

            result = personalized_baseline_delta_update(
                model_obj, consumer_data, B, new_B, features_subset_delta, result, metric,
                GPU=GPU, model_was_loaded=True,
            )

            for f, anchor_data in new_anchor_data.items():
                anchor_info[f] = anchor_data
            B = new_B
            selected.append(top_f)

            scores = _mean_abs(result, remaining, n)
            top_f = max(remaining, key=scores.get)

    for f in reversed(zero_contrib_features):
        ranking.append(f)
        ranking_values[f] = np.zeros(n)

    return ranking, ranking_values


def _abs_corr_matrix(consumer_data: pd.DataFrame, method: str) -> np.ndarray:
    """|corr| between every pair of columns (Spearman = Pearson on ranks). NaN pairs -> 0."""
    ranked = consumer_data.rank() if method == "spearman" else consumer_data.astype(float)
    if ranked.isna().to_numpy().any():
        c = ranked.corr().to_numpy()  # pairwise-complete
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.atleast_2d(np.corrcoef(ranked.to_numpy(), rowvar=False))
    return np.abs(np.nan_to_num(c, nan=0.0))


def _conditional_mean(target: str, anchor: str, consumer_data: pd.DataFrame, scheme: str, n: int) -> np.ndarray:
    """E[target | anchor] under a correlation-based scheme; NaN targets are left out of the fit."""
    x, y = consumer_data[anchor], consumer_data[target]
    if scheme == "monoton_xgboost":
        ok = y.notna().to_numpy()
        return np.asarray(_monotone_xgboost_fit_predict(x[ok], y[ok], x, n), dtype=float)
    if scheme == "pearson_correlation":
        const, coef = _linreg_coeffs(x, y)
        m = (const + coef * x).to_numpy(dtype=float)
        return np.where(np.isnan(m), float(y.mean()), m)
    raise ValueError(f"scheme {scheme!r} is not a correlation-based imputation scheme")


def iterative_assignment_ranking(
    model,
    consumer_data: pd.DataFrame,
    metric: Optional[CubeMetric] = None,
    GPU: bool = False,
    imputation_scheme: str = "monoton_xgboost",
    n_background_samples: int = 8,
    anchored_noise: str = "quantiles",
    residual_sampling: str = "global",
    residual_bin_size: Optional[int] = None,
    lookahead: int = 5,
    lookahead_until: Optional[int] = 20,
    lookahead_mode: str = "efficiency",
    min_corr: Optional[float] = 0.1,
    max_selected: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Greedy "iterative assignment" feature ranking with conditional personalized baselines.

    State: a selected set S. Every selected feature is always present (background = consumer value). Every
    remaining feature h is "missing" given what S reveals about it. With a(h) its best-correlated selected
    anchor and the monotone model x_h = m(x_a) + eps, the principled value of a coalition is the conditional
    expectation  E[F(..., X_h, ...) | x_a] = integral of F(..., m(x_a) + e, ...) dP_eps(e).  Trees are step
    functions, so plugging in the single value m(x_a) (the old baseline) is badly biased: what a split at t
    needs is P(m(x_a) + eps >= t), which depends on the whole noise distribution P_eps.
    The J = n_background_samples draws of B[h] for row i represent that integral:
        anchored_noise="quantiles" (default): B_j[h] = m(x_a,i) + q_eps((j - 1/2) / J), the J quantiles of the
            model's noise distribution (the residuals x_h - m(x_a)); each (feature, row) gets them in a random
            order so that different missing features are not coupled. A deterministic quadrature of the
            integral that keeps the true noise shape (skew, zero-inflation), unlike a Gaussian with the same std.
        anchored_noise="residuals": B_j[h] = m(x_a,i) + r_k with k a random row (Monte-Carlo version of the same
            integral: Duan's smearing estimator / the residual bootstrap), restricted to rows in the same
            anchor-quantile bin for residual_sampling="local".
    A feature with no anchor yet takes the values of random other rows (plain background Shapley).
    A feature's score is mean_i |mean_j phi_ij|, computed exactly with personalized-baseline WOODELF and kept
    up to date with exact delta updates.

    Selection criterion (look-ahead): for each of the `lookahead` highest-scoring remaining features g, the
    move "select g" is simulated exactly: g is neutralized and every remaining feature whose anchor would
    switch to g is re-imputed from g. The candidate with the largest criterion is picked:
        lookahead_mode="efficiency" (default):
            mean_i | sum_{h in R} phibar_ih(S) - sum_{h in R \\ g} phibar_ih(S + g) |,  phibar = mean over draws.
            With Shapley values, efficiency gives sum_{h in R} phi_ijh(S) = F(x_i) - F(x_i,S, B_ij,R), so this is
            exactly mean_i | mean_j [F(x_i,S+g, B'_ij) - F(x_i,S, B_ij)] |: how far revealing g (and what it
            implies about its correlated partners) moves the explained prediction.
        lookahead_mode="total":    sum_{h in R} score_h(S) - sum_{h in R \\ g} score_h(S + g)
        lookahead_mode="partners": score_g(S) + sum_{h re-anchored to g} (score_h(S) - score_h(S + g))
    A feature that carries the information of several correlated features gets the credit of all of them
    (fixing the split-credit/dilution problem), while a copy of an already-selected feature scores ~0.

    Compared with feature_selection_ranking, this (1) starts from the empty set instead of bulk-selecting the
    top path-dependent features, (2) uses a conditional background distribution instead of the point
    E[h | anchor] (a point baseline behaves like a mean baseline, which is badly off for sparse/skewed
    features), and (3) selects by the look-ahead criterion instead of the individual score.

    @param metric: Cube metric for the per-feature values (default ShapleyValues(), from which the "efficiency"
        criterion is derived; BanzhafValues() ranks almost identically on depth <= 4 trees).
    @param imputation_scheme: "monoton_xgboost" (Spearman anchors, monotone XGBoost) or "pearson_correlation".
    @param n_background_samples: Background draws per consumer row (1 = point baseline E[h | anchor]).
        Memory/time scale with len(consumer_data) * n_background_samples.
    @param anchored_noise: How the draws of an anchored feature represent the model's noise: "quantiles" (the J
        noise quantiles) or "residuals" (J random residuals). They rank equally well in our benchmarks;
        "quantiles" is deterministic per row and less sensitive to random_state.
    @param residual_sampling: For anchored_noise="residuals": "global" (residual of a random other row) or
        "local" (of another row in the same anchor-quantile bin).
    @param residual_bin_size: Rows per anchor-quantile bin for "local" (default max(10, n // 100)).
    @param lookahead: Number of top candidates evaluated with the exact look-ahead each step (1 = plain greedy).
    @param lookahead_until: Use the look-ahead only for the first `lookahead_until` picks and plain greedy on the
        individual score afterwards (its gains are in the first picks; it costs ~`lookahead` delta updates per
        pick). None = look-ahead for every pick.
    @param lookahead_mode: "efficiency", "total" or "partners" (see above).
    @param min_corr: Minimum |corr| for a selected feature to become an anchor. Weaker relations are treated
        as independent (marginal background), which avoids fitting noise. None = no threshold.
    @param max_selected: Stop the greedy loop after this many picks and rank the rest by their current
        score (None = rank everything greedily).
    @return (ranking, selection_scores): all features in selection order, and each feature's criterion value
        at the time it was ranked (0 for features with no contribution, appended last).
    """
    if metric is None:
        metric = ShapleyValues()
    if imputation_scheme not in _CORRELATION_SCHEMES:
        raise ValueError(f"imputation_scheme must be one of {_CORRELATION_SCHEMES}, got {imputation_scheme!r}")
    if anchored_noise not in ("quantiles", "residuals"):
        raise ValueError(f"anchored_noise must be 'quantiles' or 'residuals', got {anchored_noise!r}")
    if residual_sampling not in ("local", "global"):
        raise ValueError(f"residual_sampling must be 'local' or 'global', got {residual_sampling!r}")
    if lookahead_mode not in ("efficiency", "total", "partners"):
        raise ValueError(f"lookahead_mode must be 'efficiency', 'total' or 'partners', got {lookahead_mode!r}")
    if n_background_samples < 1 or lookahead < 1:
        raise ValueError("n_background_samples and lookahead must be >= 1")
    if lookahead_until is not None and lookahead_until < 0:
        raise ValueError(f"lookahead_until must be >= 0 or None, got {lookahead_until!r}")
    if min_corr is not None and not 0.0 <= min_corr <= 1.0:
        raise ValueError(f"min_corr must be in [0, 1] or None, got {min_corr!r}")

    C = consumer_data.reset_index(drop=True)
    feats = list(C.columns)
    n = len(C)
    J = n_background_samples
    model_obj = load_decision_tree_ensemble_model(model, feats)
    corr_method = "spearman" if imputation_scheme == "monoton_xgboost" else "pearson"
    abs_corr = _abs_corr_matrix(C, corr_method)
    fidx = {f: i for i, f in enumerate(feats)}

    rng = np.random.default_rng(random_state)
    global_sources = [rng.permutation(n) for _ in range(J)]
    bin_keys = [rng.random(n) for _ in range(J)]  # shared by all anchors -> features with one anchor stay jointly drawn
    bin_size = residual_bin_size or max(10, n // 100)
    stacked = pd.concat([C] * J, ignore_index=True) if J > 1 else C

    local_sources_cache: Dict[str, List[np.ndarray]] = {}

    def row_sources(anchor: Optional[str]) -> List[np.ndarray]:
        if anchor is None or residual_sampling == "global":
            return global_sources
        if anchor not in local_sources_cache:
            rank = C[anchor].rank(method="first", na_option="bottom").to_numpy() - 1
            n_bins = max(1, n // bin_size)
            bins = np.minimum((rank * n_bins // n).astype(int), n_bins - 1)
            sources = []
            for keys in bin_keys:
                order = np.lexsort((keys, bins))  # rows grouped by bin, shuffled inside each bin
                starts = np.r_[0, np.flatnonzero(np.diff(bins[order])) + 1]
                src = np.empty(n, dtype=int)
                for s, e in zip(starts, np.r_[starts[1:], n]):
                    src[order[s:e]] = np.roll(order[s:e], 1)  # every row borrows another row of its bin
                sources.append(src)
            local_sources_cache[anchor] = sources
        return local_sources_cache[anchor]

    mean_cache: Dict[Tuple[str, Optional[str]], np.ndarray] = {}

    def background_column(f: str, anchor: Optional[str]) -> np.ndarray:
        key = (f, anchor)
        if key not in mean_cache:
            if len(mean_cache) * n > 5e7:  # bound the cache (~400MB)
                mean_cache.clear()
            if anchor is None:
                mean_cache[key] = np.full(n, float(C[f].mean()))
            else:
                mean_cache[key] = _conditional_mean(f, anchor, C, imputation_scheme, n)
        m = mean_cache[key]
        x = C[f].to_numpy(dtype=float)
        if J == 1:
            col = m.copy()
        elif anchor is not None and anchored_noise == "quantiles":
            # J quantiles of the noise distribution, in a random order per row (decouples the missing features)
            q = np.nanquantile(x - m, (np.arange(J) + 0.5) / J)
            order = np.argsort(np.random.default_rng([random_state, fidx[f], 7]).random((n, J)), axis=1)
            noise = q[order].T  # (J, n)
            col = np.concatenate([m + noise[j] for j in range(J)])
        else:
            r = np.nan_to_num(x - m, nan=0.0)
            col = np.concatenate([m + r[src] for src in row_sources(anchor)])
        missing = np.isnan(x)
        if missing.any():  # a missing consumer value cannot be revealed: keep it missing in the background too
            col[np.tile(missing, J)] = np.nan
        return col

    def row_sums(values: Dict[str, np.ndarray], fs: List[str]) -> np.ndarray:
        """Per consumer row: the sum over fs of the draw-averaged values."""
        total = np.zeros(n)
        for f in fs:
            v = values.get(f)
            if v is not None:
                total = total + v.reshape(J, n).mean(axis=0)
        return total

    def scores(values: Dict[str, np.ndarray], fs: List[str]) -> Dict[str, float]:
        out = {}
        for f in fs:
            v = values.get(f)
            out[f] = 0.0 if v is None else float(np.abs(v.reshape(J, n).mean(axis=0)).mean())
        return out

    remaining = list(feats)
    best_corr = {f: 0.0 for f in feats}
    B = stacked.copy()
    for f in feats:
        B[f] = background_column(f, None)
    values = personalized_baseline_woodelf(model_obj, stacked, B, metric, GPU=GPU, model_was_loaded=True, verbose=False)

    def propose(g: str):
        """Exact state after selecting g: neutralize g and re-anchor every feature that correlates better with g."""
        gi = fidx[g]
        new_B = B.copy()
        changed, new_corr = [], {}
        for h in remaining:
            if h == g:
                continue
            c = abs_corr[fidx[h], gi]
            if c > best_corr[h] and (min_corr is None or c >= min_corr):
                new_B[h] = background_column(h, g)
                changed.append(h)
                new_corr[h] = c
        new_B[g] = stacked[g]
        new_values = personalized_baseline_delta_update(
            model_obj, stacked, B, new_B, [g] + changed, values, metric, GPU=GPU, model_was_loaded=True,
        )
        return new_B, new_values, new_corr, changed

    ranking: List[str] = []
    selection_scores: Dict[str, float] = {}
    zero_contrib: List[str] = []
    n_greedy = len(feats) if max_selected is None else min(max_selected, len(feats))
    with tqdm(total=n_greedy, desc="Selecting features", disable=not verbose) as pbar:
        while remaining:
            sc = scores(values, remaining)
            for f in [f for f in remaining if sc[f] == 0.0]:
                zero_contrib.append(f)
                remaining.remove(f)
            if not remaining:
                break
            if max_selected is not None and len(ranking) >= max_selected:
                for f in sorted(remaining, key=sc.get, reverse=True):
                    ranking.append(f)
                    selection_scores[f] = sc[f]
                remaining = []
                break

            use_lookahead = (lookahead > 1 and len(remaining) > 1
                             and (lookahead_until is None or len(ranking) < lookahead_until))
            candidates = sorted(remaining, key=sc.get, reverse=True)[: lookahead if use_lookahead else 1]
            total_before = sum(sc.values())
            sums_before = row_sums(values, remaining) if use_lookahead and lookahead_mode == "efficiency" else None
            best = None
            for g in candidates:
                new_B, new_values, new_corr, changed = propose(g)
                if not use_lookahead:
                    crit = sc[g]
                elif lookahead_mode == "efficiency":
                    sums_after = row_sums(new_values, [h for h in remaining if h != g])
                    crit = float(np.abs(sums_before - sums_after).mean())
                elif lookahead_mode == "total":
                    crit = total_before - sum(scores(new_values, [h for h in remaining if h != g]).values())
                else:
                    after = scores(new_values, changed)
                    crit = sc[g] + sum(sc[h] - after[h] for h in changed)
                if best is None or crit > best[0]:
                    best = (crit, g, new_B, new_values, new_corr)

            crit, top, B, values, new_corr = best
            best_corr.update(new_corr)
            ranking.append(top)
            selection_scores[top] = float(crit)
            remaining.remove(top)
            pbar.update(1)

    for f in reversed(zero_contrib):
        ranking.append(f)
        selection_scores[f] = 0.0
    return ranking, selection_scores


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
