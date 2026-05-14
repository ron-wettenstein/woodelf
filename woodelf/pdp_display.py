"""
WoodelfPartialDependenceDisplay
-------------------------------
Drop-in replacement for ``sklearn.inspection.PartialDependenceDisplay``,
powered by WOODELF's fast PDP computation.

The class holds computed PDP data and creates a ``PartialDependenceDisplay``
on demand when any plot method is called.  The sklearn import is lazy (inside
the plot helpers) so that woodelf does not depend on sklearn at the module level.

Typical usage
-------------
    display = WoodelfPartialDependenceDisplay.from_estimator(
        model, X_train,
        compute_pdp=True,
        compute_joint_pdp=True,
        grid_resolution=100,
    )

    display.plot(centered=True)                      # all computed PDPs
    display.plot_pdp(["f1"], centered=False)   # specific 1-way
    display.plot_joint_pdp([("f1", "f2")],           # specific 2-way
                           contour_kw={"cmap": "coolwarm"})
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from woodelf.pdp import woodelf_pdp, woodelf_pdp_joint


class WoodelfPartialDependenceDisplay:
    """
    Partial Dependence Plot display powered by WOODELF.

    The primary entry point is :meth:`from_estimator`, which computes PDPs
    without plotting.  Call :meth:`plot`, :meth:`plot_pdp`, or
    :meth:`plot_joint_pdp` to render.

    Parameters
    ----------
    pdvs : dict mapping str → sklearn Bunch
        PDP results keyed by feature name.
        Each Bunch has ``average`` of shape ``(1, n_grid)`` and ``grid_values=[x_arr]``.
    joint_pdvs : dict mapping (str, str) → sklearn Bunch
        Joint PDP results keyed by ``(f1, f2)`` pair.
        Each Bunch has ``average`` of shape ``(1, n_f1, n_f2)`` and ``grid_values=[f1_arr, f2_arr]``.
    feature_names : list of str
        All feature names in the dataset.
    name_to_idx : dict mapping str → int
        Feature name to integer column index.
    deciles : dict mapping int → 1-D array
        Per-feature decile arrays keyed by integer feature index.
    subsample, random_state :
        Passed through to ``PartialDependenceDisplay`` for rug-mark rendering.

    Attributes
    ----------
    _display : sklearn.inspection.PartialDependenceDisplay or None
        Set after the first plot call; ``None`` before.
    figure_, bounding_ax_, axes_, lines_, contours_,
    deciles_vlines_, deciles_hlines_
        Populated on the underlying ``_display`` after a plot call;
        accessible directly on this object via ``__getattr__``.
    """

    def __init__(
        self,
        pdvs: Dict[str, object],
        joint_pdvs: Dict[Tuple[str, str], object],
        feature_names: List[str],
        name_to_idx: Dict[str, int],
        deciles: Dict[int, np.ndarray],
        subsample: int = 1000,
        random_state=None,
    ):
        self._pdvs = pdvs
        self._joint_pdvs = joint_pdvs
        self._feature_names = feature_names
        self._name_to_idx = name_to_idx
        self._deciles = deciles
        self._subsample = subsample
        self._random_state = random_state
        self._display = None

    # ------------------------------------------------------------------
    # Attribute delegation — sklearn plot attributes accessible directly.
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        if name == "_display":
            raise AttributeError(name)
        display = self.__dict__.get("_display")
        if display is not None:
            return getattr(display, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}. "
            "Call plot(), plot_pdp(), or plot_joint_pdp() first."
        )

    # ------------------------------------------------------------------
    # from_estimator — compute PDPs with WOODELF, return display object.
    # ------------------------------------------------------------------

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        compute_pdp: bool = True,
        compute_joint_pdp: bool = False,
        feature_names=None,
        grid_resolution: int = 100,
        percentiles=(0.05, 0.95),
        method: str = "brute",
        verbose: int = 0,
        subsample: int = 1000,
        random_state=None,
        # WOODELF-specific
        GPU: bool = False,
        sampled: bool = False,
        full_pdp: bool = False,
        use_woodelfhd: Optional[bool] = None,
    ):
        """
        Compute partial dependence plots with WOODELF.

        Returns a :class:`WoodelfPartialDependenceDisplay` with the computed
        data stored.  Call :meth:`plot`, :meth:`plot_pdp`, or
        :meth:`plot_joint_pdp` to render.

        Parameters
        ----------
        estimator : fitted tree-ensemble
            XGBoost, LightGBM, or sklearn tree ensemble supported by WOODELF.
        X : DataFrame or array-like of shape (n_samples, n_features)
            Background dataset for computing PDPs.
        compute_pdp : bool, default True
            Compute 1-way PDPs for all features in ``X``.
        compute_joint_pdp : bool, default False
            Compute 2-way joint PDPs for all unordered feature pairs in ``X``.

        GPU : bool, default False
        sampled : bool, default False
            Sample grid values from ``X`` instead of using an equally spaced grid.
        full_pdp : bool, default False
            If True, use every threshold value present in the model as a grid
            point instead of an equally spaced grid.  This produces a complete
            view of the average prediction at every possible decision boundary.
            When True, ``grid_resolution`` is ignored; the grid size is
            determined per feature by its number of distinct threshold values
            in the tree ensemble (and therefore differs across features).

        feature_names : list of str, optional
            Required when ``X`` is a numpy array.
        grid_resolution : int, default 100
            Number of equally spaced grid points per feature.
        percentiles : tuple of float, default (0.05, 0.95)
            Percentile bounds for the grid range.
        method : {'recursion', 'brute'}
            ``'recursion'`` → ``accurate=False`` (path-dependent, faster).
            ``'brute'``     → ``accurate=True``  (interventional, exact).
        verbose : int, default 0
            Verbosity passed to WOODELF's joint PDP computation.
        subsample, random_state :
            Passed through to the display; ``random_state`` also seeds the sampler.
        use_woodelfhd : bool or None
            Force / disable WoodelfHD. ``None`` → automatic.

        
        Removed parameters: 
            Not supported: 
                target, categorical_features, kind, sample_weight, n_jobs, custom_values, response_method, ice_lines_kw.
            Pass them in the plot, plot_pdp, or plot_joint_pdp call if needed:
                line_kw, pd_line_kw, contour_kw, ax, centered.

        Returns
        -------
        WoodelfPartialDependenceDisplay
        """
        from sklearn.utils import Bunch

        accurate = method != "recursion"  # True for 'brute'; False for 'recursion'

        # ---- Normalise X to DataFrame ----------------------------------------
        if not isinstance(X, pd.DataFrame):
            if feature_names is not None:
                X = pd.DataFrame(X, columns=feature_names)
            else:
                raise ValueError(
                    "X must be a pandas DataFrame. Provide feature_names when "
                    "X is a numpy array."
                )
        if feature_names is None:
            feature_names = list(X.columns)
        all_feature_names = list(X.columns)
        name_to_idx = {name: idx for idx, name in enumerate(all_feature_names)}

        if not compute_pdp and not compute_joint_pdp:
            raise ValueError("At least one of compute_pdp or compute_joint_pdp must be True.")

        seed = random_state if random_state is not None else 42

        # ---- Compute 1-way PDPs via woodelf_pdp ------------------------------
        one_way_pdvs: Dict[str, np.ndarray] = {}
        one_way_grid: Dict[str, np.ndarray] = {}
        if compute_pdp:
            one_way_pdvs, grid_df = woodelf_pdp(
                estimator, X,
                k=grid_resolution,
                GPU=GPU,
                centered=(not accurate),  # approximate → centered; exact → not centered
                percentiles=tuple(percentiles),
                sampled=sampled,
                seed=seed,
                full_pdp=full_pdp,
                use_woodelfhd=use_woodelfhd,
                accurate=accurate,
            )
            one_way_grid = {
                f: np.trim_zeros(grid_df[f].values, trim="b")
                for f in all_feature_names
                if f in grid_df.columns
            }

        # ---- Compute 2-way (joint) PDPs via woodelf_pdp_joint ----------------
        two_way_pdvs: Dict[Tuple[str, str], np.ndarray] = {}
        two_way_f1_pts: Dict[Tuple[str, str], np.ndarray] = {}
        two_way_f2_pts: Dict[Tuple[str, str], np.ndarray] = {}
        if compute_joint_pdp:
            two_way_pdvs, two_way_f1_pts, two_way_f2_pts = woodelf_pdp_joint(
                estimator, X,
                k=grid_resolution,
                GPU=GPU,
                percentiles=tuple(percentiles),
                sampled=sampled,
                seed=seed,
                full_pdp=full_pdp,
                verbose=(verbose > 0),
                use_woodelfhd=use_woodelfhd,
                accurate=accurate,
            )

        # ---- Build sklearn-compatible Bunch dicts ----------------------------
        #
        # sklearn PartialDependenceDisplay expects:
        #   1-way: Bunch(average=shape(1, n_grid),     grid_values=[x_arr])
        #   2-way: Bunch(average=shape(1, n_f1, n_f2), grid_values=[f1_arr, f2_arr])
        #          where average[0][i, j] = PDP at (f1_arr[i], f2_arr[j])

        sk_deciles = {
            name_to_idx[f]: np.percentile(X[f].dropna().values, np.arange(10, 100, 10))
            for f in all_feature_names
        }

        # --- 1-way ---
        pdvs: Dict[str, object] = {}
        for feat in (all_feature_names if compute_pdp else []):
            grid = one_way_grid.get(
                feat,
                np.linspace(float(X[feat].min()), float(X[feat].max()), grid_resolution),
            )
            n = len(grid)
            avg = one_way_pdvs.get(feat, np.zeros(n, dtype=np.float32))
            pdvs[feat] = Bunch(
                average=avg[:n][np.newaxis, :].astype(np.float64),
                grid_values=[grid],
            )

        # --- 2-way ---
        pairs = [
            (all_feature_names[i], all_feature_names[j])
            for i in range(len(all_feature_names))
            for j in range(i + 1, len(all_feature_names))
        ] if compute_joint_pdp else []

        joint_pdvs: Dict[Tuple[str, str], object] = {}
        for f1, f2 in pairs:
            joint_pdvs[(f1, f2)] = cls._build_joint_bunch(
                two_way_pdvs[(f1, f2)],
                two_way_f1_pts[(f1, f2)],
                two_way_f2_pts[(f1, f2)],
                full_pdp,
            )

        return cls(
            pdvs=pdvs,
            joint_pdvs=joint_pdvs,
            feature_names=all_feature_names,
            name_to_idx=name_to_idx,
            deciles=sk_deciles,
            subsample=subsample,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # plot helpers — each builds a PartialDependenceDisplay and renders.
    # ------------------------------------------------------------------

    def _plot_subset(
        self,
        pd_results: list,
        sk_features: list,
        ax=None,
        n_cols: int = 3,
        centered: bool = False,
        line_kw=None,
        pd_line_kw=None,
        contour_kw=None,
    ):
        from sklearn.inspection import PartialDependenceDisplay
        self._display = PartialDependenceDisplay(
            pd_results=pd_results,
            features=sk_features,
            feature_names=self._feature_names,
            target_idx=0,
            deciles=self._deciles,
            kind="average",
            subsample=self._subsample,
            random_state=self._random_state,
        )
        plot_kwargs = dict(ax=ax, n_cols=n_cols, centered=centered)
        if line_kw is not None:
            plot_kwargs["line_kw"] = line_kw
        if pd_line_kw is not None:
            plot_kwargs["pd_line_kw"] = pd_line_kw
        if contour_kw is not None:
            plot_kwargs["contour_kw"] = contour_kw
        self._display.plot(**plot_kwargs)
        return self

    def plot(
        self,
        ax=None,
        n_cols: int = 3,
        centered: bool = False,
        line_kw=None,
        pd_line_kw=None,
        contour_kw=None,
    ):
        """
        Render all computed PDPs and joint PDPs.
        """
        pd_results = list(self._pdvs.values()) + list(self._joint_pdvs.values())
        sk_features = (
            [(self._name_to_idx[f],) for f in self._pdvs]
            + [(self._name_to_idx[f1], self._name_to_idx[f2]) for f1, f2 in self._joint_pdvs]
        )
        return self._plot_subset(pd_results, sk_features, ax, n_cols, centered, line_kw, pd_line_kw, contour_kw)

    def plot_pdp(
        self,
        features: List[str],
        ax=None,
        n_cols: int = 3,
        centered: bool = False,
        line_kw=None,
        pd_line_kw=None,
        contour_kw=None,
    ):
        """
        Render PDPs for the given features.

        Parameters
        ----------
        features : list of str
            Feature names to plot.  Must be a subset of the features for which
            PDPs were computed in :meth:`from_estimator`.
        """
        pd_results = [self._pdvs[f] for f in features]
        sk_features = [(self._name_to_idx[f],) for f in features]
        return self._plot_subset(pd_results, sk_features, ax, n_cols, centered, line_kw, pd_line_kw, contour_kw)

    def plot_joint_pdp(
        self,
        feature_pairs: List[Tuple[str, str]],
        ax=None,
        n_cols: int = 3,
        centered: bool = False,
        line_kw=None,
        pd_line_kw=None,
        contour_kw=None,
    ):
        """
        Render joint PDPs for the given feature pairs.

        Parameters
        ----------
        feature_pairs : list of (str, str)
            Feature pairs to plot.  Each pair must appear in the same order
            (f1 < f2 by column position) as computed in :meth:`from_estimator`.
        """
        pd_results = [self._joint_pdvs[pair] for pair in feature_pairs]
        sk_features = [(self._name_to_idx[f1], self._name_to_idx[f2]) for f1, f2 in feature_pairs]
        return self._plot_subset(pd_results, sk_features, ax, n_cols, centered, line_kw, pd_line_kw, contour_kw)

    # ------------------------------------------------------------------
    # Private: 2-way grid building.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_joint_bunch(raw, f1_pts, f2_pts, full_pdp):
        """Build a sklearn Bunch for one (f1, f2) joint PDP pair.

        Parameters
        ----------
        raw : ndarray, shape (k²,)
            Flat joint PDP values from woodelf_pdp_joint.
        f1_pts, f2_pts : ndarray, shape (k²,)
            Joint-grid coordinates for each feature.
        full_pdp : bool
            If True, strip trailing zero-padding from each axis.

        Returns
        -------
        sklearn Bunch with average of shape (1, k_f1, k_f2) and grid_values [x, y].
        """
        from sklearn.utils import Bunch
        k = int(round(math.sqrt(len(raw))))
        # build_points_for_full_pdp zero-pads shorter features; strip trailing zeros when needed.
        x_grid = np.trim_zeros(f1_pts[:k], 'b') if full_pdp else f1_pts[:k]
        y_grid = np.trim_zeros(f2_pts[::k], 'b') if full_pdp else f2_pts[::k]
        return Bunch(
            # reshape flat k² → (k,k) C-order (rows=f2, cols=f1); slice strips zero-padding;
            # .T swaps to (k_f1, k_f2); [None] adds the output dimension sklearn expects.
            average=raw.reshape(k, k)[:len(y_grid), :len(x_grid)].T[None].astype(np.float64),
            grid_values=[x_grid, y_grid],
        )
