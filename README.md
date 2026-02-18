<p align="center">
  <img src="https://raw.githubusercontent.com/ron-wettenstein/woodelf/main/docs/WOODELF_commercial.png" width="1000" />
</p>

### Understand trees. Decision trees.

WOODELF is a unified and efficient algorithm for computing Shapley values on decision trees. It supports:

- **CPU and GPU**
- **Path-Dependent and Background (Interventional) explanations**
- **Shapley and Banzhaf values**
- **First-order values and interaction values**

All within a single algorithmic framework. The implementation is written in Python, with all performance-critical operations vectorized and optimized using NumPy and SciPy. GPU acceleration is supported via CuPy.

Our approach is significantly faster than the one used by the `shap` package. 
In particular, the complexity of Interventional SHAP is dramatically reduced: 
when explaining $n$ samples with a background dataset of size $m$, 
the `shap` package requires $O(nm)$ time, while WOODELF requires only $O(n + m)$ time. 
We also substantially accelerate Path-Dependent SHAP, especially on large datasets and when computing interaction values.

To demonstrate the speed-up, we computed Shapley values and interaction values for 3,000,000 samples with 127 features 
using a background dataset of 5,000,000 rows. We explained the predictions of an XGBoost model with 100 trees of 
depth 6, trained on this background data. The results are reported in the table below.

| Task                             | shap package CPU | WOODELF CPU | WOODELF GPU |
|----------------------------------|------------------|-------------|-------------|
| Path Dependent SHAP              | 51 min           | 96 seconds  | 3.3 seconds |
| Background SHAP | 8 year*          | 162 seconds | 16 seconds  |
| Path Dependent SHAP interactions | 8 days*          | 193 seconds | 6 seconds   |
| Background SHAP interactions | Not implemented  | 262 seconds | 19 seconds  |

Values marked with * are runtime estimates. 
WOODELF also outperforms other state-of-the-art approaches. In this task, it is an order of magnitude faster than PLTreeSHAP and approximately 4× faster than FastTreeSHAP.


## Installations

A simple pip install:
<pre>
pip install woodelf_explainer
</pre>

The required dependencies are `pandas`, `numpy`, `scipy`, and `shap`. 
The `shap` package is used for parsing decision trees and for minor auxiliary operations, 
while the Shapley value computation is handled entirely by WOODELF.

An optional dependency is `cupy`, which enables GPU-accelerated execution.

## Usage

Use the `woodelf.explainer.WoodelfExplainer` object!

Its API is identical to the API of `shap.TreeExplainer`. The functions inputs are the same, the output is the same, just the algorithm is different.

Background (interventional) SHAP and Shapley interaction values computation:
```python
from woodelf import WoodelfExplainer

# Get X_train, y_train, X_test, y_test and train an XGBoost model

explainer = WoodelfExplainer(xgb_model, X_train)
background_values = explainer.shap_values(X_test)
background_iv = explainer.shap_interaction_values(X_test) 
```
**Good to know:** 
- The shap python package does not support Background SHAP interactions - WOODELF supports them!
- We extensively validate WOODELF against the `shap` package and confirm that both return identical Shapley values. 
Replacing `shap.TreeExplainer` with `WoodelfExplainer` will not change the output: both the format and the numerical values 
remain the same (up to minor floating-point differences - we test with a tolerance of 0.00001).

For visualization, pass the returned output to any of the shap plots:
```python
import shap

shap.summary_plot(background_values, X_test)
```


## Additional Usage Examples

#### Path Dependent SHAP

Same API as in the shap package:
```python
explainer = WoodelfExplainer(xgb_model)
pd_values = explainer.shap_values(X_test)
pd_iv = explainer.shap_interaction_values(X_test)
```

#### Using GPU
Simply pass `GPU=True` in the WoodelfExplainer initialization:
```python
explainer = WoodelfExplainer(xgb_model, X_train, GPU=True)
background_values_GPU = explainer.shap_values(X_test)
```

#### Banzhaf Values
Simply use `explainer.banzhaf_values` for banzhaf values and `explainer.banzhaf_interaction_values` for banzhaf interaction values.
```python
explainer = WoodelfExplainer(xgb_model, X_train)
banzhaf_values = explainer.banzhaf_values(X_test)
banzhaf_iv = explainer.banzhaf_interaction_values(X_test)
```

#### Better Output Format

A memory-efficient output format that returns results as a DataFrame with feature pairs as columns. Feature pairs with zero interactions are automatically excluded to save RAM.

```python
explainer = WoodelfExplainer(xgb_model, X_train)
background_iv_df = explainer.shap_interaction_values(X_test, as_df=True, exclude_zero_contribution_features=False)
```

#### Built-in Cache

By default, caching is enabled for sufficiently small decision-tree ensembles (e.g., models with low tree depth).

The cache stores the preprocessed background data, eliminating the need to recompute it across repeated uses of the same explainer instance.

You can control this behavior manually:
- Enable caching with `cache_option='yes'`
- Disable caching with `cache_option='no'`

**Note:** Caching may be memory-intensive for deep trees (e.g., `max_depth ≥ 8`).
```python
explainer = WoodelfExplainer(xgb_model, X_train, cache_option='yes')
shap_sample_1 = explainer.shap_values(X_test.sample(100))
# No need to preprocess the background data from here on, the cache will be used instead.
shap_sample_2 = explainer.shap_values(X_test.sample(100)) 
shap_sample_3 = explainer.shap_values(X_test.sample(100))
...
```



## Citations

To cite our package and algorithm, please refer to our AAAI 2026 paper. The paper was accepted to AAAI 2026 and will be published soon.
For now, refer to its [arXiv version](https://arxiv.org/abs/2511.09376). 

```bibtex
@misc{nadel2025decisiontreesbooleanlogic,
      title={From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm}, 
      author={Alexander Nadel and Ron Wettenstein},
      year={2025},
      eprint={2511.09376},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.09376}, 
}
```

## Contact & Collaboration

If you have questions, are considering using WOODELF in your research, or would like to contribute, feel free to reach out:

**Ron Wettenstein**  
Reichman University, Herzliya, Israel  
📧 ron.wettenstein@post.runi.ac.il

## Summary and Future Research

A strong explainability approach reveals what your models have learned, 
which features matter most for predicting your target, 
and which factors have the greatest influence on individual predictions. 
Accurate Shapley and Banzhaf values based on large background datasets are a 
meaningful step toward answering these questions.
We will continue to explore new explainability methods and expand this framework with increasingly powerful tools for interpreting 
and understanding the behavior of decision tree ensembles—so you can not only observe their predictions, but understand what drives them.

> "Trees much like this one date back 290 million years, around a thousand times longer than we've been here.
> To me, to sit beneath a Ginkgo tree and look up is to be reminded that we're a blip in the story of life.
> But also, what a blip. We are, after all, the only species that ever tried to name the Ginkgo. 
> We are not observers of life on Earth, as much as it may sometimes feel that way. We are participants in that life. And so trees don't just remind me how astonishing they are, they also remind me how astonishing we are." 
> 
> John Green, *At Least There Are Trees* 


