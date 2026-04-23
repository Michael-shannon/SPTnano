"""
Superwindow-level state statistics (dwell / run lengths, switches per block).

**Estimand:** each superwindow is one independent unit. Summaries average or
distribute over superwindows, unlike pooled adjacent-step Markov counts.

**Bootstrap vs Bayesian (short):**

- **Bootstrap (nonparametric):** Resample superwindows with replacement many
  times and recompute a statistic (e.g. mean switches, fraction with any change).
  Intervals reflect **sampling variability in the data** without assuming a
  parametric form for states. Good when superwindows are exchangeable-ish and
  you want percentile or BCa CIs for any functional.

- **Bayesian priors (parametric conjugate):** Put a prior on unknown
  probabilities (e.g. Beta prior for a single proportion, Dirichlet for a
  transition row), update with counts to get a **posterior**. Intervals are
  **credible intervals** (belief given prior + data). Smooths sparse rows
  (pseudo-counts) and avoids 0/100% artifacts when n is small; prior choice
  matters.

Use bootstrap for **superwindow-level** summaries; use Beta/Dirichlet **smoothing
or posterior** when you still want a **probability matrix** interpretable as
``P(next | current)`` with small per-row n.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

try:
    from scipy.stats import beta as beta_dist
except ImportError:
    beta_dist = None


def run_lengths_from_sequence(states: Sequence[Any]) -> List[Tuple[Any, int]]:
    """
    Split a state sequence into consecutive runs (state, length in windows).

    Parameters
    ----------
    states
        Ordered labels, one per window within a superwindow.

    Returns
    -------
    list of (state, run_length)
    """
    if not states:
        return []
    runs: List[Tuple[Any, int]] = []
    cur = states[0]
    ln = 1
    for s in states[1:]:
        if s == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur = s
            ln = 1
    runs.append((cur, ln))
    return runs


def count_adjacent_switches(states: Sequence[Any]) -> int:
    """Number of indices i where state[i] != state[i + 1]."""
    if len(states) < 2:
        return 0
    return sum(1 for i in range(len(states) - 1) if states[i] != states[i + 1])


def first_transition_edge(states: Sequence[Any]) -> Optional[Tuple[Any, Any]]:
    """First directed edge (from_state, to_state) where state changes; else None."""
    for i in range(len(states) - 1):
        if states[i] != states[i + 1]:
            return (states[i], states[i + 1])
    return None


def per_superwindow_metrics(
    states: Sequence[Any],
    runs: Optional[Sequence[Tuple[Any, int]]] = None,
) -> Dict[str, Any]:
    """
    Metrics for one superwindow's state sequence.

    Parameters
    ----------
    states
        Ordered state labels per window.
    runs
        Optional precomputed runs from :func:`run_lengths_from_sequence` to avoid
        duplicate work.

    Returns
    -------
    dict with:
        n_windows, n_runs, n_switches, any_change,
        first_transition (tuple or None),
        run_lengths (list of int), run_states (list),
        mean_run_length, max_run_length
    """
    states = list(states)
    n_w = len(states)
    if runs is None:
        runs = run_lengths_from_sequence(states)
    runs_list = list(runs)
    lengths = [ln for _, ln in runs_list]
    n_sw = count_adjacent_switches(states)
    return {
        "n_windows": n_w,
        "n_runs": len(runs_list),
        "n_switches": n_sw,
        "any_change": bool(n_sw > 0),
        "first_transition": first_transition_edge(states),
        "run_lengths": lengths,
        "run_states": [s for s, _ in runs_list],
        "mean_run_length": float(np.mean(lengths)) if lengths else float("nan"),
        "max_run_length": int(max(lengths)) if lengths else 0,
    }


def _to_pandas_superwindow_df(superwindow_df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
    if isinstance(superwindow_df, pl.DataFrame):
        return superwindow_df.to_pandas()
    return superwindow_df.copy()


def summarize_superwindow_sequences(
    superwindow_df: Union[pd.DataFrame, pl.DataFrame],
    state_sequence_col: str = "state_sequence",
    superwindow_id_col: str = "superwindow_id",
    return_polars: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Per-superwindow dwell and switch metrics, plus pooled run-length lists by state.

    Parameters
    ----------
    superwindow_df
        Must contain ``state_sequence_col`` (list-like per row) and
        ``superwindow_id_col``.
    state_sequence_col
        Column of lists of state labels in time order within the superwindow.
    superwindow_id_col
        Unique superwindow identifier.
    return_polars
        If True, ``per_superwindow`` is a Polars DataFrame; if False, Pandas;
        if None, match input type.

    Returns
    -------
    dict
        - ``per_superwindow``: one row per superwindow with metrics columns.
        - ``dwell_run_lengths_by_state``: mapping state -> list of run lengths
          (windows), pooling all runs across superwindows.
        - ``dwell_summary_by_state``: Pandas DataFrame: state, n_runs, mean,
          median, std run length.
        - ``aggregate``: overall means / fractions across superwindows.
    """
    if return_polars is None:
        return_polars = isinstance(superwindow_df, pl.DataFrame)

    pdf = _to_pandas_superwindow_df(superwindow_df)
    if state_sequence_col not in pdf.columns:
        raise KeyError(f"Missing {state_sequence_col!r} in superwindow dataframe.")
    if superwindow_id_col not in pdf.columns:
        raise KeyError(f"Missing {superwindow_id_col!r} in superwindow dataframe.")

    rows_out: List[Dict[str, Any]] = []
    dwell_by_state: Dict[Any, List[int]] = {}

    for _, row in pdf.iterrows():
        sid = row[superwindow_id_col]
        seq = row[state_sequence_col]
        if seq is None or (hasattr(seq, "__len__") and len(seq) == 0):
            m = {
                "n_windows": 0,
                "n_runs": 0,
                "n_switches": 0,
                "any_change": False,
                "first_transition": None,
                "run_lengths": [],
                "run_states": [],
                "mean_run_length": float("nan"),
                "max_run_length": 0,
            }
        else:
            seq = list(seq)
            runs = run_lengths_from_sequence(seq)
            for st, ln in runs:
                dwell_by_state.setdefault(st, []).append(ln)
            m = per_superwindow_metrics(seq, runs=runs)

        m[superwindow_id_col] = sid
        rows_out.append(m)

    per_sw = pd.DataFrame(rows_out)
    # first_transition as two columns for easier export
    ft = per_sw["first_transition"]
    per_sw["first_from"] = ft.map(lambda x: x[0] if x is not None else None)
    per_sw["first_to"] = ft.map(lambda x: x[1] if x is not None else None)

    dwell_rows = []
    for st, lengths in dwell_by_state.items():
        arr = np.asarray(lengths, dtype=float)
        dwell_rows.append(
            {
                "state": st,
                "n_runs": len(lengths),
                "mean_run_length": float(np.mean(arr)) if len(arr) else float("nan"),
                "median_run_length": float(np.median(arr)) if len(arr) else float("nan"),
                "std_run_length": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
        )
    dwell_summary = pd.DataFrame(dwell_rows)
    if len(dwell_summary):
        dwell_summary = dwell_summary.sort_values("state")

    n_sw = len(per_sw)
    agg = {
        "n_superwindows": n_sw,
        "mean_n_switches": float(per_sw["n_switches"].mean()) if n_sw else float("nan"),
        "mean_n_runs": float(per_sw["n_runs"].mean()) if n_sw else float("nan"),
        "frac_superwindows_with_change": float(per_sw["any_change"].mean())
        if n_sw
        else float("nan"),
        "mean_run_length_overall": float(per_sw["mean_run_length"].mean())
        if n_sw
        else float("nan"),
    }

    out: Dict[str, Any] = {
        "per_superwindow": per_sw,
        "dwell_run_lengths_by_state": dwell_by_state,
        "dwell_summary_by_state": dwell_summary,
        "aggregate": agg,
    }

    if return_polars:
        out["per_superwindow"] = pl.from_pandas(per_sw)

    return out


def summarize_superwindow_sequences_by_group(
    superwindow_df: Union[pd.DataFrame, pl.DataFrame],
    group_by_col: str,
    state_sequence_col: str = "state_sequence",
    superwindow_id_col: str = "superwindow_id",
    return_polars: Optional[bool] = None,
) -> Dict[Any, Dict[str, Any]]:
    """
    Run :func:`summarize_superwindow_sequences` separately for each level of
    ``group_by_col`` (e.g. ``mol`` or ``condition``).
    """
    if isinstance(superwindow_df, pl.DataFrame):
        pdf = superwindow_df.to_pandas()
    else:
        pdf = superwindow_df.copy()


    if group_by_col not in pdf.columns:
        raise KeyError(f"Missing {group_by_col!r} in superwindow dataframe.")

    out: Dict[Any, Dict[str, Any]] = {}
    for g, sub in pdf.groupby(group_by_col, dropna=False):
        out[g] = summarize_superwindow_sequences(
            sub,
            state_sequence_col=state_sequence_col,
            superwindow_id_col=superwindow_id_col,
            return_polars=return_polars,
        )
    return out


def concat_per_superwindow_by_group(
    by_group: Mapping[Any, Mapping[str, Any]],
    group_col: str,
) -> pd.DataFrame:
    """
    Stack ``per_superwindow`` frames from :func:`summarize_superwindow_sequences_by_group`
    with a column ``group_col`` identifying the group (e.g. molecule).
    """
    chunks: List[pd.DataFrame] = []
    for g, summ in by_group.items():
        ps = summ["per_superwindow"]
        pdf = ps.to_pandas() if isinstance(ps, pl.DataFrame) else ps.copy()
        pdf = pdf.copy()
        pdf[group_col] = g
        chunks.append(pdf)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _resample_superwindows_dataframe(
    group_df: pd.DataFrame,
    unique_id_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample superwindows (clusters) with replacement; duplicate rows if an id is drawn twice."""
    uids = group_df[unique_id_col].unique()
    if len(uids) == 0:
        return group_df.iloc[0:0].copy()
    boot_ids = rng.choice(uids, size=len(uids), replace=True)
    parts = [group_df[group_df[unique_id_col] == uid] for uid in boot_ids]
    return pd.concat(parts, ignore_index=True)


def _align_transition_matrix(
    mat: pd.DataFrame,
    rows: Sequence[Any],
    cols: Sequence[Any],
) -> pd.DataFrame:
    out = mat.reindex(index=list(rows), columns=list(cols)).fillna(0.0)
    return out.astype(float)


def _row_normalize_transition_counts(C: np.ndarray) -> np.ndarray:
    """Row-normalize nonnegative count matrix to transition probabilities."""
    row_sums = C.sum(axis=1, keepdims=True)
    P = np.zeros_like(C, dtype=float)
    np.divide(C, row_sums, out=P, where=row_sums > 0)
    return P


def _row_normalize_with_symmetric_dirichlet_prior(C: np.ndarray, alpha: float) -> np.ndarray:
    """
    Add ``alpha`` to every cell (symmetric Dirichlet prior over rows), then row-normalize.

    Stabilizes sparse rows (few outgoing transitions from a state) so a handful of
    edges does not become degenerate 0/100% probabilities. ``alpha=0`` is plain MLE.
    """
    if alpha <= 0:
        return _row_normalize_transition_counts(C)
    Cp = C.astype(float) + float(alpha)
    return _row_normalize_transition_counts(Cp)


def _transition_counts_from_state_sequence(
    states: Sequence[Any],
    state_to_idx: Mapping[Any, int],
) -> np.ndarray:
    """Directed edge counts for consecutive states (one matrix, aligned to vocab)."""
    k = len(state_to_idx)
    C = np.zeros((k, k), dtype=np.int64)
    for i in range(len(states) - 1):
        a, b = states[i], states[i + 1]
        if a is None or b is None:
            continue
        if a not in state_to_idx or b not in state_to_idx:
            continue
        C[state_to_idx[a], state_to_idx[b]] += 1
    return C


def bootstrap_transition_probabilities_by_group(
    windowed_df: Union[pd.DataFrame, pl.DataFrame],
    group_col: str = "mol",
    state_col: str = "final_population",
    unique_id_col: str = "superwindow_id",
    time_col: str = "time_window",
    state_order: Optional[Sequence[Any]] = None,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    min_windows: int = 2,
    count_smoothing_alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    Nonparametric bootstrap of row-normalized transition probabilities **per group**
    (e.g. per molecule): resample **superwindows** with replacement, then
    row-normalize pooled transition counts (same estimand as
    :func:`~SPTnano.helper_scripts.analyze_state_transitions` on the pooled sample).

    **Implementation:** transition counts are accumulated per superwindow once;
    each bootstrap replicate sums a resampled set of those matrices (fast). This
    matches resampling superwindows with replacement and does **not** re-scan every
    window row hundreds of times.

    **Sparse rows:** When a state has very few outgoing transitions (e.g. rare
    ``transport`` in one molecule), raw row-normalization is unstable. By default
    ``count_smoothing_alpha=1.0`` adds a symmetric Dirichlet prior (Laplace-style
    pseudo-count per cell) before normalizing each row, shrinking sparse rows toward
    uniform. Set to ``0.0`` for unsmoothed MLE (can look like 100% to one target
    by chance).

    Returns the same structure as ``analyze_state_transitions(..., group_by=...)``
    but each group's result adds ``transition_probabilities_ci_low`` and
    ``transition_probabilities_ci_high`` (aligned DataFrames) for
    :func:`~SPTnano.visualization.plot_transition_probabilities_stacked`.

    Parameters
    ----------
    windowed_df
        One row per **window**, with ``superwindow_id`` (or ``unique_id_col``)
        grouping windows into superwindows, and ``group_col`` (e.g. ``mol``).
    state_order
        Row/column order for probability matrices. If None, uses sorted union
        of states observed in the point-estimate runs.
    count_smoothing_alpha : float, default=1.0
        Added to **each cell** of the transition count matrix before row
        normalization (within each group, each bootstrap replicate, and the point
        estimate). ``0`` disables smoothing.

    Returns
    -------
    dict
        - ``transition_results_dict``: maps group -> result dict (drop-in for plotting).
        - ``transition_results_point``: optional access; same keys without bootstrap.
    """
    from .helper_scripts import analyze_state_transitions

    if isinstance(windowed_df, pl.DataFrame):
        df = windowed_df.to_pandas()
    else:
        df = windowed_df.copy()

    for c in (group_col, state_col, unique_id_col, time_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column {c!r} in windowed_df.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    q = (1.0 - confidence) / 2.0

    transition_results_dict: Dict[Any, Dict[str, Any]] = {}

    for g, sub in df.groupby(group_col, dropna=False):
        sub = sub.copy()
        sub = sub.sort_values([unique_id_col, time_col])
        track_sizes = sub.groupby(unique_id_col).size()
        valid_uids = sorted(
            track_sizes[track_sizes >= min_windows].index.tolist(),
            key=str,
        )
        sub_v = sub[sub[unique_id_col].isin(valid_uids)]
        n_valid = len(valid_uids)
        if n_valid == 0:
            continue

        # Point estimate + stats (single pass over window rows)
        pt = analyze_state_transitions(
            sub_v,
            state_col=state_col,
            unique_id_col=unique_id_col,
            time_col=time_col,
            group_by=None,
            normalize=True,
            min_windows=min_windows,
        )

        tp_pt = pt["transition_probabilities"]
        if state_order is not None:
            rows = cols = list(state_order)
        else:
            rows = list(tp_pt.index)
            cols = list(tp_pt.columns)

        K = len(rows)
        state_to_idx = {s: i for i, s in enumerate(rows)}

        # Per-superwindow count matrices (K×K), same vocab as point estimate
        per_sw = np.zeros((n_valid, K, K), dtype=np.int64)
        for i, uid in enumerate(valid_uids):
            sd = sub_v[sub_v[unique_id_col] == uid]
            states = sd[state_col].tolist()
            per_sw[i] = _transition_counts_from_state_sequence(states, state_to_idx)

        C_total = per_sw.sum(axis=0)
        P_point = _row_normalize_with_symmetric_dirichlet_prior(C_total, count_smoothing_alpha)
        tp_point = pd.DataFrame(P_point, index=rows, columns=cols)

        # Bootstrap: resample superwindow indices, sum counts, smooth + row-normalize
        mats = np.zeros((n_bootstrap, K, K), dtype=float)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_valid, size=n_valid, endpoint=False)
            C_b = per_sw[idx].sum(axis=0)
            mats[b, :, :] = _row_normalize_with_symmetric_dirichlet_prior(
                C_b.astype(float), count_smoothing_alpha
            )

        lo = np.quantile(mats, q, axis=0)
        hi = np.quantile(mats, 1.0 - q, axis=0)
        ci_lo = pd.DataFrame(lo, index=rows, columns=cols)
        ci_hi = pd.DataFrame(hi, index=rows, columns=cols)

        out = dict(pt)
        out["transition_probabilities"] = tp_point
        out["transition_probabilities_ci_low"] = ci_lo
        out["transition_probabilities_ci_high"] = ci_hi
        out["transition_bootstrap_n"] = int(n_bootstrap)
        out["n_superwindows"] = int(n_valid)
        out["group_name"] = str(g)
        out["count_smoothing_alpha"] = float(count_smoothing_alpha)
        out["transition_counts_pooled"] = pd.DataFrame(C_total, index=rows, columns=cols)
        transition_results_dict[g] = out

    return {
        "transition_results_dict": transition_results_dict,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "count_smoothing_alpha": float(count_smoothing_alpha),
    }


def compare_superwindow_metrics_by_group(
    superwindow_df: Union[pd.DataFrame, pl.DataFrame],
    group_col: str = "mol",
    state_sequence_col: str = "state_sequence",
    superwindow_id_col: str = "superwindow_id",
    bootstrap_columns: Sequence[str] = (
        "n_switches",
        "any_change",
        "mean_run_length",
    ),
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, Any]:
    """
    Per-group superwindow summaries plus bootstrap CIs, and a long table for plots.

    Parameters
    ----------
    superwindow_df
        Must include ``group_col`` (e.g. ``mol``), ``state_sequence``, ``superwindow_id``.
    group_col
        Categorical column to compare (e.g. ``mol``, ``condition``).

    Returns
    -------
    dict
        - ``by_group``: group -> output of :func:`summarize_superwindow_sequences`
        - ``bootstrap_by_group``: group -> output of :func:`bootstrap_superwindow_metrics`
        - ``summary_table``: one row per group with ``n_superwindows`` and
          ``{metric}_mean``, ``{metric}_ci_low``, ``{metric}_ci_high`` for each
          bootstrapped metric
        - ``per_superwindow_long``: all superwindows with ``group_col`` appended
    """
    by_group = summarize_superwindow_sequences_by_group(
        superwindow_df,
        group_by_col=group_col,
        state_sequence_col=state_sequence_col,
        superwindow_id_col=superwindow_id_col,
        return_polars=False,
    )

    bootstrap_by_group: Dict[Any, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for g, summ in by_group.items():
        n = int(summ["aggregate"]["n_superwindows"])
        per_sw = summ["per_superwindow"]
        row: Dict[str, Any] = {"group": g, "n_superwindows": n}

        if n == 0:
            bootstrap_by_group[g] = {"results": {}, "n": 0, "n_bootstrap": n_bootstrap}
            for col in bootstrap_columns:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_ci_low"] = np.nan
                row[f"{col}_ci_high"] = np.nan
            rows.append(row)
            continue

        boot = bootstrap_superwindow_metrics(
            per_sw,
            columns=bootstrap_columns,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_state=random_state,
        )
        bootstrap_by_group[g] = boot

        for col in bootstrap_columns:
            if col in boot["results"]:
                br = boot["results"][col]
                row[f"{col}_mean"] = br["point_estimate"]
                row[f"{col}_ci_low"] = br["ci_low"]
                row[f"{col}_ci_high"] = br["ci_high"]
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_ci_low"] = np.nan
                row[f"{col}_ci_high"] = np.nan
        rows.append(row)

    summary_table = pd.DataFrame(rows)
    if len(summary_table):
        summary_table = summary_table.sort_values("group", key=lambda s: s.map(str))

    per_long = concat_per_superwindow_by_group(by_group, group_col)

    return {
        "by_group": by_group,
        "bootstrap_by_group": bootstrap_by_group,
        "summary_table": summary_table,
        "per_superwindow_long": per_long,
    }


def plot_superwindow_group_comparison_bars(
    summary_table: pd.DataFrame,
    group_col: str = "group",
    metrics: Sequence[Tuple[str, str]] = (
        ("n_switches", "Mean switches / superwindow"),
        ("any_change", "Fraction with ≥1 switch"),
    ),
    group_order: Optional[Sequence[Any]] = None,
    figsize: Tuple[float, float] = (10, 4),
    colors: Optional[Sequence[str]] = None,
):
    """
    Side-by-side bar charts: one panel per metric, x = group, y = bootstrap mean with CI.

    Parameters
    ----------
    summary_table
        From :func:`compare_superwindow_metrics_by_group`.
    metrics
        List of (metric_key, y_axis_label). ``metric_key`` must match bootstrap
        column names, producing columns ``{metric_key}_mean``, etc.
    group_order
        Order of groups on the x-axis; default sorted by str(group).
    """
    import matplotlib.pyplot as plt

    if summary_table.empty:
        raise ValueError("summary_table is empty")

    df = summary_table.copy()
    if group_order is not None:
        order = list(group_order)
        df["_sort"] = df[group_col].map({g: i for i, g in enumerate(order)})
        df = df.sort_values("_sort", na_position="last").drop(columns="_sort", errors="ignore")
    else:
        df = df.sort_values(group_col, key=lambda s: s.map(str))

    groups = df[group_col].tolist()
    x = np.arange(len(groups))
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics / 2, figsize[1]))
    if n_metrics == 1:
        axes = [axes]

    cmap = colors or ["steelblue", "coral", "seagreen", "mediumpurple"][:n_metrics]

    for ax, (mkey, ylab), color in zip(axes, metrics, cmap):
        mean_c = f"{mkey}_mean"
        lo_c = f"{mkey}_ci_low"
        hi_c = f"{mkey}_ci_high"
        if mean_c not in df.columns:
            ax.set_visible(False)
            continue
        means = df[mean_c].to_numpy(dtype=float)
        lo = df[lo_c].to_numpy(dtype=float)
        hi = df[hi_c].to_numpy(dtype=float)
        err_lo = means - lo
        err_hi = hi - means
        ax.bar(x, means, yerr=[err_lo, err_hi], capsize=3, color=color, alpha=0.88, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups], rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(mkey.replace("_", " "))
        for i, nsw in enumerate(df["n_superwindows"]):
            ax.text(
                x[i],
                0.02,
                f"n={int(nsw)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=7,
                color="gray",
            )

    fig.suptitle("Superwindow metrics by group (bootstrap 95% CI)", y=1.02, fontsize=11)
    fig.tight_layout()
    return fig, axes


def plot_superwindow_group_violin(
    per_superwindow_long: pd.DataFrame,
    group_col: str = "mol",
    value_col: str = "n_switches",
    group_order: Optional[Sequence[Any]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 4),
):
    """
    Compare distributions of a per-superwindow quantity across groups (e.g. ``n_switches``).

    Uses seaborn if available, else matplotlib boxplot.
    """
    import matplotlib.pyplot as plt

    if per_superwindow_long.empty or value_col not in per_superwindow_long.columns:
        raise ValueError("Need non-empty per_superwindow_long with value_col")

    df = per_superwindow_long.dropna(subset=[group_col, value_col])
    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns

        sns.violinplot(
            data=df,
            x=group_col,
            y=value_col,
            order=group_order,
            ax=ax,
            inner="box",
            cut=0,
        )
    except ImportError:
        labs = list(group_order) if group_order is not None else sorted(df[group_col].unique(), key=str)
        data = [df.loc[df[group_col] == g, value_col].dropna().values for g in labs]
        ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([str(x) for x in labs], rotation=30, ha="right")

    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} by {group_col}")
    fig.tight_layout()
    return fig, ax


def beta_binomial_proportion_ci(
    n_success: int,
    n_trials: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    mass: float = 0.95,
) -> Tuple[float, float]:
    """
    Bayesian credible interval for a binomial proportion with Beta prior.

    Posterior is Beta(alpha_prior + k, beta_prior + n - k) for k successes
    in n trials. With alpha_prior=beta_prior=1 this is the uniform prior.

    Returns
    -------
    (low, high) quantiles of the posterior for ``mass`` probability.
    """
    if n_trials < 0 or n_success < 0 or n_success > n_trials:
        raise ValueError("Invalid n_success / n_trials")
    if beta_dist is None:
        raise ImportError("scipy is required for beta_binomial_proportion_ci")
    a = alpha_prior + n_success
    b = beta_prior + (n_trials - n_success)
    q = (1.0 - mass) / 2.0
    return float(beta_dist.ppf(q, a, b)), float(beta_dist.ppf(1.0 - q, a, b))


def dirichlet_smooth_transition_probs(
    count_matrix: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Row-wise Dirichlet-style smoothing: add ``alpha`` to each cell, then normalize rows.

    Use for sparse Markov transition count matrices (adjacent-step pooled).

    Parameters
    ----------
    count_matrix
        Rows = from-state, columns = to-state, nonnegative counts.
    alpha
        Pseudo-count per cell (e.g. 0.5 or 1.0). Larger = more shrinkage toward uniform.
    """
    sm = count_matrix.astype(float) + float(alpha)
    row_sums = sm.sum(axis=1)
    row_sums = row_sums.replace(0, np.nan)
    out = sm.div(row_sums, axis=0).fillna(0.0)
    return out


def bootstrap_superwindow_metrics(
    per_superwindow: Union[pd.DataFrame, pl.DataFrame],
    columns: Sequence[str] = (
        "n_switches",
        "any_change",
        "n_runs",
        "mean_run_length",
    ),
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, Any]:
    """
    Nonparametric bootstrap over superwindow rows (resample with replacement).

    For each numeric column, reports mean of bootstrap means and percentile CI.
    For ``any_change`` (bool), bootstrap mean is the fraction True.

    Parameters
    ----------
    per_superwindow
        Output ``per_superwindow`` from :func:`summarize_superwindow_sequences`.
    columns
        Which columns to summarize.
    n_bootstrap
        Number of bootstrap replicates.
    confidence
        Central interval probability (e.g. 0.95).
    random_state
        RNG seed or Generator.

    Returns
    -------
    dict
        ``results``: column -> point_estimate, bootstrap_mean, ci_low, ci_high
        ``n``: number of superwindows used
    """
    if isinstance(per_superwindow, pl.DataFrame):
        pdf = per_superwindow.to_pandas()
    else:
        pdf = per_superwindow.copy()

    n = len(pdf)
    if n == 0:
        return {"results": {}, "n": 0, "n_bootstrap": n_bootstrap}

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    q_lo = (1.0 - confidence) / 2.0
    q_hi = 1.0 - q_lo

    results: Dict[str, Dict[str, float]] = {}

    for col in columns:
        if col not in pdf.columns:
            continue
        if col == "any_change":
            arr = pdf[col].astype(bool).to_numpy()
            point = float(np.mean(arr))

            def stat(ix: np.ndarray) -> float:
                return float(np.mean(arr[ix]))

        else:
            arr = pd.to_numeric(pdf[col], errors="coerce").to_numpy(dtype=float)
            point = float(np.nanmean(arr))

            def stat(ix: np.ndarray) -> float:
                return float(np.nanmean(arr[ix]))

        stats = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            ix = rng.integers(0, n, size=n, endpoint=False)
            stats[b] = stat(ix)

        results[col] = {
            "point_estimate": point,
            "bootstrap_mean": float(np.mean(stats)),
            "ci_low": float(np.quantile(stats, q_lo)),
            "ci_high": float(np.quantile(stats, q_hi)),
        }

    return {"results": results, "n": n, "n_bootstrap": n_bootstrap}


def first_transition_counts(
    per_superwindow: Union[pd.DataFrame, pl.DataFrame],
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Tabulate first transition (from_state, to_state) among superwindows that switch.

    Rows with no change are ignored.
    """
    if isinstance(per_superwindow, pl.DataFrame):
        pdf = per_superwindow.to_pandas()
    else:
        pdf = per_superwindow.copy()

    if "first_from" not in pdf.columns or "first_to" not in pdf.columns:
        raise KeyError("Expected first_from / first_to columns (from summarize_superwindow_sequences).")

    sub = pdf.loc[pdf["any_change"]].copy()
    sub = sub.dropna(subset=["first_from", "first_to"])
    if sub.empty:
        return pd.DataFrame(columns=["from_state", "to_state", "count", "fraction"])

    grp = sub.groupby(["first_from", "first_to"], dropna=False).size().reset_index(name="count")
    total = grp["count"].sum()
    grp["fraction"] = grp["count"] / total if total else 0.0
    grp = grp.rename(columns={"first_from": "from_state", "first_to": "to_state"})
    grp = grp[grp["count"] >= min_count].sort_values("count", ascending=False)
    return grp.reset_index(drop=True)


def plot_dwell_run_length_violin(
    dwell_run_lengths_by_state: Mapping[Any, Sequence[int]],
    state_order: Optional[Sequence[Any]] = None,
    ax=None,
    figsize: Tuple[float, float] = (9, 4),
    title: str = "Run length (windows) within superwindows",
):
    """
    Violin plot of pooled run lengths per state (each run is one sample).

    Parameters
    ----------
    dwell_run_lengths_by_state
        From :func:`summarize_superwindow_sequences` key ``dwell_run_lengths_by_state``.
    state_order
        Optional order of states on the x-axis.
    """
    import matplotlib.pyplot as plt

    if not dwell_run_lengths_by_state:
        raise ValueError("No dwell run lengths to plot.")

    if state_order is None:
        states = sorted(dwell_run_lengths_by_state.keys(), key=lambda x: str(x))
    else:
        states = list(state_order)

    data = []
    labs = []
    for st in states:
        if st not in dwell_run_lengths_by_state:
            continue
        vals = list(dwell_run_lengths_by_state[st])
        if not vals:
            continue
        data.append(vals)
        labs.append(str(st))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.65)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("Run length (windows)")
    ax.set_title(title)
    return ax.figure, ax


def plot_bootstrap_summary_bars(
    bootstrap_result: Mapping[str, Any],
    figsize: Tuple[float, float] = (8, 4),
    title: str = "Bootstrap CIs (resample superwindows)",
):
    """
    Bar + error bars from :func:`bootstrap_superwindow_metrics` (ci_low / ci_high).
    """
    import matplotlib.pyplot as plt

    res = bootstrap_result.get("results", {})
    if not res:
        raise ValueError("No bootstrap results to plot.")

    names = list(res.keys())
    points = [res[k]["point_estimate"] for k in names]
    lo = [res[k]["ci_low"] for k in names]
    hi = [res[k]["ci_high"] for k in names]
    err_lo = [p - l for p, l in zip(points, lo)]
    err_hi = [h - p for p, h in zip(points, hi)]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    ax.bar(x, points, yerr=[err_lo, err_hi], capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    n = bootstrap_result.get("n", "")
    ax.text(
        0.02,
        0.98,
        f"n superwindows = {n}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    return fig, ax
