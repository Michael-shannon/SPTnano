"""SPTnano package."""

# from ._version import __version__

from . import augmentations, config, tensorboard_utils, training_utils
# Make batch_roi_selector import optional (requires nd2reader which isn't needed for training)
try:
    from .batch_roi_selector import ROISelector, process_directory
except ImportError:
    # nd2reader not installed - batch_roi_selector not available
    # This is fine for transformer training
    ROISelector = None
    process_directory = None

from .features import ParticleMetrics

# Make helper_scripts and visualization optional (not needed for training)
try:
    from .helper_scripts import *
except ImportError:
    pass  # Not needed for training

try:
    from .visualization import *
except (ImportError, ModuleNotFoundError):
    pass  # Requires Qt/napari - not needed for training

try:
    from .superwindow_statistics import (
        beta_binomial_proportion_ci,
        bootstrap_superwindow_metrics,
        bootstrap_transition_probabilities_by_group,
        compare_superwindow_metrics_by_group,
        concat_per_superwindow_by_group,
        dirichlet_smooth_transition_probs,
        first_transition_counts,
        per_superwindow_metrics,
        plot_bootstrap_summary_bars,
        plot_dwell_run_length_violin,
        plot_superwindow_group_comparison_bars,
        plot_superwindow_group_violin,
        run_lengths_from_sequence,
        summarize_superwindow_sequences,
        summarize_superwindow_sequences_by_group,
    )
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .attention import (
        cls_frame_attention,
        cls_query_attention_to_all_keys,
        correlate_embedding_dim_with_columns,
        diagnostic_raw_attn_column_sample,
        discover_per_frame_curve_column_names,
        flatten_batch_to_per_frame_row_dicts,
        flatten_batch_to_scalar_summary_row_dicts,
        layer_frame_attention_to_numpy,
        per_frame_attention_numpy,
        per_frame_attention_scalar_summaries,
        reduce_heads,
    )
except (ImportError, ModuleNotFoundError):
    pass  # optional interpretability helpers


def example_function(argument: str, keyword_argument: str = "default") -> str:
    """
    Concatenate string arguments - an example function docstring.

    Args:
        argument: An argument.
        keyword_argument: A keyword argument with a default value.

    Returns:
        The concatenation of `argument` and `keyword_argument`.

    """
    return argument + keyword_argument
