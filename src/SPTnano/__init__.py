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
