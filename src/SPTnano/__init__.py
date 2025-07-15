"""SPTnano package."""

# from ._version import __version__

from . import augmentations, config, tensorboard_utils, training_utils
from .batch_roi_selector import ROISelector, process_directory
from .features import ParticleMetrics
from .helper_scripts import *
from .visualization import *


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
