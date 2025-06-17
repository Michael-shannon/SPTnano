"""SPTnano package."""

# from ._version import __version__  # noqa: F401

from .batch_roi_selector import ROISelector, process_directory

from .helper_scripts import *

from .features import ParticleMetrics

from .visualization import *

from . import tensorboard_utils
from . import augmentations
from . import training_utils

# Import config from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
# from .helper_scripts import generate_file_tree, display_file_tree

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
