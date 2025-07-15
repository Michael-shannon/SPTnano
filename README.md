# SPTnano

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/Michael-shannon/SPTnano/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/Michael-shannon/SPTnano/actions/workflows/tests.yml
[linting-badge]:            https://github.com/Michael-shannon/SPTnano/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/Michael-shannon/SPTnano/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/Michael-shannon/SPTnano/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/Michael-shannon/SPTnano/actions/workflows/docs.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/SPTnano
[conda-link]:               https://github.com/conda-forge/SPTnano-feedstock
[pypi-link]:                https://pypi.org/project/SPTnano/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/SPTnano
[pypi-version]:             https://img.shields.io/pypi/v/SPTnano
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

Single particle tracking analysis with machine learning. This package combines traditional trajectory analysis with transformer-based deep learning to classify motion patterns.

## About

### Project Team

Michael Shannon ([m.j.shannon@pm.me](mailto:m.j.shannon@pm.me))

The structure for this repo was made using a cookiecutter from the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## Installation

### Quick Install (Recommended)

```bash
pip install git+https://github.com/Michael-shannon/SPTnano.git
```

### For Development

```bash
git clone https://github.com/Michael-shannon/SPTnano.git
cd SPTnano
pip install -e .
```

If you get dependency errors, try installing them first:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch trackpy
```

## Getting Started

```python
import SPTnano as spt
import pandas as pd

# Load your trajectory data
df = pd.read_csv("your_trajectories.csv")

# Calculate motion features
metrics = spt.ParticleMetrics(df, time_between_frames=0.01)
metrics.calculate_time_windowed_metrics(window_size=60, overlap=30)

# Train a transformer model
results = spt.train_motion_transformer(df, epochs=25, use_tensorboard=True)

# Analyze results
windowed_df = metrics.get_time_windowed_df()
spt.plot_tracks_by_motion_class(windowed_df, metrics.get_time_averaged_df())
```

## Data Format

Your CSV should have these columns:

- `unique_id` - Unique identifier for each track
- `frame` - Frame number
- `x`, `y` - Coordinates
- `time_s` - Time in seconds
- `condition` - Experimental condition (optional)

## Configuration

Set your data directory:

```python
# Edit this line in src/SPTnano/config.py
MASTER = '/path/to/your/data/'
```

Or change it programmatically:

```python
import SPTnano as spt
spt.config.MASTER = '/path/to/your/data/'
```

## Features

- **Traditional Analysis**: MSD, diffusion coefficients, motion classification
- **Transformer Models**: Deep learning on trajectory windows
- **Visualization**: Track plotting, clustering, UMAP embeddings
- **TensorBoard**: Training monitoring and visualization

## Examples

Check out the `notebooks/` folder for examples:

## Troubleshooting

**Import errors?** Make sure you installed the package: `pip install -e .`

**Missing dependencies?** Install them: `pip install torch numpy pandas matplotlib`

**Path issues?** Check your config: `print(spt.config.MASTER)`

## Contributing

This is research software - expect rough edges. Pull requests welcome.

## License

MIT License
