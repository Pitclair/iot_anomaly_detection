# IoT Anomaly Detection Baseline

This repo provides a scaffold for modeling and forecasting IoT network traffic using Dirichlet-based methods.

Structure:
- `src/lm_idnet/`: installable Python package
- configs/config.json: default configuration
- `tests/`: automated tests

Install for local development:

```sh
python -m pip install -e .
```

Run the CLI from any directory without setting `PYTHONPATH`:

```sh
lm-idnet --help
lm-idnet model --config configs/config.json
lm-idnet forecast --config configs/config.json
```

`python main.py ...` remains available as a compatibility launcher after the
package has been installed.

Configuration fields and validation rules are documented in
[`docs/configuration.md`](docs/configuration.md).
