# Deterministic random-number generation

LM-IDNet uses named seeds instead of NumPy's process-global random state.

The current reproducibility-critical seeds are:

```json
{
  "simulation": 1001,
  "model_fitting": 1002
}
```

`create_named_generators()` turns them into independent
`numpy.random.Generator` instances:

```python
generators = create_named_generators(config.seeds)
inject_anomaly(counts, generators.simulation)
fit_model(counts, generators.model_fitting)
```

Functions that need randomness must receive a generator explicitly. They must
not call `np.random.seed()`, `np.random.random()`, or another global-state API.

Separate seeds prevent an unrelated change in simulation draw count from
silently changing model initialization. Recreating generators from unchanged
configuration reproduces their streams. The smoke test also verifies that its
execution does not modify NumPy's legacy global RNG state.

Deterministic seeds make an experiment repeatable; they do not make a single
random realization statistically representative. Final experiments must still
use their predeclared seed policy and uncertainty analysis.
