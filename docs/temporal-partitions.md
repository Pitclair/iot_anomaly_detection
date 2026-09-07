# Frozen temporal partitions

LM-IDNet partitions complete daily captures before any model fitting or
threshold calibration. It does not randomly distribute individual windows
across research partitions.

The frozen configuration is:

| Partition | Capture dates | Allowed purpose |
|---|---|---|
| `fit` | 2020-10-09 through 2020-10-13 | Estimate normal-model parameters |
| `calibration` | 2020-10-14 through 2020-10-15 | Select anomaly-score thresholds |
| `development_test` | 2020-10-16 and 2020-10-19 | Debug and evaluate before the locked result |
| `final_test` | 2020-10-21 through 2020-10-22 | One locked final evaluation only |

This uses 11 validated D-Link captures: 5 fit, 2 calibration, 2
development-test, and 2 final-test captures. The October 8 capture is excluded
because it overlaps October 9 and contains repeated packet occurrences; October
9 is the cleaner source for the shared period.

## Why complete captures are grouped

Windows from the same daily capture are temporally and behaviorally related. A
random window split could put neighboring windows from one capture into both
training and testing, allowing information about the test period to leak into
the fitted model.

The configuration therefore stores only capture identifiers at partition
boundaries. The active configuration assigns the experimental role; the role
stored during preprocessing is retained only as historical metadata. There is
no random window-partitioning API.

## Enforced invariants

Configuration loading fails when:

- a partition is empty;
- a capture identifier lacks a valid `YYYY-MM-DD` date;
- identifiers are duplicated within a partition;
- one capture appears in two partitions;
- identifiers within a partition are not chronological;
- an earlier partition ends on or after the next partition begins.

The required global order is:

```text
fit < calibration < development_test < final_test
```

## Access boundaries

Code should not select dates directly from nested configuration fields. Use the
purpose-scoped functions in `lm_idnet.partitioning`:

```python
fit_partition_for_training(config)
calibration_partition_for_threshold(config)
development_partition_for_evaluation(config)
final_partition_for_locked_evaluation(config)
```

Training receives only `fit`. Threshold calibration receives only
`calibration`. Neither accessor exposes final-test identifiers. All-source
access is reserved for ingestion operations such as inventory, structural PCAP
validation, and preprocessing.

## What “frozen” means

The lists are version-controlled research inputs, not values selected after
looking at final-test performance. Changing a date assignment changes the
experimental protocol and must be reviewed explicitly. Once final evaluation
begins, the lists must not be adjusted to improve reported results.

This split does not by itself prove that every fit or calibration interval is
benign. Dataset provenance and benign-interval assumptions require their own
review. It only establishes chronological, disjoint usage boundaries.

The predefined expanding development folds and their commands are documented
in [Temporal rolling folds](temporal-folds.md).
