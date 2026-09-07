# Temporal rolling folds

The fold configurations keep complete captures together and move strictly from
earlier fitting data to later calibration data and then later validation data.
The locked final captures are present only to preserve the outer boundary; the
fold runner never evaluates them.

## Predefined folds

Chromecast has three expanding folds:

| Fold | Fit | Calibration | Validation |
|---|---|---|---|
| 1 | Oct 10–13 | Oct 14–15 | Oct 16–17 |
| 2 | Oct 10–15 | Oct 16–17 | Oct 18–19 |
| 3 | Oct 10–17 | Oct 18–19 | Oct 20–23 |

Samsung Camera has two folds:

| Fold | Fit | Calibration | Validation |
|---|---|---|---|
| 1 | May 28–29 | May 30 | May 31 |
| 2 | May 28–30 | May 31 | Jun 1–4 |

D-Link folds are not defined because these captures have no attack labels and
therefore cannot support attack-episode recall or detector selection. Its
October 8 capture is excluded from the fixed configuration because it overlaps
October 9 and contains repeated packet occurrences.

## Run a fold

Each configuration has separate model, threshold, event, and report paths. Run
one complete fold from the repository root with:

```sh
scripts/run_temporal_fold.sh configs/temporal_folds/unsw_chromecast_fold_1.json
```

The runner validates the canonical timestamp timeline, refits the model,
recalibrates its threshold, scores only the fold's validation captures, and
writes window, episode, and per-capture statistics.

Run the remaining folds by substituting their configuration paths from
`configs/temporal_folds/`.

## Summarise and select

First decide an acceptable false-alert episode rate without looking at final
test results. Then summarize the completed reports, for example:

```sh
python -m lm_idnet.evaluation.fold_selection \
  --max-false-alert-episodes-per-day 2 \
  --output reports/temporal-fold-summary.json \
  reports/temporal-folds/UNSW-Chromecast/fold-*/evaluation_statistics.json
```

Candidates are grouped by score type, calibration quantile, and window length.
The selector chooses the eligible candidate with the highest attack-episode
recall, breaking a tie in favour of the lower false-alert episode rate. To
compare another quantile, copy the fold configurations, change the quantile,
and give every copy its own output directory. Partition dates must not be
changed in response to their results.

Capture summaries report the minimum, median, and maximum before pooled totals.
Confidence intervals, if later required, must resample complete captures or
days rather than individual windows.
