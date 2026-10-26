# Future work

The detector, labelled development/final evaluation, local supplied-window
edge simulation, and full project report complete the agreed project scope.
The selected method keeps the four-category Dirichlet–multinomial model,
jointly calibrated composition/volume/ARP scoring, fixed detection references,
conservative updates, and ten-minute windows. The closing results and protocol
are in [`reports/final_evaluation/`](../reports/final_evaluation/).

The following are optional extensions:

- Validate the frozen detector on new external labelled traffic, more devices,
  and longer benign periods to measure generalisation beyond the current
  two labelled reference devices.
- Integrate the historical temporal branch's expanding folds, capture and
  episode metrics, and timestamp validation with the selected detector.
  Add capture-block uncertainty. Resolve the D-Link overlap before new fits
  that rely on those numerical-only captures.
- Test legitimate drift and extra buffer/cadence settings under conservative
  guards. Candidate rejection is a valid result; stable predictions do not
  demonstrate that adaptation learned a better normal profile.
- Investigate thirty-minute static operation where response time permits.
  The completed 10/20/30/60-minute study retains ten minutes as the common
  default and identifies false-alert problems with coarse adaptive windows.
- Validate physical/ARM gateways, a live packet collector, durable event
  transport, and any response integration beyond the completed simulation.

The [thesis workspace](../thesis/README.md) now contains the complete draft,
checked evidence tables, and a validated PDF/A-2b document for the author's
final review. Personal cover details and submission approval remain part of
that review. The full project report remains the retained technical record.

Older proposals, implementation plans, and status snapshots preserve design
history and dated measurements. Their unchecked proposals do not expand the
finished software scope. Broader baselines and additional packet features
require a new, common evaluation protocol before being adopted.
