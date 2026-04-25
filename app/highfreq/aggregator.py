"""1-second and 1-minute feature aggregation, Postgres writer.

Receives streams of L2 snapshots and trades from :mod:`app.highfreq.l2_consumer`,
computes :class:`~app.highfreq.ofi_features.OFIFeatures1s` once per second,
batches them, and flushes to the ``highfreq_ofi_1s`` table. A periodic
1-minute reducer aggregates the last 60 seconds into ``highfreq_features_1m``
and back-fills the ``return_1m`` / ``direction`` target columns once the
following minute completes.

Schema reference: see ``docs/highfreq/architecture.md`` §6.
"""
from __future__ import annotations

# Phase A.0 skeleton — implemented in Phase A.3 alongside the OFI feature
# computation.
