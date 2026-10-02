# Local evolutionary comparison

Measured on held-out synthetic workflows. Token counts below are simulation estimates.
The harness supplies reference repairs; the decision model does not write code.

| Policy | Completed | Proxy tokens per episode | Steps | Redundant steps |
|---|---:|---:|---:|---:|
| linear | 24/24 | 562.5 | 6.25 | 0.50 |
| neural | 12/24 | 690.0 | 8.50 | 4.25 |
| rules | 24/24 | 597.5 | 6.75 | 0.00 |
| random | 1/24 | 1130.4 | 12.00 | 8.83 |

Validation-selected candidate: **linear**.

No live Codex savings are established by this comparison.

- Small scripted Python workflows with oracle repairs
- No calibrated probability or general coding capability
- Proxy token counts do not demonstrate Codex token savings
