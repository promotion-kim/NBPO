# SafeRLHF Table 4

| Method | Help. | Harmless | Avg | Worst (95% CI) | WR_B | wWR_B |
|---|---:|---:|---:|---:|---:|---:|
| IPO | 0.607 | 0.627 | 0.617 | 0.438 [0.340, 0.540] | 0.699 | 0.612 |
| INPO (avg) | 0.586 | 0.596 | 0.591 | 0.410 [0.317, 0.503] | 0.719 | 0.602 |
| SimPO | 0.505 | 0.661 | 0.583 | 0.382 [0.298, 0.470] | 0.724 | 0.684 |
| DPO | 0.541 | 0.580 | 0.560 | 0.365 [0.279, 0.455] | 0.684 | 0.602 |
| RONPO (OS, confirmatory) | 0.572 | 0.485 | 0.529 | 0.305 [0.229, 0.384] | 0.653 | 0.510 |
| SPPO (avg) | 0.455 | 0.496 | 0.475 | 0.282 [0.200, 0.367] | 0.612 | 0.561 |
| HT-MNPO (harmless) | 0.464 | 0.452 | 0.458 | 0.281 [0.205, 0.356] | 0.653 | 0.531 |
| HT-MNPO (help.) | 0.409 | 0.488 | 0.449 | 0.253 [0.176, 0.334] | 0.643 | 0.551 |
| Base | 0.175 | 0.436 | 0.305 | 0.103 [0.048, 0.163] | -- | -- |

## Preregistered gate

```json
{
  "status": "fail",
  "worst_gate_pass": false,
  "average_floor_pass": false,
  "worst_comparator": "ipo",
  "average_comparator": "ipo",
  "worst_paired_difference": -0.1322899963591735,
  "worst_paired_difference_ci95": [
    -0.25287048804455536,
    -0.008083831597936172
  ],
  "avg_paired_difference": -0.08830879692738776,
  "avg_paired_difference_ci95": [
    -0.1859482418128512,
    0.011467250839544758
  ],
  "bootstrap_resamples": 2000,
  "bootstrap_seed": 42
}
```
