**All training windows (about 365 per fault)**

| Model and graph | Seeds | Macro-F1, 22 classes | Macro-F1 without faults 3, 9, 15 |
|---|---|---|---|
| MLP | 3 | 0.557 ± 0.000 | 0.624 ± 0.000 |
| GCN · knowledge | 3 | 0.463 ± 0.002 | 0.516 ± 0.003 |
| GCN · correlation | 3 | 0.564 ± 0.008 | 0.633 ± 0.011 |
| GCN · random | 3 | 0.551 ± 0.010 | 0.613 ± 0.010 |
| GCN · empty | 3 | 0.637 ± 0.004 | 0.711 ± 0.004 |
| GCN + residual · knowledge | 3 | 0.605 ± 0.009 | 0.672 ± 0.012 |
| GCN + residual · correlation | 3 | 0.611 ± 0.006 | 0.679 ± 0.005 |
| GCN + residual · random | 3 | 0.606 ± 0.013 | 0.675 ± 0.012 |
| GCN + residual · empty | 3 | 0.627 ± 0.005 | 0.696 ± 0.005 |
| GAT + residual · knowledge | 3 | 0.629 ± 0.009 | 0.697 ± 0.008 |
| R-GCN + residual · knowledge (typed) | 3 | 0.641 ± 0.003 | 0.712 ± 0.002 |
| R-GCN + residual · knowledge (one type) | 3 | 0.615 ± 0.007 | 0.686 ± 0.005 |
| R-GCN + residual · knowledge (shuffled types) | 3 | 0.623 ± 0.013 | 0.692 ± 0.013 |
| GCN + residual · knowledge, mean readout | 3 | 0.452 ± 0.011 | 0.506 ± 0.010 |

**30 training windows per class**

| Model and graph | Seeds | Macro-F1, 22 classes | Macro-F1 without faults 3, 9, 15 |
|---|---|---|---|
| MLP | 3 | 0.466 ± 0.003 | 0.523 ± 0.003 |
| GCN + residual · knowledge | 3 | 0.561 ± 0.009 | 0.626 ± 0.010 |
| GCN + residual · correlation | 3 | 0.557 ± 0.012 | 0.621 ± 0.015 |
| GCN + residual · random | 3 | 0.571 ± 0.010 | 0.634 ± 0.010 |
| GCN + residual · empty | 3 | 0.581 ± 0.017 | 0.645 ± 0.019 |
| R-GCN + residual · knowledge (typed) | 3 | 0.599 ± 0.013 | 0.668 ± 0.015 |
| R-GCN + residual · knowledge (one type) | 3 | 0.589 ± 0.020 | 0.656 ± 0.024 |
