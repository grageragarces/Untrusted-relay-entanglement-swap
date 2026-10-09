# When lies look like noise: certifying untrusted entanglement swapping

This is the code behind the numerical results in the paper. One script,
`sim_validation.py`, computes all of them and prints them to stdout.

## Running it

```sh
pip install -r requirements.txt
python3 sim_validation.py
```

It needs Python 3.9 or later and NumPy. It runs in about a minute on a laptop.
The Monte-Carlo parts use a fixed seed, so the output is the same on every run.

## What it reproduces

| Output section | In the paper |
|---|---|
| `TEST PAIRS vs f_target` | Test pairs against the target rate (`tab:k-vs-f`), as LaTeX table rows |
| `SCALING CHECK`, `THRESHOLDS` | Scaling of `k` with `f_target`; floors on `f_target` |
| `TRANSFER VARIANTS` | Comparison of the crude, Hoeffding and Serfling sampling transfers |
| `DEMONSTRATED MEMORY` | Demonstrated memories converted to delivered pairs; smallest useful batch |
| `JOINT BUDGET`, `STORAGE-NOISE LIFETIME` | Best certified rate and memory lifetime against memory budget (`tab:joint`) |
| `NO REALISATION AT f=0 UNDER NOISE` | Numerical values of the bound on η in `prop:f0-noisy` |
| `PER-BLOCK RELEASE` | Block-wise verification (`tab:blocks`) |
| `fig:k-vs-f` | pgfplots coordinates for the test-pair figure |
| `DETECTION CURVE`, `FALSE REJECTION`, `MIS-SPECIFIED BASELINE`, `SYMMETRISATION` | Monte-Carlo checks of soundness, completeness, calibration sensitivity and the permutation step |

The bounds use the Bernstein form of the concentration lemma together with
Serfling's sampling bound. Comments in the script give the derivation of
each margin.
