# When lies look like noise: certifying untrusted entanglement swapping

This is the code behind the numerical results in the paper. One script,
`sim_validation.py`, computes all of them and prints them to stdout.

## Running it

```sh
pip install -r requirements.txt
python3 sim_validation.py
```

It needs Python 3.9 or later and NumPy, and runs in under a minute on a
laptop. The Monte-Carlo parts use a fixed seed, so the output is the same on
every run.

## What it reproduces

Table, figure and section numbers refer to the paper.

| Output section | In the paper |
|---|---|
| `TEST PAIRS vs f_target` | Test pairs against the target rate (Table II), as LaTeX table rows |
| `SCALING CHECK`, `THRESHOLDS` | Scaling of `k` with `f_target`, floors on `f_target`, and the batch needed to reach `f_target = 1e-3` (Sec. VII B) |
| `TRANSFER VARIANTS` | Crude, Hoeffding and Serfling sampling transfers, and Hoeffding against Bernstein estimation margins (Sec. VII B) |
| `DEMONSTRATED MEMORY` | Demonstrated memories converted to delivered pairs, and the smallest useful batch (Sec. VII C) |
| `JOINT BUDGET`, `STORAGE-NOISE LIFETIME` | Best certified rate and memory lifetime against memory budget (Table III, Secs. VII D and VII E) |
| `NO REALISATION AT f=0 UNDER NOISE` | Numerical values of the bound on η in Proposition 4 |
| `PER-BLOCK RELEASE` | Block-wise verification (Table IV) |
| `fig:k-vs-f` | The pgfplots coordinates plotted in Fig. 3 |
| `RESOLUTION AT THE OPERATING POINT` | Resolution of the test against an environment holding every pair (Sec. VI A) |
| `DETECTION CURVE`, `FALSE REJECTION`, `MIS-SPECIFIED BASELINE`, `SYMMETRISATION` | Monte-Carlo checks of soundness, completeness, calibration sensitivity (Lemma 5) and the permutation step (Remark 7) |

The bounds use the Bernstein form of the concentration lemma (Lemma 4)
together with Serfling's sampling bound. Comments in the script give the
derivation of each margin.
