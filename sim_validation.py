"""Numerics and Monte-Carlo validation for "When lies look like noise:
certifying untrusted entanglement swapping".  Reproduces the test-pair table
and figure, the joint memory budget, storage-noise lifetimes and block-wise
release tables, and the other figures quoted in the resource estimation.
Run: python3 sim_validation.py

The concentration lemma establishes that the test statistic has the moment
generating function of a Binomial(m, pbar), so every Chernoff bound for that
distribution applies.  We use the Bernstein form throughout: the quantity
being estimated is an anti-correlation rate close to zero, where a bound
carrying the variance beats one that does not.
"""
import math
import numpy as np

rng = np.random.default_rng(7)
M = 10**6


def L(d):
    return math.log(6 / d)


# ------------------------------------------------------- Bernstein margins
# theta: exact positive root of  m t^2 - (2L/3) t - 2 L g(1-g) = 0.
# eps:   exact positive root of  m e^2 - (8L/3) e - 2 L (g+theta) = 0,
#        the 8/3 absorbing the unknown true rate via a(1-a) <= a <= g+theta+e.

def theta(k, d, g):
    m, l = k / 2, L(d)
    return l / (3 * m) + math.sqrt(l**2 / (9 * m**2) + 2 * g * (1 - g) * l / m)


def eps(k, d, g):
    m, l = k / 2, L(d)
    x = g + theta(k, d, g)
    return 4 * l / (3 * m) + math.sqrt(16 * l**2 / (9 * m**2) + 2 * l * x / m)


def tsamp(k, d, M=M):
    """Serfling sampling tolerance in the k-condition."""
    N = M + k
    return math.sqrt((1 - (k - 1) / N) * L(d) / (2 * k))


def lhs(k, d, gmax=0.0, M=M):
    """Left-hand side of the k-condition; compare against f_target*(1-gmax)."""
    g = gmax / 2
    return 2 * (theta(k, d, g) + eps(k, d, g)) + (2 * k / M) * tsamp(k, d, M)


def kasym(f, d, gmax=0.0):
    """Asymptotic scaling, for the discussion only -- NOT a usable closed form.

    The two regimes are separated by whether theta is small against g:
      noise-dominated (k >> L/g):  theta ~ eps ~ 2 sqrt(gL/k),  k ~ 32 gmax L/F^2
      shot-noise limited (g -> 0): theta ~ 2L/(3m), eps ~ (4+2 sqrt7) L/(3m),
                                   k ~ 4(6+2 sqrt7) L/(3F) ~ 15.1 L/F
    Neither is accurate near the crossover, and the additive L/m terms are not
    negligible at the k the protocol actually uses, so the exact condition is
    what the test-pair table solves.  Kept only to check the exponents quoted in the text.
    """
    F = f * (1 - gmax)
    return 32 * gmax * L(d) / F**2, 4 * (6 + 2 * math.sqrt(7)) * L(d) / (3 * F)


def _floor(d, gmax, M=M):
    """min over k of lhs.  Not monotone: the estimation terms fall while the
    sampling transfer rises, so bracket the minimum rather than bisecting."""
    ks = np.unique(np.logspace(1, 12, 4000).astype(np.int64))
    v = [lhs(int(k), d, gmax, M) for k in ks]
    i = int(np.argmin(v))
    lo, hi = int(ks[max(0, i - 1)]), int(ks[min(len(ks) - 1, i + 1)])
    while hi - lo > 4:
        a, b = lo + (hi - lo) // 3, hi - (hi - lo) // 3
        if lhs(a, d, gmax, M) <= lhs(b, d, gmax, M):
            hi = b
        else:
            lo = a
    kb = min(range(lo, hi + 1), key=lambda k: lhs(k, d, gmax, M))
    return lhs(kb, d, gmax, M), kb


def kexact(f, d, gmax=0.0, M=M):
    """Smallest even k satisfying the k-condition; None if unreachable."""
    tgt = f * (1 - gmax)
    fmin, kstar = _floor(d, gmax, M)
    if fmin > tgt:
        return None
    lo, hi = 4, kstar                    # lhs is decreasing on [4, kstar]
    while lo < hi:
        mid = (lo + hi) // 2
        if lhs(max(mid, 4), d, gmax, M) <= tgt:
            hi = mid
        else:
            lo = mid + 1
    return int(math.ceil(lo / 2) * 2)


# ------------------------------------------- Test-pair table (tab:k-vs-f)
GAMMAS = [0.0, 0.01, 0.05]
FS = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01,
      0.007, 0.005]

print("=== TEST PAIRS vs f_target, tab:k-vs-f  (eps_sec=0.01, M=1e6) ===")
print("% f_target & k(g=0) & k(g=0.01) & k(g=0.05) & k/M at g=0.05")
for f in FS:
    cells = []
    for g in GAMMAS:
        k = kexact(f, 0.01, g)
        cells.append("---" if k is None else f"{k:,}")
    kl5 = kexact(f, 0.01, 0.05)
    rat = "---" if kl5 is None else f"{kl5 / M * 100:.1f}\\%"
    row = " & ".join(f"${c}$" for c in cells)
    print(f"${f:.3f}$ & {row} & ${rat}$ \\\\".replace(",", "{,}"))

print("\n=== SCALING CHECK: exponent of f_target, per gamma ===")
for g in GAMMAS:
    row = []
    for f in [0.04, 0.02, 0.01]:
        row.append(kexact(f, 0.01, g))
    e1 = math.log(row[1] / row[0]) / math.log(2)
    e2 = math.log(row[2] / row[1]) / math.log(2)
    print(f"  gamma={g:<5} k(.04)={row[0]:>9,} k(.02)={row[1]:>9,} "
          f"k(.01)={row[2]:>9,}   f^-{e1:.2f}, f^-{e2:.2f}")
print("  (expected: f^-1 as gamma->0, f^-2 once noise dominates)")

print("\n=== THRESHOLDS (eps_sec=0.01, M=1e6) ===")
for g in GAMMAS:
    fmin, kb = _floor(0.01, g)
    lo, hi = 1e-4, 0.5                       # smallest f_target with k <= M
    for _ in range(80):
        mid = (lo + hi) / 2
        k = kexact(mid, 0.01, g)
        if k is None or k > M:
            lo = mid
        else:
            hi = mid
    print(f"  gamma={g:<5} floor f_target={fmin/(1-g)*100:.3f}% (at k={kb:,})"
          f"   k=M at f_target={hi*100:.3f}%")
for d in [1e-2, 1e-6, 1e-9]:
    f0, _ = _floor(d, 0.05)
    print(f"  floor at eps_sec={d:g}, gamma=0.05: {f0/0.95*100:.3f}%")


def floor_f(d, g, Mb, kcap=None):
    """Floor on f_target at delivered batch Mb; kcap=None leaves k unbounded."""
    if kcap is None:
        return _floor(d, g, Mb)[0] / (1 - g)
    ks = np.unique(np.logspace(1, math.log10(kcap), 3000).astype(np.int64))
    return min(lhs(int(x), d, g, Mb) for x in ks) / (1 - g)


def batch_for(target, d, g, cap):
    """Smallest delivered batch (to ~1%) whose floor reaches target."""
    lo, hi = 10**6, 10**9
    while hi / lo > 1.01:
        mid = int(math.sqrt(lo * hi))
        if floor_f(d, g, mid, mid if cap else None) <= target:
            hi = mid
        else:
            lo = mid
    return hi


print("  delivered batch needed for the floor to reach f_target=1e-3:")
for g in GAMMAS:
    print(f"    gamma={g:<5} {batch_for(1e-3, 0.01, g, False):.2e} (k unbounded), "
          f"{batch_for(1e-3, 0.01, g, True):.2e} (k <= M)")

# --------------------------------------------- three transfers, side by side
print("\n=== TRANSFER VARIANTS (eps_sec=0.01, gamma_max=0.05, M=1e6) ===")


def kvariant(fn, f, kmax):
    """Smallest k with fn(k) <= f.  Each variant is falling-then-rising, so
    locate the minimum first and bisect only on the falling branch."""
    ks = np.unique(np.logspace(1, math.log10(kmax), 3000).astype(np.int64))
    v = [fn(int(x)) for x in ks]
    i = int(np.argmin(v))
    if v[i] > f:
        return None
    lo, hi = 4, int(ks[i])
    while lo < hi:
        mid = (lo + hi) // 2
        if fn(max(mid, 4)) <= f:
            hi = mid
        else:
            lo = mid + 1
    return int(math.ceil(lo / 2) * 2)


_est = lambda k: 2 * (theta(k, 0.01, 0.025) + eps(k, 0.01, 0.025))
_crude = lambda k: (1 + k / M) * _est(k)
_hoefft = lambda k: _est(k) + (2 * k / M) * math.sqrt(L(0.01) / (2 * k))
_serf = lambda k: lhs(k, 0.01, 0.05)
print(f"{'f_target':>9}  {'crude N/M':>11}  {'Hoeffding':>11}  {'Serfling':>11}")
for f in [0.05, 0.03, 0.015, 0.01, 0.007, 0.005]:
    row = []
    for fn, km in ((_crude, M), (_hoefft, 1e9), (_serf, 1e12)):
        k = kvariant(fn, f * 0.95, km)
        row.append(f"{k:,}" if k else "none")
    print(f"{f:>9.3f}  {row[0]:>11}  {row[1]:>11}  {row[2]:>11}")
print("  floor on f_target for each transfer route (gamma_max=0.05):")
for name, fn, km in (("crude N/M", _crude, M), ("Hoeffding", _hoefft, 1e9),
                     ("Serfling ", _serf, 1e12)):
    ks = np.unique(np.logspace(1, math.log10(km), 4000).astype(np.int64))
    fl = min(fn(int(x)) for x in ks)
    print(f"    {name}: {fl / 0.95 * 100:.3f}%")

# Estimation margins from Hoeffding instead of Bernstein: theta = eps =
# sqrt(L/(2m)) with m = k/2, Serfling transfer unchanged.
_hoefm = lambda k: 4 * math.sqrt(L(0.01) / k) + (2 * k / M) * tsamp(k, 0.01)
kH, kB = kvariant(_hoefm, 0.01 * 0.95, 1e9), kexact(0.01, 0.01, 0.05)
print(f"  Hoeffding estimation margins at f_target=0.01: k={kH:,}, "
      f"{kH / kB:.1f}x the Bernstein k={kB:,}")

# ------------------------------------------- joint memory / precision budget
# Table: the smallest deviation rate certifiable at a given memory budget,
# optimised over the split N = M + k.  The two requirements of "Test pairs and
# memory together" are
# functions of the same N and pull against each other; this is where they meet.

def f_split(M, k, d, gmax):
    """f_target admitted by the k-condition at a given split N = M + k."""
    g, N = gmax / 2, M + k
    val = (2 * (theta(k, d, g) + eps(k, d, g))
           + (2 * k / M) * math.sqrt((1 - (k - 1) / N) * L(d) / (2 * k)))
    return val / (1 - gmax)


def best_under_budget(Ncap, gmax, d=0.01):
    """min over even k of the f_target admitted at M + k = Ncap.  Exhaustive
    for small budgets; for large ones a 2e5-point scan refined locally.  (An
    earlier 400-point grid gave M = 765 at Ncap = 1843 where the optimum is 763.)"""
    Ncap = int(Ncap)
    if Ncap <= 200_000:
        ks = range(2, Ncap - 1, 2)
    else:
        ks = sorted({int(x) // 2 * 2 for x in np.linspace(2, Ncap - 2, 200_000)})
    best = (float("inf"), None, None)
    for kk in ks:
        f = f_split(Ncap - kk, kk, d, gmax)
        if f < best[0]:
            best = (f, Ncap - kk, kk)
    if Ncap > 200_000:
        k0, step = best[2], 2 * (Ncap // 200_000 + 1)
        for kk in range(max(2, k0 - step), min(Ncap - 2, k0 + step) + 1, 2):
            f = f_split(Ncap - kk, kk, d, gmax)
            if f < best[0]:
                best = (f, Ncap - kk, kk)
    return best


def min_budget(gmax, target, d=0.01):
    """Smallest memory budget N at which some split certifies target."""
    lo, hi = 4, 10**7
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if best_under_budget(mid, gmax, d)[0] <= target:
            hi = mid
        else:
            lo = mid
    return hi


# Demonstrated memories, in DELIVERED pairs ("Quantum memory requirement").  Pompili et al. 2021
# report N_1/e = 1843 in entanglement ATTEMPTS (Table S3); an attempt lasts
# 3.8 us and succeeds with probability between 9 Hz * 3.8 us (rate bound) and
# alpha_A p_A + alpha_B p_B (Table S2).  Bradley et al. 2019: 10-qubit register,
# >75 s single-qubit and >10 s two-qubit storage, idle.
N1E_ATTEMPTS, RATE, T_ATT = 1843, 9.0, 3.8e-6
p_lo, p_hi = RATE * T_ATT, 0.07 * 3.6e-4 + 0.05 * 4.4e-4
print("\n=== DEMONSTRATED MEMORY, IN DELIVERED PAIRS ===")
print(f"  success prob per attempt in [{p_lo:.2e}, {p_hi:.2e}]")
print(f"  N_1/e = {N1E_ATTEMPTS} attempts -> {N1E_ATTEMPTS*p_lo:.3f}"
      f"..{N1E_ATTEMPTS*p_hi:.3f} heralded pairs;  T2* = 11.6 ms -> "
      f"{0.0116*RATE:.2f} pairs")
print(f"  Bradley register: capacity 10; 75 s -> {75*RATE:.0f} pairs, "
      f"10 s -> {10*RATE:.0f} pairs (idle)")
for g in (0.0, 0.05):
    print(f"  gamma={g}: smallest N certifying f<=1: {min_budget(g, 1.0):,}, "
          f"f<=0.5: {min_budget(g, 0.5):,}, f<=1e-2: {min_budget(g, 0.01):,}")

print("\n=== JOINT BUDGET, tab:joint ===")
print(f"{'N_cap':>10} {'coh@9Hz':>9} {'g=0':>8} {'g=0.05':>8}   split (M,k) at g=0.05")
for Ncap in [10**3, 10**4, 10**5, 730_000]:
    f0, M0, k0 = best_under_budget(Ncap, 0.0)
    f5, M5, k5 = best_under_budget(Ncap, 0.05)
    print(f"{Ncap:>10,} {Ncap/RATE/60:>7.0f}mn {100*f0:>7.2f}% {100*f5:>7.2f}%"
          f"   M={M5:,}, k={k5:,}   (g=0: M={M0:,}, k={k0:,})")

# Storage noise ("Memory lifetime under storage noise").  A pair
# stored for time t has gamma(t) = 1 - (1 - gamma_0) exp(-2 t / T_mem), and the
# oldest pair of a batch is stored for N rounds.  Holding it within gamma_cap
# needs T_mem / tau >= 2N / log((1 - gamma_0) / (1 - gamma_cap)).
def gamma_aged(N, g0, lam):
    """Per-pair baseline of the oldest pair after N rounds, lifetime lam rounds."""
    return 1 - (1 - g0) * math.exp(-2 * N / lam)


def lifetime_needed(N, g0, gcap=0.05):
    return 2 * N / math.log((1 - g0) / (1 - gcap))


print("\n=== STORAGE-NOISE LIFETIME (gamma_cap=0.05, eps_sec=0.01) ===")
for tgt in (0.5, 0.1, 0.03, 0.01):
    Nreq = min_budget(0.05, tgt)
    l0, l1 = lifetime_needed(Nreq, 0.0), lifetime_needed(Nreq, 0.01)
    print(f"  f={100*tgt:>4.0f}%  N={Nreq:>8,}  T_mem/tau: g0=0 {l0:.2e}, "
          f"g0=0.01 {l1:.2e}  ({l1/RATE/86400:.2f} d at 9 Hz)")
print(f"  memory lifetime per batch length: {lifetime_needed(1, 0.0):.1f}x (g0=0), "
      f"{lifetime_needed(1, 0.01):.1f}x (g0=0.01)")
print("  lifetime columns of tab:joint:")
for Ncap in [10**3, 10**4, 10**5, 730_000]:
    l0, l1 = lifetime_needed(Ncap, 0.0), lifetime_needed(Ncap, 0.01)
    print(f"    N={Ncap:>8,}  g0=0 {l0:.1e}, g0=0.01 {l1:.1e}"
          f"  ({l1/RATE/3600:.1f} h at 9 Hz)")
# Raising the ceiling: optimise over N with gamma_max set by storage itself.
for lam in (2.2e7, 2.3e7):
    fb = min((best_under_budget(n, gamma_aged(n, 0.01, lam))[0], n)
             for n in np.geomspace(1e6, 5e6, 30).astype(int))
    print(f"  uncapped, g0=0.01, T_mem/tau={lam:.1e}: best f={100*fb[0]:.3f}% "
          f"at N={fb[1]:,} (gamma_max={gamma_aged(fb[1], 0.01, lam):.3f})")

# No realisation at f = 0 under noise (Proposition "prop:f0-noisy").  Charlie
# puts sigma_x on a random fraction phi of the batch, shifting the Z rate by
# s = phi (1 - gamma); the environment measures all delivered pairs in Z and
# compares their anti-correlated fraction with g + tau.  Each term below is
# one line of the proposition's bound; we maximise over (s, t', tau).
def bern(m, x, v):
    return math.exp(-m * x * x / (2 * v + 2 * x / 3))


def eta_f0(Md, kk, gam, d=0.01):
    N, n, g = Md + kk, kk / 2, gam / 2
    th = theta(kk, d, g)
    best = (0.0, None)
    for s in np.linspace(0.05, 0.95, 91) * th:
        phi = s / (1 - gam)
        b = bern(n, th - s, g + s)
        for tp in np.linspace(0, 0.05, 201)[1:]:
            c1 = math.exp(-2 * kk * tp * tp / (1 - (kk - 1) / N))
            sD = (phi - kk / Md * tp) * (1 - gam)
            if sD <= 0:
                continue
            for tau in np.linspace(0.02, 0.98, 49) * sD:
                e = 1 - d / 6 - b - c1 - bern(Md, sD - tau, g + sD) \
                    - bern(Md, tau, g * (1 - g))
                if e > best[0]:
                    best = (e, (s / th, tp, tau))
    return best


print("\n=== NO REALISATION AT f=0 UNDER NOISE (gamma=0.05, M=1e6) ===")
for name, kk in (("operating", 168_982), ("loose", 2_420)):
    e, _ = eta_f0(M, kk, 0.05)
    print(f"  {name} point, k={kk:,}: eta >= {e:.2f}")

print("\n=== PER-BLOCK RELEASE, tab:blocks ( gamma=0.05, eps_sec/B per block) ===")
for nb, B in [(10**3, 1), (10**3, 100), (10**3, 1000), (10**4, 1),
              (10**4, 100), (10**5, 10), (10**5, 100)]:
    f, Mb, _ = best_under_budget(nb, 0.05, 0.01 / B)
    print(f"  n_b={nb:>7,} B={B:>5}: delivered M={B*Mb:>10,}  f_target={100*f:.1f}%")

# ----------------------------------------- Test-pair figure (fig:k-vs-f)
# 90 log-spaced f_target values from each curve's floor to 0.30, rounded to
# five decimals; k is evaluated at the rounded value and points above the
# plot's y-range (1.2e7) are dropped.  These are the coordinates in the paper.
print("\n=== fig:k-vs-f  (k vs f_target, one curve per gamma_max) ===")
for g in GAMMAS:
    fmin, _ = _floor(0.01, g)
    pts = []
    for f in np.logspace(math.log10(fmin / (1 - g)), math.log10(0.30), 90):
        f = round(float(f), 5)
        k = kexact(f, 0.01, g)
        if k and k <= 1.2e7:
            pts.append((f, k))
    print(f"% gamma_max={g:g}")
    print("    " + " ".join(f"({f:.5f},{k})" for f, k in pts))

# ------------------------------------- Detection curve at the operating point
d, ft, GS = 0.01, 0.01, 0.05          # the operating point of "Target misreport rate"
k = kexact(ft, d, GS)
m = k // 2
th = theta(k, d, GS / 2)
ep = eps(k, d, GS / 2)
a0 = GS / 2
print(f"\n=== RESOLUTION AT THE OPERATING POINT ===\n  anti-correlation rate"
      f" resolved to sqrt(a(1-a)/M)={100*math.sqrt(a0*(1-a0)/M):.3f} pp by an"
      f" environment holding all M pairs,\n  sqrt(a(1-a)/k)="
      f"{100*math.sqrt(a0*(1-a0)/k):.3f} pp by the test; acceptance window"
      f" 2*theta={200*th:.2f} pp")
print(f"\n=== DETECTION CURVE ===  gamma={GS} f_target={ft} k={k:,} m={m:,}"
      f"\n  theta={th:.6f} eps={ep:.6f}  eps/theta={ep/th:.2f}"
      f"\n  proven f<={ft}   per-basis crossing f=2*theta/(1-gamma)="
      f"{2*th/(1-GS):.5f}"
      f"\n  looseness = proven / per-basis crossing = {ft/(2*th/(1-GS)):.2f}x")


def run(f, T, gam_t=GS, gam_a=None, kk=None, mm=None, thr=None, seed=None):
    """Empirical rejection rate.  Worst-case adversary: f/2 sigma_x, f/2 sigma_z.
    gam_t is the true baseline, gam_a the one the test is run against."""
    kk = k if kk is None else kk
    mm = m if mm is None else mm
    thr = th if thr is None else thr
    if gam_a is None:
        gam_a = gam_t
    N = M + kk
    nx = nz = int(round(f * N / 2))          # deviated pairs, by type
    rej = 0
    for _ in range(T):
        bad = False
        for nb in (nx, nz):                  # V_Z = sigma_x pairs, V_X = sigma_z
            vis = rng.hypergeometric(nb, N - nb, mm)
            anti = rng.binomial(vis, 1 - gam_t / 2) + \
                   rng.binomial(mm - vis, gam_t / 2)
            if anti / mm > gam_a / 2 + thr:
                bad = True
        rej += bad
    return rej / T


pts = []
x0 = 2 * th / (1 - GS)                       # per-basis crossing
grid = sorted({0.0} | {round(x0 * r, 5) for r in
                       [.2, .4, .6, .7, .8, .85, .9, .95, 1.0, 1.05, 1.1,
                        1.2, 1.3, 1.5, 1.8, 2.2]} | {ft})
for f in grid:
    # the soundness claim rests on the points at and above f_target, so those
    # get enough trials for a one-sided 95% Clopper-Pearson bound on the
    # false-acceptance rate to fall below the budgeted eps_sec/3.
    T = 3000 if f >= ft else (1500 if f <= 1.3 * x0 else 800)
    pr = run(f, T)
    pts.append((f, pr))
    flag = ""
    if pr == 0.0:
        flag = f"   [0/{T}: P(accept) <= {100*(1-0.05**(1/T)):.3f}% at 95%]"
    print(f"  f={f:.5f}  P(reject)={pr:.4f}{flag}")
print("    " + " ".join(f"({f:.5f},{p:.4f})" for f, p in pts))

# ------------------------------------------------- honest-side false rejection
print("\n=== FALSE REJECTION (honest) ===")
for gam in (0.0, 0.05):
    T = 200000
    draws = rng.binomial(m, gam / 2, size=(T, 2)) / m
    exc = draws - gam / 2
    rej = int((exc > th).any(axis=1).sum())
    print(f"  gamma={gam}: P(reject)={rej/T:.2e} (budget {d/3:.2e}), "
          f"mean |excess|={abs(exc).mean():.5f}, max excess={exc.max():.5f}, "
          f"theta={th:.5f}")

# ------------------------------------------------- baseline mis-specification
print("\n=== MIS-SPECIFIED BASELINE (true gamma=0.05) ===")
for Da in (-0.02, -0.01, 0.0, 0.01, 0.02):
    honest = run(0.0, 400, gam_t=0.05, gam_a=0.05 + Da)
    lo, hi = 0.0, 0.30
    for _ in range(14):
        mid = (lo + hi) / 2
        if run(mid, 120, gam_t=0.05, gam_a=0.05 + Da) < 0.5:
            lo = mid
        else:
            hi = mid
    pred = 2 * (th + Da / 2) / (1 - 0.05)
    print(f"  Delta={Da:+.2f}: P(reject|honest)={honest:.3f}  "
          f"f at 50% detection={lo:.4f}  per-basis prediction {pred:.4f}")


# ------------------------------------------------- symmetrisation (step 5)
# Remark: a bound on the deviation RATE is not a bound a decoder can consume
# until the POSITIONS are dispersed.  Charlie fixes his pattern before T is
# drawn, and removing a random test set does not reorder the survivors, so a
# contiguous block of deviations stays contiguous in the delivered set.  The
# permutation pi of step 5 is the symmetrisation step standard in QKD proofs;
# after it, a block of B consecutive delivered pairs carries a hypergeometric
# count with mean f*B, which is the concentration lemma
# (lem:conc) at p_i in {0,1}.

def block_bound(B, f, nblocks, epsp=0.01):
    """Concentration lemma (Bernstein form) at pbar=f, union-bounded over nblocks."""
    l = math.log(nblocks / epsp)
    return f + l / (3 * B) + math.sqrt(l**2 / (9 * B**2) + 2 * f * (1 - f) * l / B)


def symmetrisation(f=0.01, Mdel=M, kk=168_982, trials=20, seed=11):
    r = np.random.default_rng(seed)
    N = Mdel + kk
    w = int(round(f * N))                      # contiguous deviation block
    out = {}
    for B in (10**4, 10**3):
        nb = Mdel // B
        clust, perm = [], []
        for _ in range(trials):
            T = r.choice(N, size=kk, replace=False)
            mask = np.ones(N, bool); mask[T] = False
            D = np.flatnonzero(mask)
            V = np.zeros(N, bool); V[:w] = True
            idx = np.flatnonzero(V[D])                     # ranks within D
            clust.append(np.bincount(idx * nb // len(D), minlength=nb).max() / B)
            pr = r.permutation(len(D))[idx]                # symmetrised order
            perm.append(np.bincount(pr * nb // len(D), minlength=nb).max() / B)
        out[B] = (max(clust), max(perm), block_bound(B, f, nb))
    return out


print("\n=== SYMMETRISATION (f=0.01, M=1e6, k=168,982) ===")
for B, (c, p_, b) in symmetrisation().items():
    print(f"  blocks of B={B:>6,}: worst block  index order {100*c:5.1f}%  "
          f"|  permuted {100*p_:.2f}%  (Lemma 4 bound {100*b:.2f}%)")
