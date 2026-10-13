"""Paired statistics for expv1's gates and its comparison with v8 (numpy + scipy only).

Every comparison expv1 makes is PAIRED: the same held-out source clip, start frame, label
and noise for both sides. Paired tests use that pairing and unpaired ones would throw it
away.

  binary outcome per pair (1-NN correct, classifier correct, rubric pass):
      b = pairs only A gets right, c = pairs only B gets right (concordant pairs carry no
      information about the difference). Under H0 (equal accuracy) b ~ Binomial(b + c, 1/2):
      the exact McNemar test. One-sided p for "A better" = P(X >= b).
      p_better = P(theta > 1/2 | b, c), theta ~ Beta(b + 1, c + 1): the posterior probability
      that A wins a discordant pair more often than B (uniform prior). It is reported next
      to p and answers "how likely is it that A is better" directly.
  continuous outcome per pair (rank, motion ratio, zoom, identity):
      Wilcoxon signed-rank on the paired differences, and a percentile bootstrap CI of
      the mean difference (10,000 resamples of pairs, fixed seed).
  several primary tests: Holm step-down adjustment (strong family-wise error control).
"""
import math

import numpy as np

SEED = 20261013
B_BOOT = 10_000


def binom_tail_ge(k, n):
    """P(X >= k), X ~ Binomial(n, 1/2), exact."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def mcnemar(b, c):
    """Exact McNemar on discordant counts. b: A-only correct, c: B-only correct."""
    n = b + c
    return {"b": int(b), "c": int(c), "n_discordant": int(n),
            "p_A_better": binom_tail_ge(b, n),            # one-sided, H1: A > B
            "p_B_better": binom_tail_ge(c, n),            # one-sided, H1: B > A
            "p_two_sided": min(1.0, 2 * binom_tail_ge(max(b, c), n)) if n else 1.0,
            "p_better_posterior": p_better(b, c)}


def sign_test(pos, neg):
    n = pos + neg
    return min(1.0, 2 * binom_tail_ge(max(pos, neg), n)) if n else 1.0


def p_better(b, c):
    """P(theta > 1/2), theta ~ Beta(b + 1, c + 1)."""
    from scipy.stats import beta
    return float(beta.sf(0.5, b + 1, c + 1))


def min_b_for_significance(n_discordant, alpha=0.05):
    """Smallest b out of n discordant pairs that reaches one-sided p < alpha (None if none):
    what a result has to look like to count, given how many pairs actually disagree."""
    for b in range(n_discordant + 1):
        if binom_tail_ge(b, n_discordant) < alpha:
            return b
    return None


def wilcoxon_p(d, alternative="two-sided"):
    from scipy.stats import wilcoxon
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], np.float64)
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method="pratt", alternative=alternative).pvalue)


def bootstrap_ci(d, stat=np.mean, level=0.95, b=B_BOOT, seed=SEED):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], np.float64)
    if len(d) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    s = np.apply_along_axis(stat, 1, d[idx]) if stat is not np.mean else d[idx].mean(1)
    lo = (1 - level) / 2
    return [float(np.quantile(s, lo)), float(np.quantile(s, 1 - lo))]


def paired_rate_diff(a, b, level=0.95):
    """a, b: 0/1 arrays over the same pairs. Mean(a) - mean(b) with a bootstrap CI."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return {"diff": float(a.mean() - b.mean()) if len(a) else None,
            "ci": bootstrap_ci(a - b, level=level)}


def holm(pvals):
    """{name: p} -> {name: Holm-adjusted p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def simulate_null_false_positive(n_pairs, p_correct=0.6, rho=0.5, sims=4000, seed=SEED):
    """Type-I check of the exact McNemar one-sided test at alpha 0.05 when A and B are
    equally good (correlated outcomes). Must come out <= 0.05."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        z = rng.random(n_pairs)
        a = z < p_correct
        b = np.where(rng.random(n_pairs) < rho, a, rng.random(n_pairs) < p_correct)
        bb = int((a & ~b).sum())
        cc = int((b & ~a).sum())
        hits += binom_tail_ge(bb, bb + cc) < 0.05
    return hits / sims


def power_mcnemar(n_pairs, p_a, p_b, rho=0.5, alpha=0.05, sims=4000, seed=SEED):
    """Simulated power of the one-sided exact McNemar test for true accuracies p_a > p_b."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        u = rng.random(n_pairs)
        shared = rng.random(n_pairs) < rho
        a = u < p_a
        b = np.where(shared, u < p_b, rng.random(n_pairs) < p_b)
        bb = int((a & ~b).sum())
        cc = int((b & ~a).sum())
        hits += binom_tail_ge(bb, bb + cc) < alpha
    return hits / sims
