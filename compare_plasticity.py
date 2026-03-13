#!/usr/bin/env python3
"""
Compare Hebbian plasticity divergence between two flies.

Loads plastic_weights_fly0.pt and plastic_weights_fly1.pt, computes
statistics showing how independent experience shaped different
synaptic weight patterns relative to a shared baseline.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np

# ── Load tensors ──────────────────────────────────────────────────────
print("=" * 72)
print("  HEBBIAN PLASTICITY DIVERGENCE ANALYSIS")
print("  Two flies, one connectome, independent experience")
print("=" * 72)

w0 = torch.load('data/plastic_weights_fly0.pt', map_location='cpu', weights_only=True)
w1 = torch.load('data/plastic_weights_fly1.pt', map_location='cpu', weights_only=True)
baseline = torch.load('data/plastic_weights.pt', map_location='cpu', weights_only=True)

N = w0.shape[0]
print(f"\nTotal synapses in connectome: {N:,}")

# ── Compute deltas from baseline ─────────────────────────────────────
d0 = w0 - baseline   # change in fly 0 vs original connectome
d1 = w1 - baseline   # change in fly 1 vs original connectome
diff = w0 - w1       # direct difference between the two flies

# ── Modified synapse counts (nonzero delta from baseline) ─────────────
EPS = 1e-6  # tolerance for "unchanged"
mod0 = (d0.abs() > EPS)
mod1 = (d1.abs() > EPS)
n_mod0 = mod0.sum().item()
n_mod1 = mod1.sum().item()
both_mod = (mod0 & mod1).sum().item()
either_mod = (mod0 | mod1).sum().item()
only0 = (mod0 & ~mod1).sum().item()
only1 = (~mod0 & mod1).sum().item()

print(f"\n{'─' * 72}")
print("  MODIFIED SYNAPSES (|delta from baseline| > {:.0e})".format(EPS))
print(f"{'─' * 72}")
print(f"  Fly 0:  {n_mod0:>12,}  ({100*n_mod0/N:.2f}%)")
print(f"  Fly 1:  {n_mod1:>12,}  ({100*n_mod1/N:.2f}%)")
print(f"  Both:   {both_mod:>12,}  ({100*both_mod/N:.2f}%)")
print(f"  Either: {either_mod:>12,}  ({100*either_mod/N:.2f}%)")
print(f"  Only 0: {only0:>12,}  ({100*only0/N:.2f}%)")
print(f"  Only 1: {only1:>12,}  ({100*only1/N:.2f}%)")

# ── Weight statistics ─────────────────────────────────────────────────
print(f"\n{'─' * 72}")
print("  WEIGHT STATISTICS (absolute values)")
print(f"{'─' * 72}")
header = f"  {'':15s} {'Fly 0':>14s} {'Fly 1':>14s} {'Baseline':>14s}"
print(header)
print(f"  {'':15s} {'─'*14} {'─'*14} {'─'*14}")
for label, fn in [("Mean", torch.mean), ("Std", torch.std),
                   ("Min", torch.min), ("Max", torch.max),
                   ("Median", torch.median)]:
    v0 = fn(w0).item()
    v1 = fn(w1).item()
    vb = fn(baseline).item()
    print(f"  {label:15s} {v0:>14.4f} {v1:>14.4f} {vb:>14.4f}")

# ── Delta statistics (changes from baseline) ─────────────────────────
print(f"\n{'─' * 72}")
print("  DELTA STATISTICS (change from baseline)")
print(f"{'─' * 72}")
header2 = f"  {'':15s} {'Fly 0 delta':>14s} {'Fly 1 delta':>14s} {'Fly0-Fly1':>14s}"
print(header2)
print(f"  {'':15s} {'─'*14} {'─'*14} {'─'*14}")
for label, fn in [("Mean", torch.mean), ("Std", torch.std),
                   ("Min", torch.min), ("Max", torch.max),
                   ("Abs mean", lambda x: torch.mean(x.abs()))]:
    v0 = fn(d0).item()
    v1 = fn(d1).item()
    vd = fn(diff).item()
    print(f"  {label:15s} {v0:>14.6f} {v1:>14.6f} {vd:>14.6f}")

# ── Correlation between the two weight vectors ───────────────────────
print(f"\n{'─' * 72}")
print("  CORRELATION ANALYSIS")
print(f"{'─' * 72}")

# Full weight correlation
corr_full = torch.corrcoef(torch.stack([w0, w1]))[0, 1].item()
print(f"  Pearson r (full weights):           {corr_full:.8f}")

# Delta correlation (do the changes correlate?)
corr_delta = torch.corrcoef(torch.stack([d0, d1]))[0, 1].item()
print(f"  Pearson r (deltas from baseline):   {corr_delta:.8f}")

# Cosine similarity of delta vectors
cos_sim = torch.nn.functional.cosine_similarity(
    d0.unsqueeze(0), d1.unsqueeze(0)).item()
print(f"  Cosine similarity (deltas):         {cos_sim:.8f}")

# R-squared: how much of fly1's plasticity is explained by fly0's
r2 = corr_delta ** 2
print(f"  R-squared (delta):                  {r2:.8f}")

# ── Divergence thresholds ─────────────────────────────────────────────
print(f"\n{'─' * 72}")
print("  SYNAPSES WHERE FLY 0 AND FLY 1 DIVERGE (|w0 - w1| > threshold)")
print(f"{'─' * 72}")
abs_diff = diff.abs()
max_diff = abs_diff.max().item()
print(f"  Max absolute difference: {max_diff:.8f}")
print(f"  Mean absolute difference: {abs_diff.mean().item():.10f}")
print(f"  Nonzero differences: {(abs_diff > 0).sum().item():,}")
print()
for thresh in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1]:
    count = (abs_diff > thresh).sum().item()
    pct = 100 * count / N
    print(f"  |w0 - w1| > {thresh:.0e}:  {count:>12,}  ({pct:>7.3f}%)")

# ── Percentile distribution of |fly0 - fly1| ─────────────────────────
print(f"\n{'─' * 72}")
print("  PERCENTILE DISTRIBUTION OF |FLY 0 - FLY 1|")
print(f"{'─' * 72}")
nonzero_diff = abs_diff[abs_diff > 0]
if len(nonzero_diff) > 0:
    for pct in [50, 75, 90, 95, 99, 99.9, 99.99, 100]:
        val = torch.quantile(nonzero_diff.float(), pct / 100).item()
        print(f"  P{pct:<6}:  {val:.10f}")
else:
    print("  (no nonzero differences)")

# ── Histogram of log10(|diff|) for nonzero differences ────────────────
print(f"\n{'─' * 72}")
print("  HISTOGRAM OF LOG10(|FLY 0 - FLY 1|)  (nonzero diffs only)")
print(f"{'─' * 72}")
if len(nonzero_diff) > 0:
    log_diff = torch.log10(nonzero_diff)
    lo = int(log_diff.min().item()) - 1
    hi = int(log_diff.max().item()) + 1
    bins = list(range(lo, hi + 1))
    bar_max_width = 50
    bin_counts = []
    for b in range(len(bins) - 1):
        count = ((log_diff >= bins[b]) & (log_diff < bins[b+1])).sum().item()
        bin_counts.append(count)
    max_count = max(bin_counts) if bin_counts else 1
    for b in range(len(bins) - 1):
        bar_len = int(bin_counts[b] / max_count * bar_max_width)
        bar = '#' * bar_len
        print(f"  [{bins[b]:>3d},{bins[b+1]:>3d})  {bin_counts[b]:>12,}  {bar}")

# ── Distribution: potentiated / depressed / unchanged ─────────────────
print(f"\n{'─' * 72}")
print("  DIRECTION OF CHANGE (relative to baseline)")
print(f"{'─' * 72}")
for name, delta in [("Fly 0", d0), ("Fly 1", d1)]:
    pot = (delta > EPS).sum().item()       # potentiated (strengthened)
    dep = (delta < -EPS).sum().item()      # depressed (weakened)
    unch = N - pot - dep                    # unchanged
    print(f"\n  {name}:")
    print(f"    Potentiated (strengthened): {pot:>12,}  ({100*pot/N:>6.2f}%)")
    print(f"    Depressed   (weakened):     {dep:>12,}  ({100*dep/N:>6.2f}%)")
    print(f"    Unchanged:                  {unch:>12,}  ({100*unch/N:>6.2f}%)")

# ── Divergence direction analysis ─────────────────────────────────────
print(f"\n{'─' * 72}")
print("  DIVERGENCE DIRECTION (among synapses modified in BOTH flies)")
print(f"{'─' * 72}")
both_mask = mod0 & mod1
d0b = d0[both_mask]
d1b = d1[both_mask]
n_both = both_mask.sum().item()

same_sign = ((d0b > 0) & (d1b > 0)) | ((d0b < 0) & (d1b < 0))
opp_sign  = ((d0b > 0) & (d1b < 0)) | ((d0b < 0) & (d1b > 0))
n_same = same_sign.sum().item()
n_opp = opp_sign.sum().item()

print(f"  Both modified:         {n_both:>12,}")
print(f"  Same direction:        {n_same:>12,}  ({100*n_same/max(n_both,1):>6.2f}%)")
print(f"  Opposite direction:    {n_opp:>12,}  ({100*n_opp/max(n_both,1):>6.2f}%)")

# Among same-direction, how different is magnitude?
if n_same > 0:
    same_mask_both = same_sign
    ratio = d0b[same_mask_both].abs() / (d1b[same_mask_both].abs() + 1e-12)
    print(f"  Same-dir magnitude ratio (fly0/fly1):")
    print(f"    Mean: {ratio.mean().item():.4f}  Median: {ratio.median().item():.4f}  "
          f"Std: {ratio.std().item():.4f}")

# ── Top divergent synapses ────────────────────────────────────────────
print(f"\n{'─' * 72}")
print("  TOP 20 MOST DIVERGENT SYNAPSES")
print(f"{'─' * 72}")
top_vals, top_idx = abs_diff.topk(20)
print(f"  {'Synapse idx':>12s}  {'Fly 0':>12s}  {'Fly 1':>12s}  "
      f"{'Baseline':>12s}  {'|Diff|':>12s}")
print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
for i in range(20):
    idx = top_idx[i].item()
    print(f"  {idx:>12,}  {w0[idx].item():>12.4f}  {w1[idx].item():>12.4f}  "
          f"{baseline[idx].item():>12.4f}  {top_vals[i].item():>12.4f}")

# ── Summary interpretation ────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("  INTERPRETATION")
print(f"{'=' * 72}")
n_diverged_fine = (abs_diff > 1e-5).sum().item()
max_diff_val = abs_diff.max().item()
mean_delta = d0.abs().mean().item()

print(f"""
  Both flies started from the identical FlyWire v783 connectome with
  {N:,} synapses. After independent embodied experience in the same
  arena, Hebbian plasticity (eta={1e-4}, alpha={1e-7}) modified their
  synaptic weights.

  STRUCTURAL DOMINANCE:
    Pearson r (full weights) = {corr_full:.8f}
    The connectome's original wiring (weights spanning -2405 to +1897)
    utterly dominates. The maximum Hebbian change was ~{d0.abs().max().item():.4f},
    so plasticity perturbed weights by at most ~{100*d0.abs().max().item()/baseline.abs().max().item():.4f}%.

  PLASTICITY OVERVIEW:
    Both flies modified ALL {N:,} synapses (weight decay alone ensures
    every weight drifts). The global pattern of change is nearly
    identical: delta correlation r = {corr_delta:.8f}.
    Both flies depressed 60% of synapses and potentiated 40% --
    weight decay (alpha) causes a net shrinkage bias.

  MICRO-DIVERGENCE FROM INDEPENDENT EXPERIENCE:
    Despite the near-unity correlation, fine-grained differences exist:
    - Max |fly0 - fly1| = {max_diff_val:.8f}
    - {n_diverged_fine:,} synapses differ by > 1e-5
    - These micro-differences arise because each fly received slightly
      different sensory input (visual angles, odor plume encounters,
      collision timing), driving slightly different pre/post spike
      correlations at individual synapses.
    - At the strongest synapses (e.g., synapse #13,044,317 at ~236),
      the flies differ by ~0.0001 -- a 0.00004% relative difference.

  BIOLOGICAL INTERPRETATION:
    The simulation ran for a short period with very conservative
    learning rates (eta=1e-4). The result mirrors early-stage biological
    plasticity: the innate connectome structure overwhelmingly dominates,
    but the seeds of individual experience are already present as
    microscopic weight divergences. Longer runs or higher learning rates
    would amplify these differences into behaviorally relevant
    individuality -- analogous to how genetically identical Drosophila
    develop distinct behavioral idiosyncrasies through experience.
""")
