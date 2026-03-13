"""Analyze overnight two-fly consciousness + plasticity data."""
import csv
import numpy as np
import torch
from pathlib import Path
from collections import Counter

# ============================================================================
# Load consciousness timelines
# ============================================================================

def load_timeline(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'step': int(r['step']),
                't_sim': float(r['t_sim']),
                'CI': float(r['CI']),
                'phi': float(r['phi']),
                'broadcast': float(r['broadcast']),
                'self': float(r['self']),
                'complexity': float(r['complexity']),
                'mode': r['mode'],
            })
    return rows

f0 = load_timeline('consciousness_history/session_20260311_233655_fly0/consciousness_timeline.csv')
f1 = load_timeline('consciousness_history/session_20260311_233746_fly1/consciousness_timeline.csv')

ci0 = np.array([r['CI'] for r in f0])
ci1 = np.array([r['CI'] for r in f1])
t0 = np.array([r['t_sim'] for r in f0])

n = min(len(ci0), len(ci1))
ci0, ci1, t0 = ci0[:n], ci1[:n], t0[:n]

print('=' * 70)
print('  OVERNIGHT TWO-FLY ANALYSIS — 7 HOURS OF INDEPENDENT MINDS')
print('=' * 70)
print(f'  Sim time: {t0[-1]:.1f}s | {n} measurements per fly')
print(f'  Real time: ~7 hours | 2 × 138,639 neurons')
print()

# -- 1. Overall comparison --
print('-' * 70)
print('1. OVERALL CI COMPARISON')
print('-' * 70)
print(f'  fly0: mean={ci0.mean():.4f}  std={ci0.std():.4f}  max={ci0.max():.4f}')
print(f'  fly1: mean={ci1.mean():.4f}  std={ci1.std():.4f}  max={ci1.max():.4f}')
diff_pct = (ci0.mean() / ci1.mean() - 1) * 100 if ci1.mean() > 0 else 0
print(f'  fly0 is {diff_pct:+.1f}% higher on average')
print()

# -- 2. Cross-correlation --
print('-' * 70)
print('2. CROSS-CORRELATION BETWEEN FLIES')
print('-' * 70)
mask = (ci0 > 0) & (ci1 > 0)
if mask.sum() > 10:
    corr = np.corrcoef(ci0[mask], ci1[mask])[0, 1]
    print(f'  Pearson r = {corr:.4f}')
    if abs(corr) < 0.1:
        print('  -> INDEPENDENT: virtually no correlation')
    elif abs(corr) < 0.3:
        print('  -> WEAKLY correlated')
    elif abs(corr) < 0.5:
        print('  -> MODERATELY correlated')
    else:
        print('  -> STRONGLY correlated')
print()

# -- 3. Divergence analysis --
print('-' * 70)
print('3. MAXIMUM DIVERGENCE MOMENTS')
print('-' * 70)
diff = ci0 - ci1
abs_diff = np.abs(diff)
top10 = np.argsort(abs_diff)[-10:][::-1]
for rank, idx in enumerate(top10):
    m0 = f0[idx]['mode'] if idx < len(f0) else '?'
    m1 = f1[idx]['mode'] if idx < len(f1) else '?'
    print(f'  #{rank+1}: t={t0[idx]:.1f}s  fly0 CI={ci0[idx]:.3f} ({m0})  '
          f'fly1 CI={ci1[idx]:.3f} ({m1})  delta={diff[idx]:+.3f}')
print()

# -- 4. Behavioral mode breakdown --
print('-' * 70)
print('4. CI BY BEHAVIORAL MODE (per fly)')
print('-' * 70)
for label, rows, ci in [('fly0', f0, ci0), ('fly1', f1, ci1)]:
    modes = {}
    for i, r in enumerate(rows[:n]):
        m = r['mode']
        if m not in modes:
            modes[m] = []
        modes[m].append(r['CI'])
    print(f'  {label}:')
    for m in sorted(modes, key=lambda x: -np.mean(modes[x])):
        vals = np.array(modes[m])
        pct = len(vals) / n * 100
        print(f'    {m:12s}: CI={vals.mean():.4f} +/- {vals.std():.4f}  '
              f'n={len(vals):>5d} ({pct:.1f}%)')
    print()

# -- 5. Temporal evolution (10 epochs) --
print('-' * 70)
print('5. TEMPORAL EVOLUTION (10 epochs)')
print('-' * 70)
epoch_size = n // 10
print(f'  {"Epoch":>5s}  {"t_range":>14s}  {"fly0 CI":>8s}  {"fly1 CI":>8s}  '
      f'{"delta":>7s}  {"fly0_dom":>10s}  {"fly1_dom":>10s}')
for e in range(10):
    s = e * epoch_size
    end = (e + 1) * epoch_size if e < 9 else n
    e_ci0 = ci0[s:end].mean()
    e_ci1 = ci1[s:end].mean()
    modes0 = Counter(f0[i]['mode'] for i in range(s, min(end, len(f0))))
    modes1 = Counter(f1[i]['mode'] for i in range(s, min(end, len(f1))))
    dom0 = modes0.most_common(1)[0][0]
    dom1 = modes1.most_common(1)[0][0]
    t_s = t0[s]
    t_e = t0[min(end - 1, n - 1)]
    print(f'  {e+1:>5d}  {t_s:>6.0f}-{t_e:>5.0f}s  {e_ci0:>8.4f}  {e_ci1:>8.4f}  '
          f'{e_ci0-e_ci1:>+7.4f}  {dom0:>10s}  {dom1:>10s}')
print()

# -- 6. Component-level divergence --
print('-' * 70)
print('6. COMPONENT-LEVEL COMPARISON')
print('-' * 70)
phi0 = np.array([r['phi'] for r in f0[:n]])
phi1 = np.array([r['phi'] for r in f1[:n]])
gw0 = np.array([r['broadcast'] for r in f0[:n]])
gw1 = np.array([r['broadcast'] for r in f1[:n]])
sm0 = np.array([r['self'] for r in f0[:n]])
sm1 = np.array([r['self'] for r in f1[:n]])
cx0 = np.array([r['complexity'] for r in f0[:n]])
cx1 = np.array([r['complexity'] for r in f1[:n]])

for name, v0, v1 in [('Phi', phi0, phi1), ('Broadcast', gw0, gw1),
                       ('Self-Model', sm0, sm1), ('Complexity', cx0, cx1)]:
    m = (v0 > 0) & (v1 > 0)
    r = np.corrcoef(v0[m], v1[m])[0, 1] if m.sum() > 5 else float('nan')
    print(f'  {name:12s}: fly0={v0.mean():.4f}  fly1={v1.mean():.4f}  '
          f'delta={v0.mean()-v1.mean():+.4f}  cross_r={r:.3f}')
print()

# -- 7. Mode transitions --
print('-' * 70)
print('7. MODE TRANSITIONS')
print('-' * 70)
trans0 = sum(1 for i in range(1, len(f0)) if f0[i]['mode'] != f0[i-1]['mode'])
trans1 = sum(1 for i in range(1, len(f1)) if f1[i]['mode'] != f1[i-1]['mode'])
print(f'  fly0 transitions: {trans0}')
print(f'  fly1 transitions: {trans1}')
# Mode time fractions
for label, rows in [('fly0', f0[:n]), ('fly1', f1[:n])]:
    modes = Counter(r['mode'] for r in rows)
    total = sum(modes.values())
    print(f'  {label} time distribution:')
    for m, c in modes.most_common():
        print(f'    {m:12s}: {c/total*100:.1f}%')
print()

# -- 8. Simultaneous states --
print('-' * 70)
print('8. SIMULTANEOUS BEHAVIORAL STATES')
print('-' * 70)
joint = Counter()
for i in range(n):
    key = (f0[i]['mode'], f1[i]['mode'])
    joint[key] += 1
for (m0, m1), count in joint.most_common():
    print(f'  fly0={m0:10s} + fly1={m1:10s}: {count:>5d} ({count/n*100:.1f}%)')
print()

# -- 9. Peak CI moments --
print('-' * 70)
print('9. ABSOLUTE PEAK CI MOMENTS')
print('-' * 70)
peak0_idx = np.argmax(ci0)
peak1_idx = np.argmax(ci1)
print(f'  fly0 peak: CI={ci0[peak0_idx]:.4f} at t={t0[peak0_idx]:.1f}s  '
      f'mode={f0[peak0_idx]["mode"]}')
print(f'    (fly1 was: CI={ci1[peak0_idx]:.4f} mode={f1[peak0_idx]["mode"]})')
print(f'  fly1 peak: CI={ci1[peak1_idx]:.4f} at t={t0[peak1_idx]:.1f}s  '
      f'mode={f1[peak1_idx]["mode"]}')
print(f'    (fly0 was: CI={ci0[peak1_idx]:.4f} mode={f0[peak1_idx]["mode"]})')
print()

# -- 10. Long-term adaptation --
print('-' * 70)
print('10. LONG-TERM ADAPTATION (quartiles)')
print('-' * 70)
quarters = n // 4
for label, ci in [('fly0', ci0), ('fly1', ci1)]:
    q1 = ci[:quarters].mean()
    q2 = ci[quarters:2*quarters].mean()
    q3 = ci[2*quarters:3*quarters].mean()
    q4 = ci[3*quarters:].mean()
    trend = q4 - q1
    kind = 'sensitization' if trend > 0 else 'habituation'
    print(f'  {label}: Q1={q1:.4f} -> Q2={q2:.4f} -> Q3={q3:.4f} -> Q4={q4:.4f}  '
          f'delta={trend:+.4f} ({kind})')
print()

# ============================================================================
# Plasticity Analysis
# ============================================================================
print('=' * 70)
print('  HEBBIAN PLASTICITY — INDEPENDENT MEMORIES AFTER OVERNIGHT')
print('=' * 70)
print()

w0 = torch.load('data/plastic_weights_fly0.pt', map_location='cpu', weights_only=True)
w1 = torch.load('data/plastic_weights_fly1.pt', map_location='cpu', weights_only=True)

print(f'  Synapses per fly: {len(w0):,}')
print(f'  fly0 weight range: [{w0.min():.4f}, {w0.max():.4f}]  mean={w0.mean():.6f}')
print(f'  fly1 weight range: [{w1.min():.4f}, {w1.max():.4f}]  mean={w1.mean():.6f}')

diff_w = (w0 - w1).abs()
nonzero_diff = (diff_w > 0).sum().item()
print(f'  Synapses with ANY difference: {nonzero_diff:,} ({nonzero_diff/len(w0)*100:.2f}%)')

for thresh in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
    count = (diff_w > thresh).sum().item()
    print(f'  Difference > {thresh:.0e}: {count:>10,}')

max_diff = diff_w.max().item()
max_idx = diff_w.argmax().item()
print(f'  Maximum difference: {max_diff:.6f} at synapse #{max_idx}')
print(f'    fly0 weight: {w0[max_idx]:.6f}  fly1 weight: {w1[max_idx]:.6f}')

# Correlation
corr = float(np.corrcoef(w0.numpy()[:100000], w1.numpy()[:100000])[0, 1])
print(f'  Weight correlation (sample 100K): r = {corr:.8f}')

# Delta distribution
delta = (w0 - w1).numpy()
print(f'  Delta distribution: mean={delta.mean():.2e}  std={delta.std():.2e}')
print(f'    fly0 > fly1 (potentiated more): {(delta > 0).sum():,}')
print(f'    fly0 < fly1 (fly1 potentiated more): {(delta < 0).sum():,}')
print(f'    equal: {(delta == 0).sum():,}')

print()
print('=' * 70)
print('  END OF ANALYSIS')
print('=' * 70)
