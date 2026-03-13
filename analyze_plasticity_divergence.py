#!/usr/bin/env python3
"""
Deep Analysis of Hebbian Plasticity Divergence Between Two Flies
================================================================
Loads plastic_weights_fly0.pt and plastic_weights_fly1.pt, maps divergent
synapses to brain regions via FlyWire annotations, and produces a
comprehensive divergence report.

Author: Enrique Manuel Rojas Aliaga
"""

import sys
import os
import math
import torch
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K = 200          # number of most-divergent synapses to analyze
DATA_DIR = Path(__file__).resolve().parent / "data"

WEIGHTS_FLY0 = DATA_DIR / "plastic_weights_fly0.pt"
WEIGHTS_FLY1 = DATA_DIR / "plastic_weights_fly1.pt"
ANNOTATIONS = DATA_DIR / "flywire_annotations.tsv"

# Functional groupings of brain regions
FUNCTIONAL_GROUPS = {
    "visual": [
        "ME(R)", "ME(L)", "LO(R)", "LO(L)", "LOP(R)", "LOP(L)",
        "AME(R)", "AME(L)", "ME", "LO", "LOP", "AME",
        "AOTU(R)", "AOTU(L)", "AOTU",
        "optic", "visual",
    ],
    "olfactory": [
        "AL(R)", "AL(L)", "AL",
        "MB(R)", "MB(L)", "MB",
        "CA(R)", "CA(L)", "CA",
        "LH(R)", "LH(L)", "LH",
        "antennal_lobe", "mushroom_body",
    ],
    "motor": [
        "VNC", "GNG", "PRW", "PENP",
        "CRE(R)", "CRE(L)", "CRE",
        "IPS(R)", "IPS(L)", "IPS",
        "SPS(R)", "SPS(L)", "SPS",
        "WED(R)", "WED(L)", "WED",
        "descending", "motor",
    ],
    "integrator": [
        "FB", "EB", "PB", "NO",
        "CX", "central_complex",
        "LAL(R)", "LAL(L)", "LAL",
        "AVLP(R)", "AVLP(L)", "AVLP",
        "PVLP(R)", "PVLP(L)", "PVLP",
        "PLP(R)", "PLP(L)", "PLP",
        "SCL(R)", "SCL(L)", "SCL",
        "SLP(R)", "SLP(L)", "SLP",
        "SIP(R)", "SIP(L)", "SIP",
        "SMP(R)", "SMP(L)", "SMP",
        "CRE", "ICL(R)", "ICL(L)", "ICL",
        "IB(R)", "IB(L)", "IB",
        "ATL(R)", "ATL(L)", "ATL",
        "lateral_horn", "superior",
    ],
}


def classify_region(region_str: str) -> str:
    """Classify a brain region string into a functional group."""
    if not region_str or region_str == "unknown":
        return "unknown"
    r = region_str.lower()
    for group, keywords in FUNCTIONAL_GROUPS.items():
        for kw in keywords:
            if kw.lower() in r:
                return group
    return "other"


# ---------------------------------------------------------------------------
# 1. Load weights
# ---------------------------------------------------------------------------
print("=" * 80)
print("HEBBIAN PLASTICITY DIVERGENCE ANALYSIS")
print("=" * 80)
print()

print("[1] Loading plastic weight tensors ...")
w0_data = torch.load(str(WEIGHTS_FLY0), map_location="cpu", weights_only=False)
w1_data = torch.load(str(WEIGHTS_FLY1), map_location="cpu", weights_only=False)

# Handle different storage formats (dict with metadata vs raw tensor)
def extract_weights(data, label):
    """Extract sparse weight tensor from saved data (handles dict or raw tensor)."""
    if isinstance(data, dict):
        print(f"  {label}: dict with keys {list(data.keys())}")
        # Common key names
        for key in ["weights", "plastic_weights", "W", "w", "weight", "state_dict"]:
            if key in data:
                w = data[key]
                if isinstance(w, dict):
                    # state_dict style
                    for k2, v2 in w.items():
                        if isinstance(v2, torch.Tensor):
                            return v2
                if isinstance(w, torch.Tensor):
                    return w
        # Try first tensor value
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                print(f"  Using key '{k}' -> shape {v.shape}")
                return v
        raise ValueError(f"No tensor found in {label} dict")
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError(f"Unexpected type for {label}: {type(data)}")

W0 = extract_weights(w0_data, "fly0")
W1 = extract_weights(w1_data, "fly1")

# Convert sparse to dense if needed, or keep sparse
if W0.is_sparse:
    print(f"  fly0 sparse tensor: {W0.shape}, nnz={W0._nnz()}")
    W0_dense = W0.to_dense()
else:
    print(f"  fly0 dense tensor: {W0.shape}")
    W0_dense = W0

if W1.is_sparse:
    print(f"  fly1 sparse tensor: {W1.shape}, nnz={W1._nnz()}")
    W1_dense = W1.to_dense()
else:
    print(f"  fly1 dense tensor: {W1.shape}")
    W1_dense = W1

assert W0_dense.shape == W1_dense.shape, (
    f"Shape mismatch: fly0={W0_dense.shape}, fly1={W1_dense.shape}"
)

N = W0_dense.shape[0]
print(f"  Weight matrix size: {N} x {N}  ({N} neurons)")
print()

# ---------------------------------------------------------------------------
# 2. Load annotations & build neuron → region mapping
# ---------------------------------------------------------------------------
print("[2] Loading FlyWire annotations ...")
ann = pd.read_csv(str(ANNOTATIONS), sep="\t", low_memory=False)
print(f"  Annotations: {len(ann)} rows, columns: {list(ann.columns)}")

# Try to import flyid2i mapping from brain_body_bridge
flyid2i = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from brain_body_bridge import BrainEngine
    # BrainEngine might store flyid2i as class or instance attribute
    # Try various access patterns
    if hasattr(BrainEngine, 'flyid2i'):
        flyid2i = BrainEngine.flyid2i
        print(f"  Loaded flyid2i from BrainEngine class attr ({len(flyid2i)} entries)")
    else:
        # It might be built during __init__; check for class-level data
        print("  flyid2i not a class attribute; searching module ...")
        import brain_body_bridge as bbb
        for name in dir(bbb):
            obj = getattr(bbb, name)
            if isinstance(obj, dict) and len(obj) > 1000:
                # Likely the mapping
                sample_key = next(iter(obj))
                if isinstance(sample_key, (int, np.integer)):
                    flyid2i = obj
                    print(f"  Found mapping '{name}' with {len(flyid2i)} entries")
                    break
except Exception as e:
    print(f"  Could not import BrainEngine: {e}")

# Build index → root_id reverse mapping
i2flyid = None
if flyid2i is not None:
    i2flyid = {v: k for k, v in flyid2i.items()}
    print(f"  Built reverse mapping i2flyid with {len(i2flyid)} entries")

# Determine the region column in annotations
region_col = None
for candidate in ["super_class", "cell_class", "flow", "hemibrain_type", "neuropil", "cell_type"]:
    if candidate in ann.columns:
        region_col = candidate
        break

# Build root_id → region mapping
root_col = None
for candidate in ["root_id", "rootid", "id", "root"]:
    if candidate in ann.columns:
        root_col = candidate
        break

print(f"  Using root_id column: '{root_col}', region column: '{region_col}'")

# Also look for a more granular neuropil/region column
region_detail_col = None
for candidate in ["neuropil", "cell_class", "hemibrain_type"]:
    if candidate in ann.columns and candidate != region_col:
        region_detail_col = candidate
        break

rootid_to_region = {}
rootid_to_superclass = {}
rootid_to_flow = {}
if root_col and region_col:
    for _, row in ann.iterrows():
        rid = row[root_col]
        rootid_to_region[rid] = str(row.get(region_col, "unknown"))
        if "super_class" in ann.columns:
            rootid_to_superclass[rid] = str(row.get("super_class", "unknown"))
        if "flow" in ann.columns:
            rootid_to_flow[rid] = str(row.get("flow", "unknown"))
        if region_detail_col:
            pass  # could store detailed region too

print(f"  Mapped {len(rootid_to_region)} root_ids to regions")

# Build index → region mapping
def get_region(idx):
    """Get brain region for neuron index."""
    if i2flyid is not None and idx in i2flyid:
        rid = i2flyid[idx]
        return rootid_to_region.get(rid, "unmapped")
    return "unmapped"

def get_superclass(idx):
    if i2flyid is not None and idx in i2flyid:
        rid = i2flyid[idx]
        return rootid_to_superclass.get(rid, "unmapped")
    return "unmapped"

def get_flow(idx):
    if i2flyid is not None and idx in i2flyid:
        rid = i2flyid[idx]
        return rootid_to_flow.get(rid, "unmapped")
    return "unmapped"

# If we couldn't get flyid2i, try to build a positional mapping
# from the annotation order (first N entries = first N neuron indices)
if i2flyid is None and root_col:
    print("  WARNING: No flyid2i mapping found. Using positional heuristic "
          "(annotation row order = neuron index).")
    root_ids = ann[root_col].values
    i2flyid_positional = {i: int(root_ids[i]) for i in range(min(N, len(root_ids)))}
    # Override get_region
    def get_region(idx):
        if idx < len(root_ids):
            rid = int(root_ids[idx])
            return rootid_to_region.get(rid, "unmapped")
        return "unmapped"
    def get_superclass(idx):
        if idx < len(root_ids):
            rid = int(root_ids[idx])
            return rootid_to_superclass.get(rid, "unmapped")
        return "unmapped"
    def get_flow(idx):
        if idx < len(root_ids):
            rid = int(root_ids[idx])
            return rootid_to_flow.get(rid, "unmapped")
        return "unmapped"

print()

# ---------------------------------------------------------------------------
# 3. Compute divergence
# ---------------------------------------------------------------------------
print("[3] Computing synapse-level divergence (|W0 - W1|) ...")
diff = (W0_dense - W1_dense).abs()

# Total stats
nonzero_mask = diff > 0
total_divergent = nonzero_mask.sum().item()
mean_div = diff[nonzero_mask].mean().item() if total_divergent > 0 else 0
max_div = diff.max().item()
std_div = diff[nonzero_mask].std().item() if total_divergent > 1 else 0

print(f"  Total synapses with divergence > 0:  {total_divergent:,}")
print(f"  Mean divergence (non-zero):          {mean_div:.6f}")
print(f"  Std divergence (non-zero):           {std_div:.6f}")
print(f"  Max divergence:                      {max_div:.6f}")

# Also compute per-fly stats
w0_nnz = (W0_dense.abs() > 0).sum().item()
w1_nnz = (W1_dense.abs() > 0).sum().item()
w0_mean = W0_dense[W0_dense.abs() > 0].mean().item() if w0_nnz > 0 else 0
w1_mean = W1_dense[W1_dense.abs() > 0].mean().item() if w1_nnz > 0 else 0
print(f"  fly0: {w0_nnz:,} non-zero weights, mean={w0_mean:.6f}")
print(f"  fly1: {w1_nnz:,} non-zero weights, mean={w1_mean:.6f}")
print()

# ---------------------------------------------------------------------------
# 4. Top-K most divergent synapses
# ---------------------------------------------------------------------------
print(f"[4] Finding top {TOP_K} most divergent synapses ...")

# Flatten, get topk
diff_flat = diff.view(-1)
topk_vals, topk_idx = torch.topk(diff_flat, min(TOP_K, diff_flat.numel()))

# Convert flat indices to (pre, post) pairs
pre_indices = (topk_idx // N).tolist()
post_indices = (topk_idx % N).tolist()
topk_vals = topk_vals.tolist()

print(f"  Top divergence range: {topk_vals[0]:.6f} ... {topk_vals[-1]:.6f}")
print()

# Build detailed table
print(f"[5] Mapping top {TOP_K} divergent synapses to brain regions ...")
print()

synapse_records = []
for rank, (pre_i, post_i, dval) in enumerate(zip(pre_indices, post_indices, topk_vals)):
    pre_region = get_region(pre_i)
    post_region = get_region(post_i)
    pre_flow = get_flow(pre_i)
    post_flow = get_flow(post_i)
    w0_val = W0_dense[pre_i, post_i].item()
    w1_val = W1_dense[pre_i, post_i].item()
    synapse_records.append({
        "rank": rank + 1,
        "pre_idx": pre_i,
        "post_idx": post_i,
        "pre_region": pre_region,
        "post_region": post_region,
        "pre_flow": pre_flow,
        "post_flow": post_flow,
        "w0": w0_val,
        "w1": w1_val,
        "divergence": dval,
        "direction": "fly0>fly1" if w0_val > w1_val else "fly1>fly0",
    })

df_top = pd.DataFrame(synapse_records)

# Print top 30
print("  --- Top 30 Most Divergent Synapses ---")
print(f"  {'Rank':>4}  {'Pre Region':<20} {'Post Region':<20} "
      f"{'W0':>10} {'W1':>10} {'|Diff|':>10}  Direction")
print("  " + "-" * 100)
for _, row in df_top.head(30).iterrows():
    print(f"  {row['rank']:4d}  {row['pre_region']:<20} {row['post_region']:<20} "
          f"{row['w0']:10.6f} {row['w1']:10.6f} {row['divergence']:10.6f}  {row['direction']}")
print()

# ---------------------------------------------------------------------------
# 6. Divergence map: region-pair → total divergence
# ---------------------------------------------------------------------------
print("[6] Building brain-region-pair divergence map ...")
print()

region_pair_div = defaultdict(float)
region_pair_count = defaultdict(int)

for rec in synapse_records:
    pair = (rec["pre_region"], rec["post_region"])
    region_pair_div[pair] += rec["divergence"]
    region_pair_count[pair] += 1

# Sort by total divergence
sorted_pairs = sorted(region_pair_div.items(), key=lambda x: -x[1])

print("  --- Top 30 Region Pairs by Cumulative Divergence (Top-200 synapses) ---")
print(f"  {'Pre Region':<22} {'Post Region':<22} {'Total Div':>12} {'Count':>6} {'Avg Div':>10}")
print("  " + "-" * 76)
for (pre_r, post_r), total_div in sorted_pairs[:30]:
    cnt = region_pair_count[(pre_r, post_r)]
    avg = total_div / cnt
    print(f"  {pre_r:<22} {post_r:<22} {total_div:12.6f} {cnt:6d} {avg:10.6f}")
print()

# ---------------------------------------------------------------------------
# 7. Functional group analysis
# ---------------------------------------------------------------------------
print("[7] Divergence by Functional Category (visual / motor / olfactory / integrator) ...")
print()

# Classify each top synapse's pre and post region
func_pair_div = defaultdict(float)
func_pair_count = defaultdict(int)
func_involvement = defaultdict(float)  # how much divergence each func group is involved in

for rec in synapse_records:
    pre_func = classify_region(rec["pre_region"])
    post_func = classify_region(rec["post_region"])
    pair_key = (pre_func, post_func)
    func_pair_div[pair_key] += rec["divergence"]
    func_pair_count[pair_key] += 1
    func_involvement[pre_func] += rec["divergence"]
    func_involvement[post_func] += rec["divergence"]

print("  --- Functional Group Pair Divergence (Top-200 synapses) ---")
print(f"  {'Pre Function':<15} {'Post Function':<15} {'Total Div':>12} {'Count':>6} {'Avg Div':>10}")
print("  " + "-" * 62)
sorted_func_pairs = sorted(func_pair_div.items(), key=lambda x: -x[1])
for (pf, qf), td in sorted_func_pairs:
    cnt = func_pair_count[(pf, qf)]
    print(f"  {pf:<15} {qf:<15} {td:12.6f} {cnt:6d} {td/cnt:10.6f}")
print()

total_div_all = sum(func_involvement.values()) / 2  # each synapse counted twice
print("  --- Functional Group Total Involvement ---")
print(f"  {'Group':<15} {'Total Div':>12} {'Share':>8}")
print("  " + "-" * 38)
for grp in sorted(func_involvement, key=lambda g: -func_involvement[g]):
    share = func_involvement[grp] / (sum(func_involvement.values())) * 100
    print(f"  {grp:<15} {func_involvement[grp]:12.6f} {share:7.1f}%")
print()

# ---------------------------------------------------------------------------
# 8. Per-region plasticity magnitude for each fly
# ---------------------------------------------------------------------------
print("[8] Total plasticity magnitude per brain region per fly ...")
print("    (Sum of |weight change from baseline| for all synapses involving each region)")
print()

# We compute per-region magnitude across ALL synapses, not just top-200
# For efficiency on large matrices, use row/column sums of absolute values
w0_abs = W0_dense.abs()
w1_abs = W1_dense.abs()

# Row sums = outgoing plasticity per neuron (as pre-synaptic)
w0_row_sum = w0_abs.sum(dim=1).numpy()  # shape (N,)
w1_row_sum = w1_abs.sum(dim=1).numpy()
# Col sums = incoming plasticity per neuron (as post-synaptic)
w0_col_sum = w0_abs.sum(dim=0).numpy()
w1_col_sum = w1_abs.sum(dim=0).numpy()

# Total plasticity per neuron = outgoing + incoming
w0_total = w0_row_sum + w0_col_sum
w1_total = w1_row_sum + w1_col_sum

# Aggregate by brain region
region_plasticity_fly0 = defaultdict(float)
region_plasticity_fly1 = defaultdict(float)
region_neuron_count = defaultdict(int)

for i in range(N):
    reg = get_region(i)
    region_plasticity_fly0[reg] += w0_total[i]
    region_plasticity_fly1[reg] += w1_total[i]
    region_neuron_count[reg] += 1

all_regions = sorted(region_neuron_count.keys(),
                     key=lambda r: -(region_plasticity_fly0[r] + region_plasticity_fly1[r]))

print(f"  {'Region':<25} {'Neurons':>7} {'Fly0 Mag':>12} {'Fly1 Mag':>12} {'Diff':>12} {'Ratio':>8}")
print("  " + "-" * 80)
for reg in all_regions[:40]:
    f0 = region_plasticity_fly0[reg]
    f1 = region_plasticity_fly1[reg]
    diff_val = f0 - f1
    ratio = f0 / f1 if f1 > 0 else float('inf')
    nc = region_neuron_count[reg]
    print(f"  {reg:<25} {nc:7d} {f0:12.2f} {f1:12.2f} {diff_val:12.2f} {ratio:8.3f}")
print()

# Per-neuron average
print("  --- Per-Neuron Average Plasticity by Region (top 30) ---")
print(f"  {'Region':<25} {'Fly0/neuron':>12} {'Fly1/neuron':>12} {'Diff/neuron':>12}")
print("  " + "-" * 65)
region_avg = []
for reg in all_regions:
    nc = region_neuron_count[reg]
    if nc > 0:
        f0_avg = region_plasticity_fly0[reg] / nc
        f1_avg = region_plasticity_fly1[reg] / nc
        region_avg.append((reg, f0_avg, f1_avg, f0_avg - f1_avg))
region_avg.sort(key=lambda x: -abs(x[3]))
for reg, f0a, f1a, da in region_avg[:30]:
    print(f"  {reg:<25} {f0a:12.4f} {f1a:12.4f} {da:12.4f}")
print()

# ---------------------------------------------------------------------------
# 9. Functional group summary
# ---------------------------------------------------------------------------
print("[9] Functional Group Plasticity Summary ...")
print()
func_plast_fly0 = defaultdict(float)
func_plast_fly1 = defaultdict(float)
func_neuron_count = defaultdict(int)

for reg in all_regions:
    fg = classify_region(reg)
    func_plast_fly0[fg] += region_plasticity_fly0[reg]
    func_plast_fly1[fg] += region_plasticity_fly1[reg]
    func_neuron_count[fg] += region_neuron_count[reg]

print(f"  {'Func Group':<15} {'Neurons':>7} {'Fly0 Total':>14} {'Fly1 Total':>14} "
      f"{'Diff':>14} {'Fly0/n':>10} {'Fly1/n':>10}")
print("  " + "-" * 90)
for fg in sorted(func_plast_fly0, key=lambda g: -(func_plast_fly0[g] + func_plast_fly1[g])):
    nc = func_neuron_count[fg]
    f0 = func_plast_fly0[fg]
    f1 = func_plast_fly1[fg]
    f0n = f0 / nc if nc > 0 else 0
    f1n = f1 / nc if nc > 0 else 0
    print(f"  {fg:<15} {nc:7d} {f0:14.2f} {f1:14.2f} {f0-f1:14.2f} {f0n:10.4f} {f1n:10.4f}")
print()

# ---------------------------------------------------------------------------
# 10. Correlation with consciousness metrics
# ---------------------------------------------------------------------------
print("[10] Correlation between regional plasticity & consciousness proxies ...")
print()

# Load consciousness history if available
consciousness_data = None
consciousness_dir = Path(__file__).resolve().parent / "consciousness_history"
if consciousness_dir.exists():
    session_dirs = sorted(consciousness_dir.iterdir())
    if session_dirs:
        latest = session_dirs[-1]
        for f in latest.iterdir():
            if f.suffix in ('.csv', '.json', '.pt', '.npz'):
                print(f"  Found consciousness data: {f.name}")
                if f.suffix == '.csv':
                    consciousness_data = pd.read_csv(f)
                elif f.suffix == '.json':
                    import json
                    with open(f) as fh:
                        consciousness_data = json.load(fh)
                elif f.suffix == '.pt':
                    consciousness_data = torch.load(str(f), map_location="cpu", weights_only=False)
                elif f.suffix == '.npz':
                    consciousness_data = dict(np.load(str(f), allow_pickle=True))
                break

# Compute region-level "activity proxy" = total weight change magnitude
# We use |W_fly - baseline| as proxy for how much learning/activity occurred
# The divergence itself (|W0 - W1|) is another proxy for experiential difference

# For consciousness correlation, we look at whether regions with more plasticity
# also show higher integration (a la IIT) -- approximated here by how much a
# region's plastic weights connect to OTHER regions vs. within itself.

print("  Computing plasticity integration metrics per region ...")

# For each region, compute:
#   intra_plasticity = sum of |weights| within the region
#   inter_plasticity = sum of |weights| connecting to/from other regions
#   integration_ratio = inter / (intra + inter)

region_neurons = defaultdict(list)
for i in range(N):
    reg = get_region(i)
    region_neurons[reg].append(i)

# We'll use fly0 and fly1 averages for the integration analysis
W_avg = ((W0_dense.abs() + W1_dense.abs()) / 2)

# Due to potentially large N, compute integration for top regions only
top_regions = [r for r in all_regions if region_neuron_count[r] >= 5][:25]

print()
print("  --- Regional Plasticity Integration Analysis ---")
print(f"  {'Region':<25} {'Intra':>10} {'Inter':>10} {'Integ Ratio':>12} "
      f"{'Mean Div':>10} {'Consciousness':>14}")
print("  " + "-" * 85)

region_integration = {}
region_mean_divergence = {}

# Compute mean divergence per region from top-200 synapses
region_div_accum = defaultdict(list)
for rec in synapse_records:
    region_div_accum[rec["pre_region"]].append(rec["divergence"])
    region_div_accum[rec["post_region"]].append(rec["divergence"])

for reg in top_regions:
    neurons = region_neurons[reg]
    neuron_set = set(neurons)
    n_idx = torch.tensor(neurons, dtype=torch.long)

    if len(neurons) == 0:
        continue

    # Submatrix for this region (outgoing)
    region_rows = W_avg[n_idx, :]  # shape (len(neurons), N)

    intra = 0.0
    inter = 0.0
    for local_i, global_i in enumerate(neurons):
        row = region_rows[local_i]
        for j in range(N):
            val = row[j].item()
            if val > 0:
                if j in neuron_set:
                    intra += val
                else:
                    inter += val

    # This loop could be slow for large N; use vectorized approach
    # Actually let's use a mask approach
    # Recompute with masks for efficiency
    if len(neurons) > 0 and N <= 200000:
        mask = torch.zeros(N, dtype=torch.bool)
        mask[n_idx] = True

        outgoing = W_avg[n_idx, :]  # (|reg|, N)
        incoming = W_avg[:, n_idx]  # (N, |reg|)

        # Intra = outgoing to neurons in same region
        intra = outgoing[:, mask].sum().item()
        # Inter = outgoing to neurons NOT in region + incoming from neurons NOT in region
        inter_out = outgoing[:, ~mask].sum().item()
        inter_in = incoming[~mask, :].sum().item()
        inter = inter_out + inter_in
    else:
        intra = 0.0
        inter = 1.0

    total_connect = intra + inter
    integration = inter / total_connect if total_connect > 0 else 0

    region_integration[reg] = integration

    # Mean divergence for this region in top-200
    divs = region_div_accum.get(reg, [])
    mean_d = np.mean(divs) if divs else 0
    region_mean_divergence[reg] = mean_d

    # Consciousness proxy = integration * plasticity_magnitude
    consciousness_proxy = integration * (region_plasticity_fly0[reg] + region_plasticity_fly1[reg]) / 2

    print(f"  {reg:<25} {intra:10.2f} {inter:10.2f} {integration:12.4f} "
          f"{mean_d:10.6f} {consciousness_proxy:14.2f}")

print()

# Compute Pearson correlation between integration ratio and mean divergence
if len(region_integration) > 3:
    integ_vals = []
    div_vals = []
    plast_vals = []
    for reg in region_integration:
        integ_vals.append(region_integration[reg])
        div_vals.append(region_mean_divergence.get(reg, 0))
        plast_vals.append((region_plasticity_fly0[reg] + region_plasticity_fly1[reg]) / 2)

    integ_arr = np.array(integ_vals)
    div_arr = np.array(div_vals)
    plast_arr = np.array(plast_vals)

    def pearson_r(x, y):
        if len(x) < 3:
            return 0.0, 1.0
        xm = x - x.mean()
        ym = y - y.mean()
        r_num = np.sum(xm * ym)
        r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
        if r_den < 1e-12:
            return 0.0, 1.0
        r = r_num / r_den
        # t-test for significance
        n = len(x)
        t = r * math.sqrt((n - 2) / (1 - r**2 + 1e-12))
        return r, t

    r_integ_div, t_integ_div = pearson_r(integ_arr, div_arr)
    r_plast_div, t_plast_div = pearson_r(plast_arr, div_arr)
    r_integ_plast, t_integ_plast = pearson_r(integ_arr, plast_arr)

    print("  --- Correlation Summary ---")
    print(f"  Integration vs. Divergence:   r = {r_integ_div:+.4f}  (t = {t_integ_div:+.3f})")
    print(f"  Plasticity vs. Divergence:    r = {r_plast_div:+.4f}  (t = {t_plast_div:+.3f})")
    print(f"  Integration vs. Plasticity:   r = {r_integ_plast:+.4f}  (t = {t_integ_plast:+.3f})")
    print()

    if abs(r_integ_div) > 0.5:
        print(f"  ** Strong correlation between integration and divergence (r={r_integ_div:+.4f}).")
        print(f"     Regions with higher inter-region connectivity show "
              f"{'MORE' if r_integ_div > 0 else 'LESS'} divergent plasticity.")
    elif abs(r_integ_div) > 0.3:
        print(f"  * Moderate correlation between integration and divergence (r={r_integ_div:+.4f}).")
    else:
        print(f"  No strong correlation between integration and divergence (r={r_integ_div:+.4f}).")

    if abs(r_plast_div) > 0.5:
        print(f"  ** Strong correlation between total plasticity and divergence (r={r_plast_div:+.4f}).")
        print(f"     Regions that are more plastic overall show "
              f"{'MORE' if r_plast_div > 0 else 'LESS'} inter-fly divergence.")
    print()

# ---------------------------------------------------------------------------
# 11. Directionality analysis
# ---------------------------------------------------------------------------
print("[11] Directionality: Which fly strengthened which pathways more? ...")
print()

fly0_stronger = sum(1 for r in synapse_records if r["direction"] == "fly0>fly1")
fly1_stronger = sum(1 for r in synapse_records if r["direction"] == "fly1>fly0")

print(f"  In top-{TOP_K} divergent synapses:")
print(f"    fly0 has stronger weight: {fly0_stronger} ({100*fly0_stronger/TOP_K:.1f}%)")
print(f"    fly1 has stronger weight: {fly1_stronger} ({100*fly1_stronger/TOP_K:.1f}%)")
print()

# Breakdown by functional group
fly0_func = defaultdict(int)
fly1_func = defaultdict(int)
for rec in synapse_records:
    pre_func = classify_region(rec["pre_region"])
    post_func = classify_region(rec["post_region"])
    pathway = f"{pre_func}->{post_func}"
    if rec["direction"] == "fly0>fly1":
        fly0_func[pathway] += 1
    else:
        fly1_func[pathway] += 1

all_pathways = set(fly0_func.keys()) | set(fly1_func.keys())
print("  --- Directional Breakdown by Pathway ---")
print(f"  {'Pathway':<30} {'fly0>fly1':>10} {'fly1>fly0':>10} {'Bias':>8}")
print("  " + "-" * 62)
for pw in sorted(all_pathways, key=lambda p: -(fly0_func[p] + fly1_func[p])):
    f0 = fly0_func[pw]
    f1 = fly1_func[pw]
    bias = "fly0" if f0 > f1 else ("fly1" if f1 > f0 else "neutral")
    print(f"  {pw:<30} {f0:10d} {f1:10d} {bias:>8}")
print()

# ---------------------------------------------------------------------------
# 12. Summary & interpretation
# ---------------------------------------------------------------------------
print("=" * 80)
print("SUMMARY & INTERPRETATION")
print("=" * 80)
print()
print(f"Two flies with identical connectomes ({N:,} neurons) developed distinct")
print(f"Hebbian weight profiles through embodied experience.")
print()
print(f"Key findings:")
print(f"  - {total_divergent:,} synapses show non-zero divergence")
print(f"  - Mean divergence magnitude: {mean_div:.6f}")
print(f"  - Maximum single-synapse divergence: {max_div:.6f}")
print()

# Identify most divergent functional group
if func_involvement:
    most_div_group = max(func_involvement, key=func_involvement.get)
    print(f"  - Most divergent functional group: {most_div_group} "
          f"(total involvement = {func_involvement[most_div_group]:.4f})")

# Identify most divergent region pair
if sorted_pairs:
    top_pair = sorted_pairs[0]
    print(f"  - Most divergent region pair: {top_pair[0][0]} -> {top_pair[0][1]} "
          f"(cumulative = {top_pair[1]:.6f})")

print()
print(f"  Directional bias: {'fly0' if fly0_stronger > fly1_stronger else 'fly1'} "
      f"shows stronger weights in {max(fly0_stronger, fly1_stronger)}/{TOP_K} top synapses")
print()

if len(region_integration) > 3:
    print(f"  Consciousness correlation:")
    print(f"    Integration-Divergence r = {r_integ_div:+.4f}")
    print(f"    Plasticity-Divergence r  = {r_plast_div:+.4f}")
    if abs(r_integ_div) > 0.3 or abs(r_plast_div) > 0.3:
        print(f"    => Significant relationship between network integration and")
        print(f"       experiential divergence, consistent with IIT predictions.")
    else:
        print(f"    => Weak correlation suggests plasticity divergence is distributed")
        print(f"       without strong concentration in high-integration regions.")
print()
print("=" * 80)
print("Analysis complete.")
print("=" * 80)
