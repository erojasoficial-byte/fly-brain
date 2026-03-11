"""
NEST GPU benchmark runner for the Drosophila brain model.

Uses the custom user_m1 neuron (LIF + alpha synapse) compiled into NEST GPU.
NEST GPU cannot reset state within a process, so each benchmark trial runs
in a separate subprocess. This file serves dual purposes:

  1. Importable module with run_all_benchmarks() for the orchestrator
  2. Standalone subprocess worker when invoked with --worker

Called by benchmark.py orchestrator.
"""

import subprocess
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import traceback

from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    path_comp, path_con, path_res,
    get_experiment, print_summary_table, save_result_csv,
)

# ============================================================================
# NEST GPU Configuration
# ============================================================================

# Max spike times per neuron per simulation call.
# Scales with duration; too high causes CUDA OOM, too low truncates spikes.
N_MAX_SPIKE_TIMES = {0.1: 4000, 1: 4000, 10: 4000, 100: 15000, 1000: 15000}

# Model parameters (unitless; NEST GPU uses raw numbers, not Brian2 units)
MODEL_PARAMS = {
    'v_0': -52,       # mV  resting potential
    'v_rst': -52,     # mV  reset potential
    'v_th': -45,      # mV  spike threshold
    't_mbr': 20,      # ms  membrane time constant
    'tau': 5,         # ms  synaptic time constant
    't_rfc': 2.2,     # ms  refractory period
    't_dly': 1.8,     # ms  synaptic delay
    'w_syn': 0.275,   # mV  weight per synapse
    'f_poi': 250,     # Poisson weight scaling factor
}

# ============================================================================
# Worker: runs a single NEST GPU trial (called via subprocess)
# ============================================================================

def _run_worker_trial(t_run_sec, trial_num, experiment_name=None):
    """Build network, simulate, and return timing + spike counts as dict.

    Imports nestgpu only here so the main orchestrator process never loads it.
    """
    import nestgpu as ngpu

    experiment = get_experiment(experiment_name)

    t_run_ms = t_run_sec * 1000
    n_max_spikes = N_MAX_SPIKE_TIMES.get(t_run_sec, 4000)
    params = dict(MODEL_PARAMS)
    params['r_poi'] = experiment['stim_rate']

    result = {
        'trial': trial_num,
        't_run_sec': t_run_sec,
        'status': 'error',
        'network_creation_time': 0,
        'simulation_time': 0,
        'spike_retrieval_time': 0,
        'total_elapsed_time': 0,
        'n_spikes': 0,
        'n_active_neurons': 0,
    }

    total_start = time()

    try:
        # ---- ID mappings ----
        df_comp = pd.read_csv(str(path_comp), index_col=0)
        flyid2i = {j: i for i, j in enumerate(df_comp.index)}
        exc = [flyid2i[n] for n in experiment['neu_exc']]
        exc2 = [flyid2i[n] for n in experiment['neu_exc2']]
        slnc = [flyid2i[n] for n in experiment['neu_slnc']]

        # ---- Network creation ----
        net_start = time()
        df_con = pd.read_parquet(str(path_con))

        neu = ngpu.Create('user_m1', len(df_comp))
        ngpu.SetStatus(neu, {
            'v_m': params['v_0'], 'g_m': 0.0,
            'ref_on': 1, 'slnc_on': 0,
            'v_0': params['v_0'], 'v_rst': params['v_rst'],
            'v_th': params['v_th'], 'tau_mbr': params['t_mbr'],
            'tau_g': params['tau'], 't_ref': params['t_rfc'],
        })

        for i in slnc:
            ngpu.SetStatus([i], {'slnc_on': 1})

        i_pre = np.array(df_con['Presynaptic_Index'].values).tolist()
        i_post = np.array(df_con['Postsynaptic_Index'].values).tolist()
        conn_w = np.array(
            df_con['Excitatory x Connectivity'].values * params['w_syn']
        ).tolist()

        conn_spec = {'rule': 'one_to_one'}
        syn_spec = {
            'weight': {'array': conn_w},
            'delay': params['t_dly'],
            'receptor': 0,
        }
        ngpu.Connect(i_pre, i_post, conn_spec, syn_spec)

        # ---- Poisson inputs ----
        syn_spec_poi = {
            'receptor': 1, 'delay': 0.1,
            'weight': params['w_syn'] * params['f_poi'],
        }

        if len(exc) > 0:
            pois1 = ngpu.Create('poisson_generator', len(exc))
            ngpu.SetStatus(pois1, {'rate': params['r_poi']})
            i_poi = np.arange(pois1.i0, pois1.i0 + pois1.n).tolist()
            ngpu.Connect(i_poi, exc, {'rule': 'one_to_one'}, syn_spec_poi)
            for j in exc:
                ngpu.SetStatus([j], {'ref_on': 0})

        if len(exc2) > 0:
            pois2 = ngpu.Create('poisson_generator', len(exc2))
            ngpu.SetStatus(pois2, {'rate': 0})
            i_poi2 = np.arange(pois2.i0, pois2.i0 + pois2.n).tolist()
            ngpu.Connect(i_poi2, exc2, {'rule': 'one_to_one'}, syn_spec_poi)
            for j in exc2:
                ngpu.SetStatus([j], {'ref_on': 0})

        result['network_creation_time'] = time() - net_start

        # ---- Simulate ----
        ngpu.ActivateRecSpikeTimes(neu, n_max_spikes)

        sim_start = time()
        ngpu.Simulate(t_run_ms)
        result['simulation_time'] = time() - sim_start

        retrieval_start = time()
        spk_trn = ngpu.GetRecSpikeTimes(neu)
        result['spike_retrieval_time'] = time() - retrieval_start

        result['n_spikes'] = sum(len(s) for s in spk_trn)
        result['n_active_neurons'] = sum(1 for s in spk_trn if len(s) > 0)
        result['status'] = 'success'

    except Exception as e:
        result['status'] = f'error: {e}'

    result['total_elapsed_time'] = time() - total_start
    return result

# ============================================================================
# Orchestrator: spawns subprocess per trial, aggregates results
# ============================================================================

def run_single_benchmark(t_run_sec, n_run, experiment, logger,
                         run_idx=None, total_runs=None):
    """Run a NEST GPU benchmark by spawning one subprocess per trial."""
    exp_name = f'nestgpu_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: GPU (NEST GPU, user_m1 neuron)")
    logger.log(f"Experiment: {exp_name}")
    logger.log(f"Each trial runs in a separate subprocess (NEST GPU limitation)")

    experiment_name = experiment['key']

    timings = {}
    trial_results = []

    t_total_start = time()

    try:
        for trial in range(n_run):
            logger.log(f"  Spawning trial {trial + 1}/{n_run}...")

            worker_script = str(Path(__file__).resolve())
            cmd = [
                sys.executable, worker_script,
                '--worker', str(t_run_sec), str(trial),
                '--experiment', experiment_name,
            ]

            t_trial_start = time()
            timeout = max(t_run_sec * 20 + 120, 300)
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            trial_elapsed = time() - t_trial_start

            if proc.returncode != 0:
                logger.log(
                    f"    Trial {trial + 1} FAILED (exit {proc.returncode})"
                )
                for line in (proc.stderr or '').strip().split('\n')[-5:]:
                    if line.strip():
                        logger.log(f"    stderr: {line}")
                trial_results.append({
                    'trial': trial,
                    'status': f'error: exit code {proc.returncode}',
                    'simulation_time': 0,
                    'network_creation_time': 0,
                    'spike_retrieval_time': 0,
                    'n_spikes': 0,
                    'n_active_neurons': 0,
                    'total_elapsed_time': trial_elapsed,
                })
                continue

            trial_data = None
            for line in proc.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('{'):
                    try:
                        trial_data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if trial_data is None:
                logger.log(f"    Trial {trial + 1}: could not parse worker JSON")
                trial_results.append({
                    'trial': trial, 'status': 'error: no JSON output',
                    'simulation_time': 0, 'network_creation_time': 0,
                    'spike_retrieval_time': 0, 'n_spikes': 0,
                    'n_active_neurons': 0, 'total_elapsed_time': trial_elapsed,
                })
                continue

            trial_results.append(trial_data)

            logger.log(
                f"    Trial {trial + 1}/{n_run}: "
                f"net={trial_data['network_creation_time']:.2f}s, "
                f"sim={trial_data['simulation_time']:.2f}s, "
                f"retrieval={trial_data['spike_retrieval_time']:.2f}s, "
                f"spikes={trial_data['n_spikes']}"
            )

        # ---- Aggregate ----
        successful = [r for r in trial_results if r.get('status') == 'success']
        n_ok = len(successful)

        if successful:
            total_sim = sum(r['simulation_time'] for r in successful)
            avg_net = float(np.mean([r['network_creation_time'] for r in successful]))
            avg_retrieval = float(np.mean([r['spike_retrieval_time'] for r in successful]))
            total_spikes = sum(r['n_spikes'] for r in successful)
            n_active = max(r['n_active_neurons'] for r in successful)

            timings['network_creation_total'] = avg_net
            timings['device_build'] = 0.0
            timings['simulation_total'] = total_sim
            timings['simulation_avg_per_trial'] = total_sim / n_ok
            timings['spike_retrieval_avg'] = avg_retrieval
            timings['total_elapsed'] = time() - t_total_start

            total_simulated_time = t_run_sec * n_ok
            timings['realtime_ratio'] = (
                total_simulated_time / total_sim
                if total_sim > 0 else float('inf')
            )
            timings['realtime_ratio_total'] = (
                total_simulated_time / timings['total_elapsed']
                if timings['total_elapsed'] > 0 else float('inf')
            )

            results = {
                't_run_sec': t_run_sec,
                'n_run': n_run,
                'n_active_neurons': n_active,
                'n_spikes': total_spikes,
                'status': 'success',
                'timings': timings,
            }

            logger.log_raw("")
            logger.log_raw("-" * 60)
            logger.log("TIMING SUMMARY")
            logger.log_raw("-" * 60)
            logger.log(f"  Network creation (avg): {avg_net:>10.3f}s")
            logger.log(f"  Simulation (total):     {total_sim:>10.3f}s")
            logger.log(f"  Simulation (avg/trial): {total_sim / n_ok:>10.3f}s")
            logger.log(f"  Spike retrieval (avg):  {avg_retrieval:>10.3f}s")
            logger.log(f"  Total wall time:        {timings['total_elapsed']:>10.3f}s")
            logger.log(f"  -----------------------------------------")
            logger.log(f"  Simulated time:         {total_simulated_time:>10.1f}s ({n_ok} x {t_run_sec}s)")
            logger.log(f"  Realtime ratio (sim):   {timings['realtime_ratio']:>10.3f}x")
            logger.log(f"  Realtime ratio (total): {timings['realtime_ratio_total']:>10.3f}x")
            logger.log_raw("")
            logger.log(f"  Active neurons:         {n_active:>10d}")
            logger.log(f"  Total spikes:           {total_spikes:>10d}")
            logger.log_raw("-" * 60)
        else:
            timings['total_elapsed'] = time() - t_total_start
            timings['device_build'] = 0.0
            results = {
                't_run_sec': t_run_sec,
                'n_run': n_run,
                'n_active_neurons': 0,
                'n_spikes': 0,
                'status': 'error: all trials failed',
                'timings': timings,
            }
            logger.log("ERROR: all trials failed")

    except Exception as e:
        logger.log(f"ERROR: {e}")
        logger.log_raw(traceback.format_exc())
        timings['total_elapsed'] = time() - t_total_start
        timings['device_build'] = 0.0
        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': 0,
            'n_spikes': 0,
            'status': f'error: {e}',
            'timings': timings,
        }

    return results


def run_all_benchmarks(t_run_values=None, n_run_values=None,
                       experiment=None, logger=None):
    """
    Run all NEST GPU benchmark combinations.

    Args:
        t_run_values: List of t_run durations in seconds, or None for all
        n_run_values: List of n_run values to test, or None for all
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance
    """
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if experiment is None:
        experiment = get_experiment()

    backend_name = 'NEST GPU'

    benchmarks = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            benchmarks.append((t_run_sec, n_run))

    total_runs = len(benchmarks)

    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"BENCHMARK SUITE: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: GPU (NEST GPU, custom user_m1 neuron)")
    logger.log(f"t_run values: {t_run_values} seconds")
    logger.log(f"n_run values: {n_run_values}")
    logger.log(f"Total benchmarks: {total_runs}")
    logger.log_raw("=" * 80)

    all_results = []

    for run_idx, (t_run_sec, n_run) in enumerate(benchmarks, 1):
        result = run_single_benchmark(
            t_run_sec=t_run_sec,
            n_run=n_run,
            experiment=experiment,
            logger=logger,
            run_idx=run_idx,
            total_runs=total_runs,
        )
        all_results.append(result)
        save_result_csv(backend_name, result)

    print_summary_table(all_results, backend_name, logger)

    return all_results


# ============================================================================
# Subprocess entry point
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == '--worker':
        t_run_sec = float(sys.argv[2])
        trial_num = int(sys.argv[3])
        exp_name = None
        if '--experiment' in sys.argv:
            exp_idx = sys.argv.index('--experiment')
            if exp_idx + 1 < len(sys.argv):
                exp_name = sys.argv[exp_idx + 1]
        result = _run_worker_trial(t_run_sec, trial_num, exp_name)
        print(json.dumps(result), flush=True)
    else:
        print("This module is used as a worker subprocess by the benchmark system.")
        print("Run benchmarks via: python main.py --nestgpu")
