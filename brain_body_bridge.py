"""
Brain-Body Bridge: Connects fly-brain connectome simulation
to NeuroMechFly v2 (flygym) biomechanical body via descending neuron decoding.

Architecture:
  Sensory stimulus -> fly-brain (LIF neurons on GPU) -> DN spike readout
  -> firing rate decoder -> [left_drive, right_drive] -> HybridTurningController
  -> CPG -> 42 joint angles -> MuJoCo physics -> 3D viewer
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import deque

# Add fly-brain code directory to path for run_pytorch imports
_CODE_DIR = Path(__file__).resolve().parent / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from run_pytorch import TorchModel, MODEL_PARAMS, DT, get_weights, get_hash_tables

# ============================================================================
# Hebbian Plasticity Constants — inherent neural tissue property
# ============================================================================

HEBB_BATCH   = 10     # accumulate spikes for 10 brain steps before update
HEBB_ETA     = 1e-4   # learning rate
HEBB_ALPHA   = 1e-7   # weight decay rate (multiplicative)
PLASTIC_PATH = Path(__file__).resolve().parent / 'data' / 'plastic_weights.pt'

# ============================================================================
# Descending Neuron (DN) Definitions — FlyWire IDs from example.ipynb
# ============================================================================

DN_NEURONS = {
    # Forward walking (P9 / oDN1) — 4 neurons
    'P9_left':       720575940627652358,
    'P9_right':      720575940635872101,
    'P9_oDN1_left':  720575940626730883,
    'P9_oDN1_right': 720575940620300308,
    # Sustained turning (DNa01) — 2 neurons
    'DNa01_left':    720575940644438551,
    'DNa01_right':   720575940627787609,
    # Transient turning (DNa02) — 2 neurons
    'DNa02_left':    720575940604737708,
    'DNa02_right':   720575940629327659,
    # Backward walking (MDN — Moonwalker Descending Neuron) — 4 neurons
    'MDN_1':         720575940616026939,
    'MDN_2':         720575940631082808,
    'MDN_3':         720575940640331472,
    'MDN_4':         720575940610236514,
    # Escape (Giant Fiber) — 2 neurons
    'GF_1':          720575940622838154,
    'GF_2':          720575940632499757,
    # Antennal grooming (aDN1) — 2 neurons
    'aDN1_left':     720575940624319124,
    'aDN1_right':    720575940616185531,
    # Feeding / proboscis motor (MN9) — 2 neurons
    'MN9_left':      720575940660219265,
    'MN9_right':     720575940618238523,
}

# Grouped DN names for readability
DN_GROUPS = {
    'forward':  ['P9_left', 'P9_right', 'P9_oDN1_left', 'P9_oDN1_right'],
    'turn_L':   ['DNa01_left', 'DNa02_left'],
    'turn_R':   ['DNa01_right', 'DNa02_right'],
    'backward': ['MDN_1', 'MDN_2', 'MDN_3', 'MDN_4'],
    'escape':   ['GF_1', 'GF_2'],
    'groom':    ['aDN1_left', 'aDN1_right'],
    'feed':     ['MN9_left', 'MN9_right'],
}

# ============================================================================
# Sensory Stimulus Definitions — neuron IDs and rates from example.ipynb
# ============================================================================

STIMULI = {
    'sugar': {
        'neurons': [
            720575940624963786, 720575940630233916, 720575940637568838,
            720575940638202345, 720575940617000768, 720575940630797113,
            720575940632889389, 720575940621754367, 720575940621502051,
            720575940640649691, 720575940639332736, 720575940616885538,
            720575940639198653, 720575940639259967, 720575940617937543,
            720575940632425919, 720575940633143833, 720575940612670570,
            720575940628853239, 720575940629176663, 720575940611875570,
        ],
        'rate': 200.0,
        'description': 'Sugar GRNs (21 neurons, 200 Hz)',
    },
    'p9': {
        'neurons': [
            720575940627652358,  # P9 left
            720575940635872101,  # P9 right
        ],
        'rate': 100.0,
        'description': 'P9 forward walking (2 neurons, 100 Hz)',
    },
    'lc4': {
        'neurons': [
            720575940605598892, 720575940611134833, 720575940612580977,
            720575940613256863, 720575940613260959, 720575940614914107,
            720575940615462587, 720575940617176321, 720575940617266722,
            720575940618807105, 720575940620795728, 720575940622108001,
            720575940624017251, 720575940625038090, 720575940625934973,
            720575940625991043, 720575940626605200, 720575940626626895,
            720575940628454522, 720575940628462340, 720575940630851036,
            720575940638496720, 720575940603637438, 720575940610522009,
            720575940612093351, 720575940612323025, 720575940612380723,
            720575940612498129, 720575940612518055, 720575940612968421,
            720575940613609484, 720575940613638041, 720575940614572742,
            720575940614582946, 720575940615053580, 720575940615127227,
            720575940615232217, 720575940615575007, 720575940616066705,
            720575940616713355, 720575940617026260, 720575940617348379,
            720575940618002644, 720575940618234704, 720575940618234715,
            720575940618266459, 720575940618267227, 720575940618275520,
            720575940618312606, 720575940618676440, 720575940618709158,
            720575940618723749, 720575940619397542, 720575940620314221,
            720575940620314612, 720575940620731380, 720575940620903551,
            720575940621145821, 720575940621522458, 720575940621753579,
            720575940622330582, 720575940622531767, 720575940622939836,
            720575940624111763, 720575940624790781, 720575940624856762,
            720575940625841351, 720575940625845447, 720575940625906702,
            720575940625932421, 720575940626553596, 720575940626916936,
            720575940627519107, 720575940628064260, 720575940628081541,
            720575940628419527, 720575940628518400, 720575940628599895,
            720575940628606713, 720575940628699560, 720575940628891863,
            720575940629753807, 720575940629964591, 720575940630154660,
            720575940630484495, 720575940630998339, 720575940631032657,
            720575940631338271, 720575940632475449, 720575940632715234,
            720575940632769180, 720575940633013355, 720575940633218863,
            720575940633580384, 720575940634517856, 720575940635835967,
            720575940636957006, 720575940638456227, 720575940639817947,
            720575940640612480, 720575940641213824, 720575940645821316,
            720575940649229433, 720575940652611745,
        ],
        'rate': 200.0,
        'description': 'LC4 looming (104 neurons, 200 Hz)',
    },
    'jo': {
        'neurons': [
            # JO-E subtypes (vibration/touch — connects to aDN1 for grooming)
            720575940645106376, 720575940615272415, 720575940619869120,
            720575940620257345, 720575940620382889, 720575940630834683,
            720575940632449619, 720575940634020508, 720575940605530302,
            720575940607140035, 720575940608742409, 720575940615590843,
            720575940620410177, 720575940621870618, 720575940622344170,
            720575940623298559, 720575940626042149, 720575940627379333,
            720575940630080071, 720575940632128031, 720575940632307527,
            720575940634820703,
            # JO-C subtypes
            720575940606154370, 720575940605919334, 720575940608884931,
            720575940616655989, 720575940620543110, 720575940622937528,
            720575940624799290, 720575940626565455, 720575940627941431,
            720575940627977457, 720575940628160617, 720575940629188251,
            720575940641921421,
            # JO-EDM subset
            720575940615972027, 720575940618941037, 720575940619729835,
            720575940627282279, 720575940628903247, 720575940604122982,
            720575940609486690, 720575940609541917, 720575940610018266,
            720575940611061526, 720575940611273395, 720575940611684787,
            720575940614060829, 720575940616040587, 720575940618599872,
            720575940618684481, 720575940619663239, 720575940619932654,
            720575940620919578, 720575940621218729, 720575940622271684,
            720575940622638276, 720575940623312828, 720575940625797617,
            720575940625962568, 720575940626309438, 720575940626666066,
            720575940627109991, 720575940628101126, 720575940628978450,
            720575940629055721, 720575940629650997, 720575940629985900,
            720575940630992557, 720575940637054835, 720575940637084762,
            720575940638664437, 720575940646927668, 720575940646929204,
            720575940659131009,
            # JO-EDP
            720575940609522461, 720575940610261346, 720575940613641915,
            720575940615469785, 720575940616589878, 720575940616951124,
            720575940619479979, 720575940621218985, 720575940628444667,
            720575940634634606, 720575940640753267, 720575940650244342,
            # JO-EVL
            720575940615573597, 720575940615848788, 720575940619083349,
            720575940621397417, 720575940621625597, 720575940622283912,
            720575940627049731, 720575940629022149, 720575940630122015,
            720575940630564179, 720575940633153375, 720575940637410869,
            720575940638681845, 720575940621033477, 720575940621776410,
            720575940621815690, 720575940622234211, 720575940622635817,
            720575940623897096, 720575940626148354, 720575940626540821,
            720575940628258715, 720575940629743063, 720575940630202624,
            720575940630544967, 720575940633553820, 720575940644036644,
            # JO-EVM
            720575940602132509, 720575940602506208, 720575940610759634,
            720575940614188149, 720575940615809349, 720575940615976891,
            720575940619341105, 720575940621092534, 720575940622419165,
            720575940622449388, 720575940623108134, 720575940624981436,
            720575940628192055, 720575940630059847, 720575940632767383,
            720575940639296189, 720575940645466500, 720575940611783464,
            720575940612307478, 720575940612960552, 720575940614351477,
            720575940617212134, 720575940617434086, 720575940618130334,
            720575940620249734, 720575940620940276, 720575940621010352,
            720575940621729757, 720575940623437547, 720575940624546062,
            720575940624686268, 720575940625054647, 720575940625605905,
            720575940626795909, 720575940627585688, 720575940630020111,
            720575940632175268, 720575940634073183, 720575940634891700,
            720575940637012196, 720575940637243504, 720575940639339392,
            720575940659426177,
            # JO-EVP
            720575940620444654, 720575940631866508, 720575940607853833,
            720575940611088563, 720575940612773374, 720575940613221928,
            720575940615024543, 720575940615986459, 720575940617811013,
            720575940618467195, 720575940621442224, 720575940622199977,
            720575940624915230, 720575940625559358, 720575940627104649,
            720575940627314088, 720575940633058989, 720575940636335735,
            # JO-CA
            720575940605800369, 720575940608784579, 720575940618135109,
            720575940626719101, 720575940629296185, 720575940636137591,
            720575940602720940, 720575940610079857, 720575940614427195,
            720575940616501787, 720575940617156445, 720575940625909962,
            720575940626241369, 720575940629105658, 720575940629138959,
            720575940636559534, 720575940641372661,
            # JO-CL
            720575940626135548, 720575940627751567, 720575940604753437,
            720575940613971485, 720575940614835362, 720575940623399059,
            720575940630319671, 720575940639082062,
            # JO-CM
            720575940607386307, 720575940634512992, 720575940614035485,
            720575940618901424, 720575940630070343, 720575940633443353,
            720575940635058612, 720575940637632419, 720575940625626000,
        ],
        'rate': 300.0,
        'description': 'JO touch/vibration (188 neurons, 300 Hz)',
    },
    'bitter': {
        'neurons': [
            720575940619072513, 720575940646212996, 720575940622298631,
            720575940642088333, 720575940627692048, 720575940617239197,
            720575940618682526, 720575940604714528, 720575940603266592,
            720575940604027168, 720575940619197093, 720575940610259370,
            720575940627578156, 720575940629481516, 720575940618887217,
            720575940614281266, 720575940634859188, 720575940645743412,
            720575940637742911, 720575940617094208, 720575940629416318,
            720575940630195909, 720575940615641798, 720575940638312262,
            720575940624310345, 720575940621778381, 720575940619659861,
            720575940629146711, 720575940625750105, 720575940610483162,
            720575940610481370, 720575940602353632, 720575940610773090,
            720575940617433830, 720575940628962407, 720575940626287336,
            720575940623183083, 720575940618025199, 720575940619028208,
            720575940621864060, 720575940613061118, 720575940621008895,
        ],
        'rate': 200.0,
        'description': 'Bitter GRNs (41 neurons, 200 Hz)',
    },
    'or56a': {
        'neurons': [
            720575940659222657, 720575940641403021, 720575940624211470,
            720575940616536209, 720575940615427734, 720575940628380827,
            720575940654069409, 720575940613671330, 720575940644590116,
            720575940612972328, 720575940627318696, 720575940627805096,
            720575940632190765, 720575940633031085, 720575940634955188,
            720575940621106102, 720575940615923131, 720575940608928324,
            720575940631467591, 720575940622553420, 720575940628086607,
            720575940626357586, 720575940632041043, 720575940618946901,
            720575940616095318, 720575940626411097, 720575940634614367,
            720575940603832288, 720575940620055905, 720575940609633378,
            720575940637704676, 720575940638202852, 720575940622713578,
            720575940635705963, 720575940629830508, 720575940630257772,
            720575940619539182, 720575940612019442, 720575940639931893,
        ],
        'rate': 250.0,
        'description': 'Or56a olfactory (39 neurons, 250 Hz)',
    },
}


# ============================================================================
# Brain Engine
# ============================================================================

class BrainEngine:
    """Wraps fly-brain TorchModel for step-by-step execution on GPU."""

    def __init__(self, device='cuda', plastic_path=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dt = DT  # 0.1 ms
        self._plastic_path = Path(plastic_path) if plastic_path else PLASTIC_PATH

        data_dir = Path(__file__).resolve().parent / 'data'
        comp_path = data_dir / '2025_Completeness_783.csv'
        conn_path = data_dir / '2025_Connectivity_783.parquet'

        # Build FlyWire ID <-> tensor index mappings
        self.flyid2i, self.i2flyid = get_hash_tables(str(comp_path))
        self.num_neurons = len(self.flyid2i)

        # Load sparse weight matrix
        weights = get_weights(str(conn_path), str(comp_path), str(data_dir))
        weights = weights.to(device=self.device)

        # Create model (batch=1 for real-time single-trial)
        self.model = TorchModel(
            batch=1, size=self.num_neurons, dt=self.dt,
            params=MODEL_PARAMS, weights=weights, device=self.device,
        )

        # Initialize neural state
        self.state = self.model.state_init()

        # Input rate tensor (modified by set_stimulus)
        self.rates = torch.zeros(1, self.num_neurons, device=self.device)

        # Precompute DN neuron tensor indices
        self.dn_indices = {}
        for name, flyid in DN_NEURONS.items():
            if flyid in self.flyid2i:
                self.dn_indices[name] = self.flyid2i[flyid]

        # Precompute stimulus neuron tensor indices
        self.stim_indices = {}
        for stim_name, stim_info in STIMULI.items():
            indices = [self.flyid2i[nid] for nid in stim_info['neurons']
                       if nid in self.flyid2i]
            self.stim_indices[stim_name] = indices

        # Population monitoring (registered externally)
        self.populations = {}  # {name: tensor_indices}

        print(f"[BrainEngine] {self.num_neurons} neurons on {self.device}")
        print(f"[BrainEngine] DN neurons mapped: "
              f"{len(self.dn_indices)}/{len(DN_NEURONS)}")
        for s, idx in self.stim_indices.items():
            print(f"  '{s}': {len(idx)}/{len(STIMULI[s]['neurons'])} neurons")

        # Hebbian plasticity — inherent property of neural tissue
        self._init_plasticity()

    # ── Hebbian Plasticity ──────────────────────────────────────────────

    def _init_plasticity(self):
        """Initialize Hebbian plasticity as inherent neural tissue property.

        Every synapse changes continuously from the first brain step.
        Co-active synapses strengthen; non-co-active synapses decay.
        """
        w = self.model.weights
        self._row_ptr = w.crow_indices()
        self._col_idx = w.col_indices()
        self._syn_vals = w.values()  # mutable view into weight matrix

        # Preserve excitatory/inhibitory identity
        self._sign_mask = torch.sign(self._syn_vals)

        # Original magnitudes for clamping (max 3× growth)
        self._abs_orig = self._syn_vals.abs().clone()

        # Expand row_ptr to per-synapse post-synaptic indices
        row_lengths = self._row_ptr[1:] - self._row_ptr[:-1]
        self._post_idx = torch.repeat_interleave(
            torch.arange(self.num_neurons, device=self.device), row_lengths
        )

        # Spike accumulator and step counter
        self._spike_acc = torch.zeros(self.num_neurons, device=self.device)
        self._hebb_count = 0

        # Load persisted weights if available
        if self._plastic_path.exists():
            saved = torch.load(self._plastic_path, map_location=self.device,
                               weights_only=True)
            if saved.shape == self._syn_vals.shape:
                self._syn_vals.copy_(saved)
                self._sign_mask = torch.sign(self._syn_vals)
                print(f"[BrainEngine] Loaded plastic weights from {self._plastic_path}")
            else:
                print(f"[BrainEngine] Plastic weights shape mismatch, starting fresh")

        # Precompute per-synapse clamp bounds (based on original magnitudes)
        max_mag = 3.0 * self._abs_orig
        self._clamp_min = torch.where(
            self._sign_mask < 0, -max_mag, torch.zeros_like(max_mag))
        self._clamp_max = torch.where(
            self._sign_mask > 0, max_mag, torch.zeros_like(max_mag))

        print(f"[BrainEngine] Hebbian plasticity active: "
              f"{len(self._syn_vals)} synapses")

    def _hebb_update(self):
        """Hebbian update: co-active synapses strengthen, all decay."""
        avg = self._spike_acc / HEBB_BATCH
        self._spike_acc.zero_()

        pre = avg[self._col_idx]
        post = avg[self._post_idx]

        # Hebbian potentiation + weight decay
        dW = HEBB_ETA * pre * post * self._sign_mask - HEBB_ALPHA * self._syn_vals
        self._syn_vals.add_(dW)

        # Sign-preserving clamp: cap magnitude at 3× original
        self._syn_vals.clamp_(min=self._clamp_min, max=self._clamp_max)

    def save_plastic_weights(self):
        """Persist current synaptic weights to disk."""
        torch.save(self._syn_vals.detach().cpu(), self._plastic_path)
        print(f"[BrainEngine] Saved plastic weights to {self._plastic_path}")

    # ────────────────────────────────────────────────────────────────────

    def set_stimulus(self, stim_name):
        """Set input firing rates for a named stimulus. None clears all."""
        self.rates.zero_()
        if stim_name and stim_name in STIMULI:
            idx = self.stim_indices.get(stim_name, [])
            if idx:
                self.rates[0, idx] = STIMULI[stim_name]['rate']

    def set_visual_rates(self, photo_indices, photo_rates):
        """Set firing rates for photoreceptor neurons from visual input.

        Unlike set_stimulus, this does NOT zero all rates first —
        photoreceptor rates coexist with other sensory stimuli
        (GRNs, JO neurons, etc.) that may be active simultaneously.

        Args:
            photo_indices: np.ndarray of tensor indices (int64)
            photo_rates:   np.ndarray of firing rates in Hz (float64)
        """
        if photo_indices is None or len(photo_indices) == 0:
            return
        self.rates[0, photo_indices] = torch.as_tensor(
            photo_rates, dtype=torch.float32, device=self.device,
        )

    def set_sensory_rates(self, indices, rates):
        """Set firing rates for somatosensory/auditory (JO) neurons.

        Uses element-wise maximum with existing rates so that manual
        keyboard stimuli (set_stimulus) are not overwritten by lower
        somatosensory rates.

        Args:
            indices: np.ndarray of tensor indices (int64)
            rates:   np.ndarray of firing rates in Hz (float64)
        """
        if indices is None or len(indices) == 0:
            return
        new_rates = torch.as_tensor(
            rates, dtype=torch.float32, device=self.device)
        current = self.rates[0, indices]
        self.rates[0, indices] = torch.maximum(current, new_rates)

    @torch.no_grad()
    def step(self):
        """Advance brain by one timestep (0.1 ms). Returns spike tensor."""
        cond, dbuf, spk, v, ref = self.state
        self.state = self.model(self.rates, cond, dbuf, spk, v, ref)
        # Hebbian plasticity — inherent tissue property
        spikes = self.state[2]
        self._spike_acc += spikes.squeeze(0)
        self._hebb_count += 1
        if self._hebb_count >= HEBB_BATCH:
            self._hebb_update()
            self._hebb_count = 0
        return spikes  # shape (1, num_neurons)

    def get_dn_spikes(self):
        """Return dict of current DN spike values {name: 0.0 or 1.0}."""
        spk = self.state[2]
        return {name: spk[0, idx].item()
                for name, idx in self.dn_indices.items()}

    def register_population(self, name, tensor_indices):
        """Register a neuron population for aggregate spike monitoring."""
        self.populations[name] = tensor_indices
        print(f"[BrainEngine] Population '{name}': {len(tensor_indices)} neurons")

    def get_population_spikes(self):
        """Return {name: mean_spike_fraction} for all registered populations."""
        spk = self.state[2]
        result = {}
        for name, indices in self.populations.items():
            result[name] = spk[0, indices].mean().item()
        return result


# ============================================================================
# DN Rate Decoder
# ============================================================================

class DNRateDecoder:
    """Computes firing rates of DN neurons using a sliding window over spikes."""

    def __init__(self, window_ms=50.0, dt_ms=0.1, max_rate=200.0):
        self.window_steps = int(window_ms / dt_ms)
        self.dt_s = dt_ms / 1000.0
        self.max_rate = max_rate
        self.dn_names = list(DN_NEURONS.keys())
        self.spike_buffer = {
            n: deque(maxlen=self.window_steps) for n in self.dn_names
        }
        self.rates = {n: 0.0 for n in self.dn_names}
        # Population tracking (LPLC2, LC4 by side)
        self.pop_names = []
        self.pop_buffer = {}
        self.pop_rates = {}

    def register_population(self, name):
        """Register a population name for rate tracking."""
        self.pop_names.append(name)
        self.pop_buffer[name] = deque(maxlen=self.window_steps)
        self.pop_rates[name] = 0.0

    def update(self, dn_spikes, pop_spikes=None):
        """Add one timestep of DN spikes and recompute firing rates."""
        for name in self.dn_names:
            self.spike_buffer[name].append(dn_spikes.get(name, 0.0))
            # Use actual buffer length to avoid underestimating rates at startup
            n_samples = len(self.spike_buffer[name])
            actual_window_s = n_samples * self.dt_s
            if actual_window_s > 0:
                self.rates[name] = sum(self.spike_buffer[name]) / actual_window_s
            else:
                self.rates[name] = 0.0
        # Update population rates
        if pop_spikes:
            for name in self.pop_names:
                val = pop_spikes.get(name, 0.0)
                self.pop_buffer[name].append(val)
                n_samples = len(self.pop_buffer[name])
                actual_window_s = n_samples * self.dt_s
                if actual_window_s > 0:
                    self.pop_rates[name] = sum(self.pop_buffer[name]) / actual_window_s
                else:
                    self.pop_rates[name] = 0.0

    def get_rate(self, name):
        """Current firing rate in Hz."""
        return self.rates.get(name, 0.0)

    def get_normalized(self, name):
        """Firing rate normalized to [0, 1] by max_rate."""
        return min(self.rates.get(name, 0.0) / self.max_rate, 1.0)

    def get_group_rate(self, group_name):
        """Mean normalized rate for a DN group (forward, escape, etc.)."""
        names = DN_GROUPS.get(group_name, [])
        if not names:
            return 0.0
        return np.mean([self.get_normalized(n) for n in names])

    def get_pop_rate(self, name):
        """Current population firing rate (mean spike fraction smoothed)."""
        return self.pop_rates.get(name, 0.0)


# ============================================================================
# Brain-Body Bridge
# ============================================================================

class BrainBodyBridge:
    """Converts DN firing rates to [left_drive, right_drive] for flygym."""

    def __init__(self, decoder,
                 escape_threshold=0.3, groom_threshold=0.02,
                 feeding_threshold=0.05,
                 escape_turn_gain=4.0,
                 tactile_escape_force=35.0,
                 sound_orientation_gain=0.3,
                 olfactory_attraction_gain=10.0):
        self.decoder = decoder
        self.escape_threshold = escape_threshold
        self.groom_threshold = groom_threshold
        self.feeding_threshold = feeding_threshold
        self.escape_turn_gain = escape_turn_gain
        self.tactile_escape_force = tactile_escape_force
        self.sound_orientation_gain = sound_orientation_gain
        self.olfactory_attraction_gain = olfactory_attraction_gain

        # Current behavioral state
        self.mode = 'walking'  # 'walking', 'grooming', 'escape', 'feeding', 'flight'
        self.left_drive = 0.0
        self.right_drive = 0.0
        self.flight_active = False  # set by FlightSystem when airborne
        self._mode_timer = 0.0     # time in current mode (seconds)
        self._min_mode_dur = 0.3   # minimum seconds before mode switch

        # Directional escape state
        self.threat_asym = 0.0  # +1 = right threat, -1 = left threat
        self.visual_threat_bias = 0.0  # T2 per-eye fallback (set by fly_embodied)

        # Somatosensory state (set externally by fly_embodied)
        self.tactile_force = 0.0           # max contact force (N)
        self.sound_orientation_bias = 0.0  # +1 = turn right, -1 = turn left

        # Gustatory state (set externally by fly_embodied)
        self.bitter_active = False  # True when legs on bitter zone

        # Olfactory state (set externally by fly_embodied)
        self.olfactory_attraction_bias = 0.0  # +1=turn right, -1=turn left
        self.olfactory_repulsive = False      # True when strong repulsive odor
        self.olfactory_repulsion_bias = 0.0   # +1=threat on right

    def _set_mode(self, new_mode):
        """Switch mode with hysteresis — ignore rapid switches."""
        if new_mode != self.mode:
            if self._mode_timer >= self._min_mode_dur:
                self.mode = new_mode
                self._mode_timer = 0.0
        # If same mode, just keep the timer running

    def compute_drive(self, dt=0.01):
        """Compute [left_drive, right_drive] from current DN firing rates.

        Returns np.ndarray of shape (2,) suitable for HybridTurningController.
        """
        d = self.decoder
        self._mode_timer += dt

        # Flight mode: legs inactive, forces applied externally
        if self.flight_active:
            self._set_mode('flight')
            self.left_drive = 0.0
            self.right_drive = 0.0
            return np.array([0.0, 0.0])

        # Forward drive from P9/oDN1 (4 neurons) + MN9 feeding approach (2 neurons)
        p9_drive = np.mean([
            d.get_normalized('P9_left'),
            d.get_normalized('P9_right'),
            d.get_normalized('P9_oDN1_left'),
            d.get_normalized('P9_oDN1_right'),
        ])
        mn9_drive = np.mean([
            d.get_normalized('MN9_left'),
            d.get_normalized('MN9_right'),
        ])
        forward = p9_drive + 0.5 * mn9_drive  # MN9 adds approach behavior

        # Turning: left-right asymmetry of DNa01 (sustained) + DNa02 (transient)
        turn_sustained = (d.get_normalized('DNa01_left') -
                          d.get_normalized('DNa01_right'))
        turn_transient = (d.get_normalized('DNa02_left') -
                          d.get_normalized('DNa02_right'))
        turn = turn_sustained + 0.5 * turn_transient

        # Backward drive from MDN (4 neurons)
        backward = np.mean([
            d.get_normalized('MDN_1'),
            d.get_normalized('MDN_2'),
            d.get_normalized('MDN_3'),
            d.get_normalized('MDN_4'),
        ])

        # Escape: Giant Fiber (2 neurons)
        gf = np.mean([
            d.get_normalized('GF_1'),
            d.get_normalized('GF_2'),
        ])

        # Grooming: aDN1 (2 neurons)
        adn1 = np.mean([
            d.get_normalized('aDN1_left'),
            d.get_normalized('aDN1_right'),
        ])

        # Tactile escape: strong contact force triggers escape even if
        # GF hasn't fired via the connectome (bridge-level backup)
        tactile_escape = self.tactile_force > self.tactile_escape_force

        # Bitter taste escape: bridge-level backup for bitter aversion
        bitter_escape = self.bitter_active

        # Feeding: MN9 (proboscis motor neurons)
        mn9 = np.mean([
            d.get_normalized('MN9_left'),
            d.get_normalized('MN9_right'),
        ])

        # Behavioral mode selection (with hysteresis)
        # Priority: escape > grooming > feeding > walking
        olfactory_escape = self.olfactory_repulsive

        if (gf > self.escape_threshold or tactile_escape
                or bitter_escape or olfactory_escape):
            self._set_mode('escape')
        elif adn1 > self.groom_threshold:
            self._set_mode('grooming')
        elif mn9 > self.feeding_threshold:
            self._set_mode('feeding')
        else:
            self._set_mode('walking')

        # Compute drives based on ACTUAL mode (after hysteresis)
        if self.mode == 'escape':
            if olfactory_escape and gf <= self.escape_threshold:
                self.threat_asym = self.olfactory_repulsion_bias
            elif tactile_escape and gf <= self.escape_threshold:
                self.threat_asym = 0.0
            else:
                lplc2_L = d.get_pop_rate('LPLC2_left')
                lplc2_R = d.get_pop_rate('LPLC2_right')
                lc4_L = d.get_pop_rate('LC4_left')
                lc4_R = d.get_pop_rate('LC4_right')
                threat_left = lplc2_L + 0.5 * lc4_L
                threat_right = lplc2_R + 0.5 * lc4_R
                threat_total = threat_left + threat_right
                if threat_total > 1e-6:
                    self.threat_asym = (threat_right - threat_left) / threat_total
                else:
                    self.threat_asym = self.visual_threat_bias
            escape_turn = self.threat_asym * self.escape_turn_gain
            escape_speed = 1.3
            self.left_drive = np.clip(
                escape_speed * (1.0 - escape_turn), -0.5, 1.5)
            self.right_drive = np.clip(
                escape_speed * (1.0 + escape_turn), -0.5, 1.5)

        elif self.mode in ('grooming', 'feeding'):
            self.left_drive = 0.0
            self.right_drive = 0.0

        else:  # walking
            sound_turn = (self.sound_orientation_bias *
                          self.sound_orientation_gain)
            olfactory_turn = (self.olfactory_attraction_bias *
                              self.olfactory_attraction_gain)
            effective_turn = turn + sound_turn + olfactory_turn
            self.left_drive = np.clip(
                forward * (1.0 + effective_turn) - backward, -0.5, 1.5)
            self.right_drive = np.clip(
                forward * (1.0 - effective_turn) - backward, -0.5, 1.5)

        return np.array([self.left_drive, self.right_drive])

    def get_status_str(self):
        """One-line status string for HUD display."""
        d = self.decoder
        fwd = np.mean([d.get_normalized(n) for n in DN_GROUPS['forward']])
        esc = np.mean([d.get_normalized(n) for n in DN_GROUPS['escape']])
        grm = np.mean([d.get_normalized(n) for n in DN_GROUPS['groom']])
        bkw = np.mean([d.get_normalized(n) for n in DN_GROUPS['backward']])
        threat_str = ""
        if self.mode == 'escape':
            side = "R" if self.threat_asym > 0 else "L"
            threat_str = f" threat={side}({self.threat_asym:+.2f})"
        return (f"[{self.mode:>8s}] fwd={fwd:.2f} bkw={bkw:.2f} "
                f"esc={esc:.2f} grm={grm:.2f} "
                f"drive=[{self.left_drive:.2f}, {self.right_drive:.2f}]"
                f"{threat_str}")
