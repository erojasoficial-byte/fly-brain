"""
Visual System: Maps flygym compound eye to connectome visual pathway.

Pipeline:
    flygym retina (2, 721, 2)  ->  brightness per ommatidium
        ->  R1-R8 photoreceptor rates = brightness (ON)
        ->  L1 lamina (OFF) rates = (1 - brightness)
        ->  L2 lamina (ON)  rates = brightness
        ->  Mi1 medulla: tonic 100Hz (L1 inhibits via connectome)
        ->  Tm1/Tm2 medulla (ON): rates = brightness
        ->  T2 lobula (OFF): rates = (1 - brightness)

    From T2 onwards the REAL connectome propagates spikes:
        T2 -> LC4 -> Giant Fiber -> escape

Biology:
    The early visual system (retina through medulla) uses graded
    potentials in real Drosophila. The LIF connectome model has a
    50x scale mismatch between Poisson stimulation (amp=250) and
    network spikes (amp=1), making multi-layer synaptic propagation
    impossible. We inject firing rates at each layer based on
    visual input; the connectome weights determine the final
    integration at LC4 and the escape decision at Giant Fiber.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

# FlyWire annotations (Schlegel et al. 2024)
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)

CACHE_DIR = Path(__file__).resolve().parent / "data"

# Firing rate parameters (Hz)
BASE_RATE = 20.0     # basal rate (silence / no stimulus)
MAX_RATE = 200.0     # max rate for retina / lamina
MI1_TONIC = 100.0    # Mi1 tonic baseline (modulated by L1 inhibition)
TM_MAX = 80.0        # Tm1/Tm2 max (ON pathway)
T2_MAX = 120.0       # T2 max (OFF pathway, drives LC4)
CONTRAST_THRESH = 0.3  # min contrast to activate T2 (filters checkerboard noise)
NUM_OMMATIDIA = 721  # per eye in flygym


class VisualSystem:
    """Maps flygym compound eye vision to connectome visual pathway rates.

    Identifies neurons at each visual layer from FlyWire annotations,
    maps them to flygym ommatidia, and computes firing rates from
    brightness. The key method is process_visual_layers() which
    returns combined (indices, rates) for ALL layers at once.
    """

    def __init__(self, flyid2i, i2flyid):
        self.flyid2i = flyid2i
        self.i2flyid = i2flyid

        # Load or download annotations
        annotations = self._load_annotations()

        # Identify photoreceptors R1-R8
        self.photo_left, self.photo_right = self._find_photoreceptors(annotations)

        # Identify lamina interneurons L1 (OFF) and L2 (ON)
        self.L1_left, self.L1_right = self._find_neurons_by_type(
            annotations, ['L1'])
        self.L2_left, self.L2_right = self._find_neurons_by_type(
            annotations, ['L2'])

        # Identify medulla neurons
        Mi1_L, Mi1_R = self._find_neurons_by_type(annotations, ['Mi1'])
        Tm1_L, Tm1_R = self._find_neurons_by_type(annotations, ['Tm1'])
        Tm2_L, Tm2_R = self._find_neurons_by_type(annotations, ['Tm2'])

        # Identify lobula neurons: T2 (OFF pathway -> LC4)
        T2_L, T2_R = self._find_neurons_by_type(annotations, ['T2', 'T2a'])

        # Looming detectors: LPLC2 (directional) and LC4 (lateralized)
        self.LPLC2_left, self.LPLC2_right = self._find_neurons_by_type(
            annotations, ['LPLC2'])
        self.LC4_left, self.LC4_right = self._find_neurons_by_type(
            annotations, ['LC4'])

        # Map ommatidia -> neuron groups (photoreceptors)
        self.omm_to_photo_left = self._map_ommatidia(self.photo_left)
        self.omm_to_photo_right = self._map_ommatidia(self.photo_right)

        # Build flat arrays for vectorized batch updates
        self._build_batch_arrays()

        # Build batch arrays for each layer
        self._L1_indices, self._L1_omm, self._L1_eye, self._n_L1 = \
            self._build_omm_batch(self.L1_left, self.L1_right)
        self._L2_indices, self._L2_omm, self._L2_eye, self._n_L2 = \
            self._build_omm_batch(self.L2_left, self.L2_right)

        # Mi1: tonic baseline (no ommatidium mapping needed)
        self._Mi1_all = np.array(
            [self.flyid2i[x] for x in Mi1_L + Mi1_R], dtype=np.int64)

        # Tm1, Tm2: ON pathway mapped to ommatidia
        self._Tm1_indices, self._Tm1_omm, self._Tm1_eye, self._n_Tm1 = \
            self._build_omm_batch(Tm1_L, Tm1_R)
        self._Tm2_indices, self._Tm2_omm, self._Tm2_eye, self._n_Tm2 = \
            self._build_omm_batch(Tm2_L, Tm2_R)

        # T2: OFF pathway mapped to ommatidia (drives LC4 via connectome)
        self._T2_indices, self._T2_omm, self._T2_eye, self._n_T2 = \
            self._build_omm_batch(T2_L, T2_R)

        # Print summary
        total_photo = len(self.photo_left) + len(self.photo_right)
        print(f"[VisualSystem] R1-R8: {total_photo} photoreceptors "
              f"(L={len(self.photo_left)}, R={len(self.photo_right)})")
        print(f"[VisualSystem] L1={self._n_L1} L2={self._n_L2} "
              f"Mi1={len(self._Mi1_all)} "
              f"Tm1={self._n_Tm1} Tm2={self._n_Tm2} T2={self._n_T2}")
        print(f"[VisualSystem] LPLC2: L={len(self.LPLC2_left)} "
              f"R={len(self.LPLC2_right)}  "
              f"LC4: L={len(self.LC4_left)} R={len(self.LC4_right)}")
        print("[VisualSystem] T2->LC4->GF via pure connectome weights")

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------

    def _load_annotations(self):
        """Download or load cached FlyWire annotations TSV."""
        cache_path = CACHE_DIR / "flywire_annotations.tsv"
        if cache_path.exists():
            print(f"[VisualSystem] Loading cached annotations from {cache_path}")
        else:
            print("[VisualSystem] Downloading FlyWire annotations...")
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                urlretrieve(ANNOTATIONS_URL, cache_path)
                print(f"[VisualSystem] Saved to {cache_path}")
            except Exception as e:
                print(f"[VisualSystem] Download failed: {e}")
                return None
        try:
            df = pd.read_csv(cache_path, sep='\t', low_memory=False)
            print(f"[VisualSystem] Annotations: {len(df)} neurons")
            return df
        except Exception as e:
            print(f"[VisualSystem] Error reading annotations: {e}")
            return None

    # ------------------------------------------------------------------
    # Photoreceptor identification
    # ------------------------------------------------------------------

    def _find_photoreceptors(self, annotations):
        """Find R1-R8 photoreceptor neuron IDs, separated by eye."""
        photo_left, photo_right = [], []
        if annotations is not None:
            photo_left, photo_right = self._find_from_annotations(annotations)
        if not photo_left and not photo_right:
            photo_left, photo_right = self._find_from_heuristic()
        return photo_left, photo_right

    def _find_from_annotations(self, df):
        """Find photoreceptors from cell_type column in annotations."""
        type_col = self._find_col(df, ['cell_type', 'type', 'hemibrain_type'])
        if type_col is None:
            return [], []
        id_col = self._find_col(df, ['root_id', 'Root ID', 'root_ID', 'flywire_id'])
        if id_col is None:
            id_col = df.columns[0]
        side_col = self._find_col(df, ['side', 'hemisphere', 'Side'])

        photo_pattern = re.compile(r'^R[1-8]', re.IGNORECASE)
        mask = df[type_col].astype(str).apply(lambda x: bool(photo_pattern.match(x)))
        photo_df = df[mask]

        print(f"[VisualSystem] Found {len(photo_df)} photoreceptor annotations")
        if len(photo_df) == 0:
            if 'super_class' in df.columns and 'cell_class' in df.columns:
                mask2 = (df['super_class'].str.lower() == 'sensory') & \
                        (df['cell_class'].str.lower().str.contains('photo|visual', na=False))
                photo_df = df[mask2]
            if len(photo_df) == 0:
                return [], []

        return self._split_lr(photo_df, id_col, side_col)

    def _find_from_heuristic(self):
        """Fallback: find photoreceptors from connectivity patterns."""
        data_dir = Path(__file__).resolve().parent / "data"
        try:
            conn = pd.read_parquet(data_dir / "2025_Connectivity_783.parquet")
        except Exception:
            return self._fallback_uniform()
        post_counts = conn.groupby("Postsynaptic_Index").size()
        pre_counts = conn.groupby("Presynaptic_Index").size()
        input_only = []
        for idx in range(len(self.flyid2i)):
            if post_counts.get(idx, 0) == 0 and pre_counts.get(idx, 0) > 10:
                input_only.append(self.i2flyid[idx])
        mid = len(input_only) // 2
        return input_only[:mid], input_only[mid:]

    def _fallback_uniform(self):
        """Last resort: distribute arbitrary neuron IDs as pseudo-photoreceptors."""
        n_per_eye = NUM_OMMATIDIA * 8
        all_ids = sorted(self.flyid2i.keys())
        return all_ids[:n_per_eye], all_ids[n_per_eye:2 * n_per_eye]

    # ------------------------------------------------------------------
    # General neuron-type finder
    # ------------------------------------------------------------------

    def _find_neurons_by_type(self, annotations, type_names):
        """Find neurons by exact cell_type match, separated by side."""
        if annotations is None:
            return [], []
        type_col = self._find_col(annotations, ['cell_type', 'type', 'hemibrain_type'])
        if type_col is None:
            return [], []
        id_col = self._find_col(annotations, ['root_id', 'Root ID', 'root_ID', 'flywire_id'])
        if id_col is None:
            id_col = annotations.columns[0]
        side_col = self._find_col(annotations, ['side', 'hemisphere', 'Side'])

        mask = annotations[type_col].astype(str).isin(set(type_names))
        return self._split_lr(annotations[mask], id_col, side_col)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_col(df, candidates):
        """Return the first column name from candidates that exists in df."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _split_lr(self, df, id_col, side_col):
        """Split a DataFrame of neurons into left/right lists."""
        left, right = [], []
        for _, row in df.iterrows():
            try:
                flyid = int(row[id_col])
            except (ValueError, TypeError):
                continue
            if flyid not in self.flyid2i:
                continue
            if side_col and pd.notna(row.get(side_col)):
                side = str(row[side_col]).lower().strip()
                if side in ('left', 'l'):
                    left.append(flyid)
                elif side in ('right', 'r'):
                    right.append(flyid)
                else:
                    (left if len(left) <= len(right) else right).append(flyid)
            else:
                (left if len(left) <= len(right) else right).append(flyid)
        return left, right

    # ------------------------------------------------------------------
    # Ommatidium -> neuron mapping
    # ------------------------------------------------------------------

    def _build_omm_batch(self, left_ids, right_ids):
        """Build vectorized batch arrays for any set of per-eye neurons."""
        left_map = self._map_ommatidia(left_ids)
        right_map = self._map_ommatidia(right_ids)
        indices, omm_ids, eye_ids = [], [], []
        for omm_idx, flyids in left_map.items():
            for flyid in flyids:
                indices.append(self.flyid2i[flyid])
                omm_ids.append(omm_idx)
                eye_ids.append(0)
        for omm_idx, flyids in right_map.items():
            for flyid in flyids:
                indices.append(self.flyid2i[flyid])
                omm_ids.append(omm_idx)
                eye_ids.append(1)
        return (np.array(indices, dtype=np.int64),
                np.array(omm_ids, dtype=np.int64),
                np.array(eye_ids, dtype=np.int64),
                len(indices))

    def _map_ommatidia(self, neuron_ids):
        """Map 721 ommatidia to groups of neuron IDs (sorted, evenly distributed)."""
        if not neuron_ids:
            return {}
        n = len(neuron_ids)
        sorted_ids = sorted(neuron_ids)
        omm_map = {}
        for omm_idx in range(NUM_OMMATIDIA):
            start = (omm_idx * n) // NUM_OMMATIDIA
            end = ((omm_idx + 1) * n) // NUM_OMMATIDIA
            if start < end:
                omm_map[omm_idx] = sorted_ids[start:end]
        return omm_map

    # ------------------------------------------------------------------
    # Batch arrays for photoreceptors
    # ------------------------------------------------------------------

    def _build_batch_arrays(self):
        """Precompute flat arrays for efficient batch rate updates."""
        indices, omm_ids, eye_ids = [], [], []
        for omm_idx, flyids in self.omm_to_photo_left.items():
            for flyid in flyids:
                indices.append(self.flyid2i[flyid])
                omm_ids.append(omm_idx)
                eye_ids.append(0)
        for omm_idx, flyids in self.omm_to_photo_right.items():
            for flyid in flyids:
                indices.append(self.flyid2i[flyid])
                omm_ids.append(omm_idx)
                eye_ids.append(1)
        self._photo_indices = np.array(indices, dtype=np.int64)
        self._omm_ids = np.array(omm_ids, dtype=np.int64)
        self._eye_ids = np.array(eye_ids, dtype=np.int64)
        self._n_photo = len(indices)

    # ------------------------------------------------------------------
    # Main API: process all visual layers at once
    # ------------------------------------------------------------------

    def process_visual_layers(self, vision_obs):
        """Convert flygym vision to T2 lobula firing rates based on contrast.

        Only T2 neurons are injected with Poisson rates. Other visual layers
        (R1-R8, L1, L2, Mi1, Tm) are identified but not injected because
        the Poisson scale factor (250x) causes widespread network noise that
        activates GF regardless of visual stimulus.

        T2 -> LC4 -> GF propagates through pure connectome weights.
        T2 rates are contrast-based: only ommatidia darker than the
        background mean get nonzero rates, preventing false escape.

        Args:
            vision_obs: np.ndarray shape (2, 721, 2), values in [0, 1]

        Returns:
            (t2_indices, t2_rates): T2 neuron indices and contrast rates.
            None, None if no T2 neurons are mapped.
        """
        if self._n_T2 == 0:
            return None, None

        brightness = np.mean(vision_obs, axis=2)  # (2, 721) in [0, 1]
        mean_bright = brightness.mean()

        # Contrast: how much darker than the background mean
        # 0 = at or above mean brightness, 1 = fully dark against bright bg
        raw_contrast = (mean_bright - brightness) / max(mean_bright, 0.05)

        # Threshold: ignore small contrast from checkerboard floor (~0.12)
        # Only respond to strong contrast from dark objects (ball ~1.0)
        contrast = np.clip(raw_contrast - CONTRAST_THRESH, 0.0, 1.0)

        # T2 rate: 0 for background, up to T2_MAX for darkest ommatidia
        omm_rates = T2_MAX * contrast  # (2, 721)

        t2_rates = omm_rates[self._T2_eye, self._T2_omm]
        return self._T2_indices, t2_rates

    # ------------------------------------------------------------------
    # Population index accessors (for brain_body_bridge monitoring)
    # ------------------------------------------------------------------

    def get_lplc2_indices(self, flyid2i):
        """Return LPLC2 left/right tensor indices for population monitoring."""
        result = {}
        left_idx = [flyid2i[fid] for fid in self.LPLC2_left if fid in flyid2i]
        right_idx = [flyid2i[fid] for fid in self.LPLC2_right if fid in flyid2i]
        if left_idx:
            result['LPLC2_left'] = np.array(left_idx, dtype=np.int64)
        if right_idx:
            result['LPLC2_right'] = np.array(right_idx, dtype=np.int64)
        return result

    def get_lc4_indices(self, flyid2i):
        """Return LC4 left/right tensor indices for population monitoring."""
        result = {}
        left_idx = [flyid2i[fid] for fid in self.LC4_left if fid in flyid2i]
        right_idx = [flyid2i[fid] for fid in self.LC4_right if fid in flyid2i]
        if left_idx:
            result['LC4_left'] = np.array(left_idx, dtype=np.int64)
        if right_idx:
            result['LC4_right'] = np.array(right_idx, dtype=np.int64)
        return result

    # ------------------------------------------------------------------
    # Legacy methods (kept for compatibility)
    # ------------------------------------------------------------------

    def process_vision(self, vision_obs):
        """Convert flygym vision to photoreceptor firing rates only."""
        if self._n_photo == 0:
            return None, None
        brightness = np.mean(vision_obs, axis=2)
        omm_rates = BASE_RATE + (MAX_RATE - BASE_RATE) * brightness
        return self._photo_indices, omm_rates[self._eye_ids, self._omm_ids]

    def process_lamina(self, vision_obs):
        """Compute lamina L1+L2 rates only."""
        if self._n_L1 == 0 and self._n_L2 == 0:
            return None, None
        brightness = np.mean(vision_obs, axis=2)
        parts_idx, parts_rate = [], []
        if self._n_L1 > 0:
            omm_rates = BASE_RATE + (MAX_RATE - BASE_RATE) * (1.0 - brightness)
            parts_idx.append(self._L1_indices)
            parts_rate.append(omm_rates[self._L1_eye, self._L1_omm])
        if self._n_L2 > 0:
            omm_rates = BASE_RATE + (MAX_RATE - BASE_RATE) * brightness
            parts_idx.append(self._L2_indices)
            parts_rate.append(omm_rates[self._L2_eye, self._L2_omm])
        return np.concatenate(parts_idx), np.concatenate(parts_rate)
