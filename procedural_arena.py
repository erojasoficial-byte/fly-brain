"""
Procedural Arena: Minecraft-like world that generates itself around the flies.

Chunks (60mm cells) are seeded deterministically — revisiting a chunk produces
identical content.  40 pre-allocated mocap bodies form an obstacle pool; they
are placed/hidden by moving them above-ground or to z=-1000.

Obstacles are visual-only (contype=0, conaffinity=0): flies perceive them
through compound eyes and react via Giant Fiber escape or avoidance, but
there are no extra collision pairs that could cause NaN instability.

Each chunk may also spawn odor, vibration, and taste sources that the sensory
systems consume through the arena's properties.
"""

import numpy as np
from numpy.random import RandomState

from flygym.arena import BaseArena
from somatosensory import VibrationSource
from olfactory import OdorSource
from gustatory import TasteZone


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 60.0       # mm — fly body ~2.5mm, walk ~2mm/s → ~30s to cross
ACTIVE_RADIUS = 2       # chunks around each fly (5×5 grid per fly)
POOL_SIZE = 40          # pre-allocated mocap obstacle bodies
UNDERGROUND = -1000.0   # z for hidden bodies

# Materials: (name, rgba)
_MATERIALS = {
    'obs_bark':     (0.35, 0.25, 0.15, 1.0),
    'obs_leaf':     (0.15, 0.35, 0.10, 1.0),
    'obs_stone':    (0.30, 0.30, 0.32, 1.0),
    'obs_dirt':     (0.25, 0.20, 0.12, 1.0),
    'obs_berry':    (0.50, 0.05, 0.10, 1.0),
    'obs_mushroom': (0.60, 0.55, 0.40, 1.0),
    'obs_petal':    (0.70, 0.50, 0.60, 1.0),
    'obs_shadow':   (0.08, 0.08, 0.10, 1.0),
}

# Pool body shapes: (geom_type, size_tuple, material_name)
POOL_SHAPES = [
    ('box',      (2, 2, 3),    'obs_bark'),      # twig
    ('box',      (5, 1, 8),    'obs_bark'),      # branch (tall)
    ('capsule',  (1.5, 4),     'obs_leaf'),      # stem
    ('capsule',  (2, 6),       'obs_bark'),      # stick
    ('sphere',   (2,),         'obs_berry'),     # berry
    ('sphere',   (3.5,),       'obs_stone'),     # pebble
    ('sphere',   (5,),         'obs_shadow'),    # dark → GF escape trigger
    ('cylinder', (3, 5),       'obs_mushroom'),  # mushroom cap
    ('cylinder', (1, 8),       'obs_bark'),      # thin stem
    ('box',      (8, 8, 2),   'obs_leaf'),      # fallen leaf
]


# ═══════════════════════════════════════════════════════════════════════
# Chunk data container
# ═══════════════════════════════════════════════════════════════════════

class ChunkData:
    """Generated content for a single chunk."""
    __slots__ = ('obstacles', 'odor_sources', 'vibration_sources',
                 'taste_zones', 'pool_ids')

    def __init__(self, obstacles, odor_sources, vibration_sources, taste_zones):
        self.obstacles = obstacles            # list of (local_x, local_y, shape_idx)
        self.odor_sources = odor_sources      # list of OdorSource
        self.vibration_sources = vibration_sources  # list of VibrationSource
        self.taste_zones = taste_zones        # list of TasteZone
        self.pool_ids = []                    # assigned pool body indices


# ═══════════════════════════════════════════════════════════════════════
# ProceduralArena
# ═══════════════════════════════════════════════════════════════════════

class ProceduralArena(BaseArena):
    """Procedural Minecraft-like arena with dynamic chunks and obstacle pool.

    Parameters
    ----------
    world_seed : int
        Seed for deterministic chunk generation.
    ground_size : float
        Half-extent of the ground plane in mm.
    """

    def __init__(self, world_seed=42, ground_size=500):
        super().__init__()

        self.world_seed = world_seed
        self.ground_size = ground_size
        self.curr_time = 0.0

        # Fly positions (set externally before each sim.step)
        self._fly_positions = []

        # Chunk bookkeeping
        self._active_chunks = {}   # (cx, cy) -> ChunkData
        self._free_pool = list(range(POOL_SIZE))  # available body indices
        self._pool_bodies = []     # MJCF body handles (set in __init__)

        # ── Visual environment ──
        self.root_element.asset.add(
            'texture', type='skybox', builtin='gradient',
            rgb1=(0.35, 0.55, 0.8), rgb2=(0.85, 0.9, 1.0),
            width=512, height=512,
        )
        headlight = self.root_element.visual.headlight
        headlight.ambient = (0.35, 0.35, 0.4)
        headlight.diffuse = (0.7, 0.7, 0.65)
        headlight.specular = (0.3, 0.3, 0.3)

        self.root_element.worldbody.add(
            'light', name='sun', pos=(0, 0, 200), dir=(0.4, 0.3, -1.0),
            diffuse=(1.0, 0.95, 0.85), specular=(0.5, 0.5, 0.4),
            castshadow=True, directional=True,
        )

        # ── Ground plane ──
        ground_tex = self.root_element.asset.add(
            'texture', type='2d', builtin='checker', width=512, height=512,
            rgb1=(0.28, 0.38, 0.2), rgb2=(0.35, 0.42, 0.25),
        )
        ground_mat = self.root_element.asset.add(
            'material', name='ground_mat', texture=ground_tex,
            texrepeat=(30, 30), reflectance=0.05,
        )
        self.root_element.worldbody.add(
            'geom', type='plane', name='ground', material=ground_mat,
            size=[ground_size, ground_size, 1],
            friction=(1, 0.005, 0.0001), conaffinity=0,
        )
        self.friction = (1, 0.005, 0.0001)

        # ── Obstacle materials ──
        self._materials = {}
        for mat_name, rgba in _MATERIALS.items():
            self._materials[mat_name] = self.root_element.asset.add(
                'material', name=mat_name, rgba=rgba, reflectance=0.1,
            )

        # ── Pre-allocate obstacle pool (40 mocap bodies, underground) ──
        for i in range(POOL_SIZE):
            shape = POOL_SHAPES[i % len(POOL_SHAPES)]
            body = self.root_element.worldbody.add(
                'body', name=f'obs_{i}', mocap=True,
                pos=[0, 0, UNDERGROUND], gravcomp=1,
            )
            body.add(
                'geom', type=shape[0], size=list(shape[1]),
                material=self._materials[shape[2]],
                conaffinity=0, contype=0,
            )
            self._pool_bodies.append(body)

    # ──────────────────────────────────────────────────────────────────
    # BaseArena interface
    # ──────────────────────────────────────────────────────────────────

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return 0.0

    # ──────────────────────────────────────────────────────────────────
    # Fly position interface (called from two_flies main loop)
    # ──────────────────────────────────────────────────────────────────

    def set_fly_positions(self, positions):
        """Set current fly positions (list of [x,y,z] arrays)."""
        self._fly_positions = [np.asarray(p) for p in positions]

    # ──────────────────────────────────────────────────────────────────
    # Chunk coordinate helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _pos_to_chunk(x, y):
        """World coords → chunk grid coords."""
        return (int(np.floor(x / CHUNK_SIZE)),
                int(np.floor(y / CHUNK_SIZE)))

    @staticmethod
    def _chunk_center(cx, cy):
        """Chunk grid coords → world center."""
        return ((cx + 0.5) * CHUNK_SIZE,
                (cy + 0.5) * CHUNK_SIZE)

    # ──────────────────────────────────────────────────────────────────
    # Chunk generation (deterministic)
    # ──────────────────────────────────────────────────────────────────

    def _generate_chunk(self, cx, cy):
        """Create ChunkData with deterministic content for grid cell (cx, cy)."""
        seed = hash((self.world_seed, cx, cy)) & 0xFFFFFFFF
        rng = RandomState(seed)

        center_x, center_y = self._chunk_center(cx, cy)
        half = CHUNK_SIZE / 2.0

        # 1-3 obstacles
        n_obs = rng.randint(1, 4)
        obstacles = []
        for _ in range(n_obs):
            lx = rng.uniform(-half * 0.8, half * 0.8)
            ly = rng.uniform(-half * 0.8, half * 0.8)
            shape_idx = rng.randint(0, len(POOL_SHAPES))
            obstacles.append((center_x + lx, center_y + ly, shape_idx))

        # 30% chance: odor source
        odor_sources = []
        if rng.random() < 0.30:
            ox = center_x + rng.uniform(-half * 0.5, half * 0.5)
            oy = center_y + rng.uniform(-half * 0.5, half * 0.5)
            if rng.random() < 0.70:
                odor_sources.append(OdorSource(
                    position=[ox, oy, 1.0], odor_type='attractive',
                    amplitude=rng.uniform(0.4, 0.9),
                    spread=rng.uniform(15.0, 35.0),
                    label=f'food_{cx}_{cy}',
                ))
            else:
                odor_sources.append(OdorSource(
                    position=[ox, oy, 1.0], odor_type='repulsive',
                    amplitude=rng.uniform(0.5, 1.0),
                    spread=rng.uniform(10.0, 20.0),
                    label=f'danger_{cx}_{cy}',
                ))

        # 20% chance: vibration source
        vibration_sources = []
        if rng.random() < 0.20:
            vx = center_x + rng.uniform(-half * 0.5, half * 0.5)
            vy = center_y + rng.uniform(-half * 0.5, half * 0.5)
            freq = rng.choice([200.0, 300.0, 400.0])
            vibration_sources.append(VibrationSource(
                position=[vx, vy, 1.0], frequency=freq,
                amplitude=rng.uniform(0.5, 1.0),
                label=f'vib_{cx}_{cy}',
            ))

        # 15% chance: taste zone
        taste_zones = []
        if rng.random() < 0.15:
            tx = center_x + rng.uniform(-half * 0.4, half * 0.4)
            ty = center_y + rng.uniform(-half * 0.4, half * 0.4)
            taste = 'sugar' if rng.random() < 0.60 else 'bitter'
            taste_zones.append(TasteZone(
                center=[tx, ty],
                radius=rng.uniform(4.0, 10.0),
                taste=taste,
                label=f'{taste}_{cx}_{cy}',
            ))

        return ChunkData(obstacles, odor_sources, vibration_sources, taste_zones)

    # ──────────────────────────────────────────────────────────────────
    # Chunk activation / deactivation
    # ──────────────────────────────────────────────────────────────────

    def _desired_chunks(self):
        """Compute the set of chunk coords that should be active."""
        desired = set()
        for pos in self._fly_positions:
            cx, cy = self._pos_to_chunk(pos[0], pos[1])
            for dx in range(-ACTIVE_RADIUS, ACTIVE_RADIUS + 1):
                for dy in range(-ACTIVE_RADIUS, ACTIVE_RADIUS + 1):
                    desired.add((cx + dx, cy + dy))
        return desired

    def _deactivate_chunk(self, key, physics):
        """Send a chunk's pool bodies underground and return them to free list."""
        chunk = self._active_chunks.pop(key, None)
        if chunk is None:
            return
        for pid in chunk.pool_ids:
            physics.bind(self._pool_bodies[pid]).mocap_pos = [0, 0, UNDERGROUND]
            self._free_pool.append(pid)

    def _activate_chunk(self, key, physics):
        """Generate chunk content and place pool bodies."""
        cx, cy = key
        chunk = self._generate_chunk(cx, cy)

        for wx, wy, shape_idx in chunk.obstacles:
            if not self._free_pool:
                break  # pool exhausted — skip remaining
            # Pick a pool body whose shape matches (best effort: use shape_idx mod)
            pid = self._free_pool.pop()
            # Obstacle height: half the z-extent of its shape
            shape = POOL_SHAPES[pid % len(POOL_SHAPES)]
            if shape[0] in ('box',):
                z_half = shape[1][2]
            elif shape[0] in ('capsule',):
                z_half = shape[1][0] + shape[1][1]
            elif shape[0] in ('sphere',):
                z_half = shape[1][0]
            elif shape[0] in ('cylinder',):
                z_half = shape[1][1]
            else:
                z_half = 3.0
            physics.bind(self._pool_bodies[pid]).mocap_pos = [wx, wy, z_half]
            chunk.pool_ids.append(pid)

        self._active_chunks[key] = chunk

    # ──────────────────────────────────────────────────────────────────
    # step() — called by Simulation.step() every physics tick
    # ──────────────────────────────────────────────────────────────────

    def step(self, dt, physics, *args, **kwargs):
        """Update active chunks based on fly positions."""
        self.curr_time += dt

        if not self._fly_positions:
            return

        desired = self._desired_chunks()

        # Deactivate distant chunks
        to_remove = [k for k in self._active_chunks if k not in desired]
        for k in to_remove:
            self._deactivate_chunk(k, physics)

        # Activate new chunks
        for k in desired:
            if k not in self._active_chunks:
                self._activate_chunk(k, physics)

    # ──────────────────────────────────────────────────────────────────
    # Sensory source aggregation (consumed by two_flies main loop)
    # ──────────────────────────────────────────────────────────────────

    @property
    def all_odor_sources(self):
        """All odor sources from active chunks."""
        sources = []
        for chunk in self._active_chunks.values():
            sources.extend(chunk.odor_sources)
        return sources

    @property
    def all_vibration_sources(self):
        """All vibration sources from active chunks."""
        sources = []
        for chunk in self._active_chunks.values():
            sources.extend(chunk.vibration_sources)
        return sources

    @property
    def all_taste_zones(self):
        """All taste zones from active chunks."""
        zones = []
        for chunk in self._active_chunks.values():
            zones.extend(chunk.taste_zones)
        return zones

    # ──────────────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────────────

    @property
    def n_active_chunks(self):
        return len(self._active_chunks)

    @property
    def n_pool_used(self):
        return POOL_SIZE - len(self._free_pool)

    # ──────────────────────────────────────────────────────────────────
    # Respawn support (used by PhysicsWatchdog)
    # ──────────────────────────────────────────────────────────────────

    def get_safe_respawn_position(self, avoid_positions, min_distance=30.0):
        """Find a chunk center far from given positions.

        Searches in a spiral outward from the centroid of avoid_positions,
        returning the first chunk center that is at least min_distance from
        all positions in avoid_positions.
        """
        if not avoid_positions:
            return np.array([0.0, 0.0, 0.6])

        centroid = np.mean([p[:2] for p in avoid_positions], axis=0)
        cx0, cy0 = self._pos_to_chunk(centroid[0], centroid[1])

        # Spiral search
        for radius in range(1, 20):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # only perimeter
                    wx, wy = self._chunk_center(cx0 + dx, cy0 + dy)
                    pos = np.array([wx, wy])
                    far_enough = all(
                        np.linalg.norm(pos - np.asarray(ap)[:2]) >= min_distance
                        for ap in avoid_positions
                    )
                    if far_enough:
                        return np.array([wx, wy, 0.6])

        # Fallback: offset from centroid
        return np.array([centroid[0] + 60.0, centroid[1] + 60.0, 0.6])
