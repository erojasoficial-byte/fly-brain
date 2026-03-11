#!/usr/bin/env python3
"""
Brain Monitor v2 — Futuristic real-time dorsal brain visualization.

Sci-fi / Black Mirror aesthetic: gaussian glow, animated particles,
dashed connections, hex grid, scanlines, pulsing regions.

Runs in a separate process (pygame) to avoid GL context conflicts with MuJoCo.
Receives neural activity data via multiprocessing.Queue and renders a glowing
dorsal brain map with HUD overlay.

Usage:
    Launched automatically by fly_embodied.py --monitor
"""

import multiprocessing as mp
import math
import time
import random

# ---------------------------------------------------------------------------
# Neon color palette
# ---------------------------------------------------------------------------
COL_BG = (5, 5, 20)
COL_VISUAL = (0, 255, 200)
COL_LOOMING = (255, 120, 0)
COL_ESCAPE = (255, 20, 60)
COL_MOTOR = (0, 255, 100)
COL_BACKWARD = (180, 60, 255)
COL_GROOM = (255, 200, 0)
COL_FEED = (255, 80, 180)
COL_JO_TOUCH = (0, 200, 255)     # cyan-blue for mechanosensory
COL_JO_SOUND = (120, 80, 255)    # indigo for auditory
COL_SUGAR = (80, 255, 80)        # bright green for sugar GRN
COL_BITTER = (255, 80, 80)       # bright red for bitter GRN
COL_OR_ATT = (120, 255, 60)     # lime green for attractive ORN
COL_OR_REP = (255, 60, 200)     # hot pink for repulsive ORN
COL_WING = (255, 220, 50)       # gold for wing song
COL_FLIGHT = (50, 200, 255)     # cyan for flight system
COL_HUD = (0, 200, 220)
COL_TITLE = (180, 230, 255)
COL_HEX = (15, 20, 50)
COL_BRAIN_CENTER = (18, 22, 50)

# Consciousness index gradient: black → blue → green → white
CI_GRADIENT = [
    (0.0,  (5, 5, 20)),       # black/dark
    (0.25, (20, 60, 200)),    # blue
    (0.5,  (0, 200, 100)),    # green
    (0.75, (100, 255, 200)),  # bright green-white
    (1.0,  (240, 255, 255)),  # white
]
COL_CI_PEAK = (255, 215, 0)   # gold for peak markers

# ---------------------------------------------------------------------------
# Brain regions: (name, x, y, radius, color_rgb, group)
# ---------------------------------------------------------------------------
REGIONS = [
    # Visual pathway
    ('Retina_L',  120, 100, 35, COL_VISUAL,   'visual'),
    ('Retina_R',  680, 100, 35, COL_VISUAL,   'visual'),
    ('T2_L',      185, 155, 22, COL_VISUAL,   'visual'),
    ('T2_R',      615, 155, 22, COL_VISUAL,   'visual'),
    # Looming detectors
    ('LC4_L',     215, 200, 20, COL_LOOMING,  'looming'),
    ('LC4_R',     585, 200, 20, COL_LOOMING,  'looming'),
    ('LPLC2_L',   245, 245, 20, COL_LOOMING,  'looming'),
    ('LPLC2_R',   555, 245, 20, COL_LOOMING,  'looming'),
    # Johnston's Organ — touch (antenna area, dorsal)
    ('JO_tch_L',  155, 60,  16, COL_JO_TOUCH, 'jo_touch'),
    ('JO_tch_R',  645, 60,  16, COL_JO_TOUCH, 'jo_touch'),
    # Johnston's Organ — sound (antenna area, slightly lower)
    ('JO_snd_L',  170, 90,  14, COL_JO_SOUND, 'jo_sound'),
    ('JO_snd_R',  630, 90,  14, COL_JO_SOUND, 'jo_sound'),
    # Olfactory — attractive ORN (DM1/Or42b, antennal lobe)
    ('OR_att_L',  255, 55,  14, COL_OR_ATT,   'olfactory'),
    ('OR_att_R',  545, 55,  14, COL_OR_ATT,   'olfactory'),
    # Olfactory — repulsive ORN (DA2/Or56a, antennal lobe)
    ('OR_rep_L',  290, 75,  12, COL_OR_REP,   'olfactory'),
    ('OR_rep_R',  510, 75,  12, COL_OR_REP,   'olfactory'),
    # Gustatory — sugar / bitter GRNs (SEZ input)
    ('Sugar_GRN', 360, 385, 14, COL_SUGAR,    'gustatory'),
    ('Bitter_GRN',440, 385, 14, COL_BITTER,   'gustatory'),
    # Giant Fiber — escape command
    ('GF',        400, 290, 28, COL_ESCAPE,   'escape'),
    # Motor — turning
    ('DNa_L',     320, 350, 16, COL_MOTOR,    'motor'),
    ('DNa_R',     480, 350, 16, COL_MOTOR,    'motor'),
    # Motor — forward
    ('P9_L',      340, 420, 16, COL_MOTOR,    'motor'),
    ('P9_R',      460, 420, 16, COL_MOTOR,    'motor'),
    # Backward
    ('MDN',       400, 460, 16, COL_BACKWARD, 'backward'),
    # Grooming
    ('aDN1',      300, 460, 14, COL_GROOM,    'groom'),
    # Feeding
    ('MN9',       500, 460, 14, COL_FEED,     'feed'),
    # Wing song
    ('Wing_Song', 400, 510, 16, COL_WING,     'wing'),
    # Flight
    ('Flight',    400, 555, 18, COL_FLIGHT,    'flight'),
]

_REGION_IDX = {r[0]: i for i, r in enumerate(REGIONS)}

CONNECTIONS = [
    ('Retina_L', 'T2_L'),
    ('Retina_R', 'T2_R'),
    ('T2_L',     'LC4_L'),
    ('T2_R',     'LC4_R'),
    ('LC4_L',    'LPLC2_L'),
    ('LC4_R',    'LPLC2_R'),
    ('LPLC2_L',  'GF'),
    ('LPLC2_R',  'GF'),
    ('GF',       'DNa_L'),
    ('GF',       'DNa_R'),
    ('DNa_L',    'P9_L'),
    ('DNa_R',    'P9_R'),
    ('GF',       'MDN'),
    # JO touch → grooming
    ('JO_tch_L', 'aDN1'),
    ('JO_tch_R', 'aDN1'),
    # JO touch → escape (strong tactile)
    ('JO_tch_L', 'GF'),
    ('JO_tch_R', 'GF'),
    # JO sound → turning (orientation)
    ('JO_snd_L', 'DNa_L'),
    ('JO_snd_R', 'DNa_R'),
    # Sugar GRN → feeding (MN9) + approach (P9)
    ('Sugar_GRN', 'MN9'),
    ('Sugar_GRN', 'P9_L'),
    # Bitter GRN → escape (GF) + backward (MDN)
    ('Bitter_GRN', 'GF'),
    ('Bitter_GRN', 'MDN'),
    # Attractive ORN → approach (P9) + turning (DNa)
    ('OR_att_L', 'P9_L'),
    ('OR_att_R', 'P9_R'),
    ('OR_att_L', 'DNa_L'),
    ('OR_att_R', 'DNa_R'),
    # Repulsive ORN → escape (GF)
    ('OR_rep_L', 'GF'),
    ('OR_rep_R', 'GF'),
    # Wing song ← motor triggers
    ('MN9',      'Wing_Song'),
    ('GF',       'Wing_Song'),
    # Wing song → JO self-hearing
    ('Wing_Song', 'JO_snd_L'),
    ('Wing_Song', 'JO_snd_R'),
    # Flight: GF triggers takeoff, DNa controls direction, P9 thrust
    ('GF',        'Flight'),
    ('DNa_L',     'Flight'),
    ('DNa_R',     'Flight'),
    ('P9_L',      'Flight'),
]

_DATA_KEY_MAP = {
    'bright_left':  'Retina_L',
    'bright_right': 'Retina_R',
    't2_left':      'T2_L',
    't2_right':     'T2_R',
    'lc4_left':     'LC4_L',
    'lc4_right':    'LC4_R',
    'lplc2_left':   'LPLC2_L',
    'lplc2_right':  'LPLC2_R',
    'dn_escape':    'GF',
    'dn_turn_L':    'DNa_L',
    'dn_turn_R':    'DNa_R',
    'dn_forward':   'P9_L',
    'dn_forward_R': 'P9_R',
    'dn_backward':  'MDN',
    'dn_groom':     'aDN1',
    'dn_feed':      'MN9',
    'jo_touch_L':   'JO_tch_L',
    'jo_touch_R':   'JO_tch_R',
    'jo_sound_L':   'JO_snd_L',
    'jo_sound_R':   'JO_snd_R',
    'sugar_level':  'Sugar_GRN',
    'bitter_level': 'Bitter_GRN',
    'or_att_L':     'OR_att_L',
    'or_att_R':     'OR_att_R',
    'or_rep_L':     'OR_rep_L',
    'or_rep_R':     'OR_rep_R',
    'wing_level':   'Wing_Song',
    'flight_level': 'Flight',
}

SIDEBAR_BARS = [
    ('FWD', 'dn_forward',  COL_MOTOR),
    ('ESC', 'dn_escape',   COL_ESCAPE),
    ('TRN', 'dn_turn_L',   COL_MOTOR),
    ('GRM', 'dn_groom',    COL_GROOM),
    ('BKW', 'dn_backward', COL_BACKWARD),
    ('FED', 'dn_feed',     COL_FEED),
    ('TCH', 'jo_contact',  COL_JO_TOUCH),
    ('SND', 'jo_sound',    COL_JO_SOUND),
    ('SGR', 'sugar_level', COL_SUGAR),
    ('BTR', 'bitter_level',COL_BITTER),
    ('ATT', 'or_attractive',COL_OR_ATT),
    ('REP', 'or_repulsive',COL_OR_REP),
    ('WNG', 'wing_level',  COL_WING),
    ('FLT', 'flight_level', COL_FLIGHT),
]

MODE_COLORS = {
    'walking':  COL_MOTOR,
    'escape':   COL_ESCAPE,
    'grooming': COL_GROOM,
    'feeding':  COL_FEED,
    'flight':   COL_FLIGHT,
}

GLOW_LEVELS = 16


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ============================================================================
# Glow Cache — gaussian numpy pre-render
# ============================================================================

class GlowCache:
    """Pre-renders gaussian glow textures at 16 intensity levels per region."""

    def __init__(self, pygame_module):
        self.pg = pygame_module
        self.cache = {}  # (region_idx, level) -> Surface

    def _render_glow(self, radius, color, intensity):
        """Render a gaussian glow surface with 3 layers."""
        import numpy as np
        pg = self.pg

        # Surface size: enough for the outermost halo
        half = int(radius * 4) + 4
        size = half * 2
        surf = pg.Surface((size, size), pg.SRCALPHA)
        if intensity < 0.005:
            return surf

        # Build coordinate grid centered on (half, half)
        y_coords, x_coords = np.mgrid[0:size, 0:size]
        dist_sq = (x_coords - half).astype(np.float32) ** 2 + \
                  (y_coords - half).astype(np.float32) ** 2

        r, g, b = color

        # Accumulate RGB + alpha in float
        out_r = np.zeros((size, size), dtype=np.float32)
        out_g = np.zeros((size, size), dtype=np.float32)
        out_b = np.zeros((size, size), dtype=np.float32)
        out_a = np.zeros((size, size), dtype=np.float32)

        # Layer 1: Outer halo (σ = r×1.8)
        sigma1 = radius * 1.8
        gauss1 = np.exp(-dist_sq / (2.0 * sigma1 * sigma1))
        a1 = gauss1 * 0.25 * intensity
        out_r += r * a1
        out_g += g * a1
        out_b += b * a1
        out_a += a1 * 255

        # Layer 2: Inner glow (σ = r×0.8)
        sigma2 = radius * 0.8
        gauss2 = np.exp(-dist_sq / (2.0 * sigma2 * sigma2))
        a2 = gauss2 * 0.55 * intensity
        out_r += r * a2
        out_g += g * a2
        out_b += b * a2
        out_a += a2 * 255

        # Layer 3: Hot core (σ = r×0.3), shifts toward white at high intensity
        sigma3 = radius * 0.3
        gauss3 = np.exp(-dist_sq / (2.0 * sigma3 * sigma3))
        a3 = gauss3 * 0.9 * intensity
        white_mix = intensity * 0.6  # how much to blend toward white
        cr = r + (255 - r) * white_mix
        cg = g + (255 - g) * white_mix
        cb = b + (255 - b) * white_mix
        out_r += cr * a3
        out_g += cg * a3
        out_b += cb * a3
        out_a += a3 * 255

        # Clamp and assemble RGBA
        out_r = np.clip(out_r, 0, 255).astype(np.uint8)
        out_g = np.clip(out_g, 0, 255).astype(np.uint8)
        out_b = np.clip(out_b, 0, 255).astype(np.uint8)
        out_a = np.clip(out_a, 0, 255).astype(np.uint8)

        # Stack into (size, size, 4) RGBA array
        rgba = np.stack([out_r, out_g, out_b, out_a], axis=-1)

        # Blit numpy array to surface
        # pygame surfarray wants (width, height, 4) = transposed
        pg.surfarray.blit_array(surf, rgba[:, :, :3].transpose(1, 0, 2))
        # Set alpha channel via separate pixel_alpha array
        alpha_surf = pg.surfarray.pixels_alpha(surf)
        alpha_surf[:] = out_a.T
        del alpha_surf  # unlock surface

        return surf

    def build(self):
        """Pre-render all region x intensity combinations."""
        for idx, (name, x, y, radius, color, group) in enumerate(REGIONS):
            for level in range(GLOW_LEVELS + 1):
                intensity = level / GLOW_LEVELS
                surf = self._render_glow(radius, color, intensity)
                self.cache[(idx, level)] = surf

    def get(self, region_idx, intensity):
        """Get pre-rendered surface for region at given intensity [0-1]."""
        level = int(round(_clamp(intensity) * GLOW_LEVELS))
        return self.cache.get((region_idx, level))


# ============================================================================
# Particle System
# ============================================================================

class ConnectionParticle:
    """A single luminous particle traveling along a connection."""
    __slots__ = ('conn_idx', 't', 'speed', 'life', 'max_life')

    def __init__(self, conn_idx, speed):
        self.conn_idx = conn_idx
        self.t = 0.0  # 0..1 progress along connection
        self.speed = speed  # units per second (t goes 0->1)
        self.life = 0.0
        self.max_life = 1.0 / max(speed, 0.01)


class ParticleSystem:
    """Manages particles flowing along neural connections."""

    MAX_PER_CONN = 6
    MAX_RATE = 4.0  # max spawns/sec per connection

    def __init__(self):
        self.particles = []  # list of ConnectionParticle
        self._spawn_accum = {}  # conn_idx -> accumulated spawn fraction

    def update(self, dt, conn_intensities):
        """Spawn new particles and advance existing ones."""
        # Spawn
        for ci, intensity in enumerate(conn_intensities):
            if intensity < 0.05:
                self._spawn_accum[ci] = 0.0
                continue

            rate = intensity * self.MAX_RATE
            acc = self._spawn_accum.get(ci, 0.0) + rate * dt
            count_on_conn = sum(1 for p in self.particles if p.conn_idx == ci)

            while acc >= 1.0 and count_on_conn < self.MAX_PER_CONN:
                speed = 0.6 + 0.8 * intensity + random.random() * 0.3
                p = ConnectionParticle(ci, speed)
                self.particles.append(p)
                acc -= 1.0
                count_on_conn += 1

            self._spawn_accum[ci] = acc

        # Update positions, remove dead
        alive = []
        for p in self.particles:
            p.t += p.speed * dt
            p.life += dt
            if p.t < 1.0:
                alive.append(p)
        self.particles = alive

    def draw(self, screen, pg, connections):
        """Draw all particles as small glowing dots."""
        for p in self.particles:
            src_name, dst_name = connections[p.conn_idx]
            src = REGIONS[_REGION_IDX[src_name]]
            dst = REGIONS[_REGION_IDX[dst_name]]

            # Interpolate position
            x = src[1] + (dst[1] - src[1]) * p.t
            y = src[2] + (dst[2] - src[2]) * p.t

            # Fade in/out at endpoints
            fade = 1.0
            if p.t < 0.15:
                fade = p.t / 0.15
            elif p.t > 0.85:
                fade = (1.0 - p.t) / 0.15
            fade = _clamp(fade)

            # Color from source region
            color = src[4]
            alpha = int(220 * fade)
            r2 = 3 if fade > 0.5 else 2

            # Draw glow dot (outer + core)
            glow_surf = pg.Surface((12, 12), pg.SRCALPHA)
            pg.draw.circle(glow_surf,
                           (color[0], color[1], color[2], alpha // 3),
                           (6, 6), 5)
            pg.draw.circle(glow_surf,
                           (min(color[0] + 80, 255),
                            min(color[1] + 80, 255),
                            min(color[2] + 80, 255), alpha),
                           (6, 6), r2)
            screen.blit(glow_surf, (int(x) - 6, int(y) - 6),
                        special_flags=pg.BLEND_ADD)


# ============================================================================
# Brain Renderer — all drawing logic (futuristic overhaul)
# ============================================================================

class BrainRenderer:
    """Renders the dorsal brain view with gaussian glow, particles, and HUD."""

    WIDTH = 800
    HEIGHT = 600

    # Smoothing
    TAU_SMOOTH = 0.12  # exponential smoothing time constant (seconds)
    PULSE_FREQ = 2.5   # Hz breathing animation
    PULSE_AMP = 0.08   # ±8% intensity modulation

    def __init__(self, pygame_module):
        self.pg = pygame_module
        self.screen = None
        self.clock = None
        self.font = None
        self.font_sm = None
        self.font_title = None
        self.glow_cache = GlowCache(pygame_module)

        # Audio: wing song tones
        self._audio_tones = {}     # freq -> pygame.Sound
        self._current_tone_freq = 0
        self._audio_ready = False
        self.particles = ParticleSystem()

        n = len(REGIONS)
        self.raw_intensities = [0.0] * n
        self.smoothed = [0.0] * n
        self.display_intensity = [0.0] * n
        self.phase = [random.random() * math.tau for _ in range(n)]

        self.data = {}
        self.frame_time = 0.0     # monotonic seconds
        self.last_time = None
        self.dash_offset = 0.0    # animated dash offset

        # Pre-rendered surfaces (built in init_display)
        self._hex_grid = None
        self._brain_sil = None
        self._scanlines = None
        self._title_glow = None

        # Pre-rendered sidebar bar gradient surfaces
        self._bar_gradients = {}

    def init_display(self):
        """Initialize pygame display, fonts, caches, pre-rendered assets."""
        pg = self.pg
        self.screen = pg.display.set_mode(
            (self.WIDTH, self.HEIGHT), pg.DOUBLEBUF)
        pg.display.set_caption('Drosophila Brain Monitor')
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont('consolas', 14)
        self.font_sm = pg.font.SysFont('consolas', 11)
        self.font_title = pg.font.SysFont('consolas', 18, bold=True)

        self.glow_cache.build()
        self._build_hex_grid()
        self._build_brain_silhouette()
        self._build_scanline_overlay()
        self._build_title_glow()
        self._build_bar_gradients()
        self._init_audio()

        self.last_time = time.monotonic()

    # ── Pre-render: Hex Grid ──────────────────────────────────────────────

    def _build_hex_grid(self):
        """Pre-render a subtle hexagonal grid pattern."""
        pg = self.pg
        surf = pg.Surface((self.WIDTH, self.HEIGHT), pg.SRCALPHA)

        hex_r = 20  # hex radius
        w = hex_r * 2
        h = int(hex_r * math.sqrt(3))
        color = (*COL_HEX, 35)  # subtle alpha

        for row in range(-1, self.HEIGHT // h + 2):
            for col in range(-1, self.WIDTH // w + 2):
                cx = int(col * w * 0.75)
                cy = int(row * h + (col % 2) * h * 0.5)
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i + math.pi / 6
                    px = cx + int(hex_r * math.cos(angle))
                    py = cy + int(hex_r * math.sin(angle))
                    points.append((px, py))
                if len(points) == 6:
                    pg.draw.polygon(surf, color, points, 1)

        self._hex_grid = surf

    # ── Pre-render: Brain Silhouette ──────────────────────────────────────

    def _build_brain_silhouette(self):
        """Pre-render brain silhouette with radial gradient via numpy."""
        import numpy as np
        pg = self.pg

        surf = pg.Surface((self.WIDTH, self.HEIGHT), pg.SRCALPHA)

        # Three elliptical regions: left optic, right optic, central brain
        ellipses = [
            (200, 160, 150, 110),  # left optic lobe (cx, cy, rx, ry)
            (600, 160, 150, 110),  # right optic lobe
            (400, 350, 170, 160),  # central brain
        ]

        y_coords, x_coords = np.mgrid[0:self.HEIGHT, 0:self.WIDTH]
        x_f = x_coords.astype(np.float32)
        y_f = y_coords.astype(np.float32)

        # Combined mask: union of ellipses with soft falloff
        combined = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.float32)
        for cx, cy, rx, ry in ellipses:
            dist = ((x_f - cx) / rx) ** 2 + ((y_f - cy) / ry) ** 2
            # Smooth falloff: 1.0 inside, fades to 0 outside
            mask = np.clip(1.0 - (dist - 0.7) * 2.5, 0.0, 1.0)
            combined = np.maximum(combined, mask)

        # Radial gradient: center brighter
        center_x, center_y = 400, 300
        global_dist = np.sqrt((x_f - center_x) ** 2 +
                              (y_f - center_y) ** 2)
        radial = np.clip(1.0 - global_dist / 350.0, 0.2, 1.0)

        alpha = (combined * radial * 45).astype(np.uint8)  # subtle

        # Color: COL_BRAIN_CENTER
        r_arr = np.full_like(alpha, COL_BRAIN_CENTER[0])
        g_arr = np.full_like(alpha, COL_BRAIN_CENTER[1])
        b_arr = np.full_like(alpha, COL_BRAIN_CENTER[2])

        rgba = np.stack([r_arr, g_arr, b_arr, alpha], axis=-1)
        pg.surfarray.blit_array(surf, rgba[:, :, :3].transpose(1, 0, 2))
        a_view = pg.surfarray.pixels_alpha(surf)
        a_view[:] = alpha.T
        del a_view

        self._brain_sil = surf

    # ── Pre-render: Scanline Overlay ──────────────────────────────────────

    def _build_scanline_overlay(self):
        """Pre-render CRT scanline effect."""
        pg = self.pg
        surf = pg.Surface((self.WIDTH, self.HEIGHT), pg.SRCALPHA)

        for y in range(0, self.HEIGHT, 3):
            pg.draw.line(surf, (0, 0, 0, 18), (0, y), (self.WIDTH, y), 1)

        self._scanlines = surf

    # ── Pre-render: Title Glow ────────────────────────────────────────────

    def _build_title_glow(self):
        """Pre-render title text with glow halo."""
        pg = self.pg
        text = 'DROSOPHILA BRAIN MONITOR'
        base = self.font_title.render(text, True, COL_TITLE)
        w, h = base.get_size()
        pad = 6
        surf = pg.Surface((w + pad * 2, h + pad * 2), pg.SRCALPHA)

        # Multi-offset glow
        glow_color = (COL_TITLE[0] // 3, COL_TITLE[1] // 3,
                      COL_TITLE[2] // 3)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                glow = self.font_title.render(text, True, glow_color)
                surf.blit(glow, (pad + dx, pad + dy))

        surf.blit(base, (pad, pad))
        self._title_glow = surf

    # ── Pre-render: Bar Gradients ─────────────────────────────────────────

    def _build_bar_gradients(self):
        """Pre-render horizontal gradient bars for sidebar."""
        pg = self.pg
        bar_w = 70
        bar_h = 12

        for label, key, color in SIDEBAR_BARS:
            surf = pg.Surface((bar_w, bar_h), pg.SRCALPHA)
            for x in range(bar_w):
                t = x / bar_w
                r = int(color[0] * (0.3 + 0.7 * t))
                g = int(color[1] * (0.3 + 0.7 * t))
                b = int(color[2] * (0.3 + 0.7 * t))
                a = int(180 + 75 * t)
                pg.draw.line(surf, (min(r, 255), min(g, 255),
                                    min(b, 255), min(a, 255)),
                             (x, 0), (x, bar_h - 1), 1)
            self._bar_gradients[key] = surf

    # ── Audio: Wing Song Tones ───────────────────────────────────────────

    def _init_audio(self):
        """Initialize pygame.mixer and pre-generate wing song tones."""
        pg = self.pg
        try:
            pg.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            import numpy as np

            sample_rate = 22050
            duration = 0.5  # seconds per tone loop chunk
            n_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, n_samples, endpoint=False)

            for freq in [160, 200, 400]:
                # Generate sine wave, low volume (10% amplitude)
                mono = (np.sin(2.0 * np.pi * freq * t) * 3276).astype(
                    np.int16)
                # Stereo: duplicate to 2 channels (N, 2)
                wave = np.column_stack([mono, mono])
                sound = pg.sndarray.make_sound(wave)
                self._audio_tones[freq] = sound

            self._audio_ready = True
            print("[BrainMonitor] Audio tones ready (160/200/400 Hz)",
                  flush=True)
        except Exception as e:
            print(f"[BrainMonitor] Audio init failed: {e}", flush=True)
            self._audio_ready = False

    def _update_audio(self, data):
        """Play or stop wing song audio based on current data."""
        if not self._audio_ready:
            return
        freq = int(data.get('wing_freq', 0))
        if freq == self._current_tone_freq:
            return  # no change

        # Stop current tone
        if self._current_tone_freq > 0:
            tone = self._audio_tones.get(self._current_tone_freq)
            if tone:
                tone.stop()

        # Play new tone
        self._current_tone_freq = freq
        if freq > 0:
            tone = self._audio_tones.get(freq)
            if tone:
                tone.play(loops=-1)  # loop indefinitely

    # ── Data Update ───────────────────────────────────────────────────────

    def update_data(self, data):
        """Update neural data from main process."""
        self.data = data
        self._compute_raw_intensities()
        self._update_audio(data)

    def _compute_raw_intensities(self):
        """Map data dict values to raw region intensities [0-1]."""
        d = self.data
        for key, region_name in _DATA_KEY_MAP.items():
            idx = _REGION_IDX.get(region_name)
            if idx is not None:
                val = d.get(key, 0.0)
                if region_name.startswith('Retina'):
                    dark_key = ('dark_omm_left' if 'left' in key
                                else 'dark_omm_right')
                    dark_count = d.get(dark_key, 0)
                    self.raw_intensities[idx] = _clamp(dark_count / 100.0)
                else:
                    self.raw_intensities[idx] = _clamp(float(val))

        # P9_R mirrors forward drive
        fwd = d.get('dn_forward', 0.0)
        idx_r = _REGION_IDX.get('P9_R')
        if idx_r is not None:
            self.raw_intensities[idx_r] = _clamp(float(fwd))

    # ── Smoothing & Pulse ─────────────────────────────────────────────────

    def _smooth_intensities(self, dt):
        """Exponential smoothing toward raw intensities."""
        if dt <= 0:
            return
        alpha = 1.0 - math.exp(-dt / self.TAU_SMOOTH)
        for i in range(len(REGIONS)):
            self.smoothed[i] += (self.raw_intensities[i] -
                                 self.smoothed[i]) * alpha

    def _compute_pulse(self):
        """Apply sinusoidal pulse modulation to smoothed intensities."""
        t = self.frame_time
        for i in range(len(REGIONS)):
            base = self.smoothed[i]
            if base > 0.02:
                pulse = 1.0 + self.PULSE_AMP * math.sin(
                    math.tau * self.PULSE_FREQ * t + self.phase[i])
                self.display_intensity[i] = _clamp(base * pulse)
            else:
                self.display_intensity[i] = base

    # ── Consciousness Index Visualization ────────────────────────────────

    @staticmethod
    def _ci_color(ci):
        """Interpolate CI gradient for a value in [0, 1]."""
        ci = max(0.0, min(1.0, ci))
        for i in range(len(CI_GRADIENT) - 1):
            t0, c0 = CI_GRADIENT[i]
            t1, c1 = CI_GRADIENT[i + 1]
            if ci <= t1:
                f = (ci - t0) / (t1 - t0) if t1 > t0 else 0.0
                return (
                    int(c0[0] + (c1[0] - c0[0]) * f),
                    int(c0[1] + (c1[1] - c0[1]) * f),
                    int(c0[2] + (c1[2] - c0[2]) * f),
                )
        return CI_GRADIENT[-1][1]

    def _draw_consciousness(self):
        """Draw CI timeline graph and value bar at top of screen (y=38)."""
        d = self.data
        pg = self.pg
        ci = d.get('consciousness_ci', 0.0)
        timeline = d.get('consciousness_timeline', [])

        if ci == 0.0 and not timeline:
            return  # no consciousness data yet

        y_base = 38
        graph_w = 420
        graph_h = 30
        bar_x = 440
        bar_w = 340
        bar_h = 16

        # ── Timeline graph (left side) ──
        if len(timeline) > 1:
            # Background
            pg.draw.rect(self.screen, (8, 10, 25),
                         (10, y_base, graph_w, graph_h))
            pg.draw.rect(self.screen, (30, 40, 70),
                         (10, y_base, graph_w, graph_h), 1)

            # Plot CI timeline
            n = len(timeline)
            step = max(1, graph_w / max(n - 1, 1))
            points = []
            for i, v in enumerate(timeline):
                x = 10 + int(i * step)
                y = y_base + graph_h - int(v * graph_h * 0.9) - 2
                y = max(y_base + 1, min(y_base + graph_h - 1, y))
                points.append((x, y))

            if len(points) >= 2:
                # Draw filled area
                fill_points = list(points) + [
                    (points[-1][0], y_base + graph_h - 1),
                    (points[0][0], y_base + graph_h - 1)]
                # Draw line on top
                for i in range(len(points) - 1):
                    color = self._ci_color(timeline[min(i, len(timeline) - 1)])
                    pg.draw.line(self.screen, color,
                                 points[i], points[i + 1], 2)

            # Peak markers (gold stars)
            peaks = d.get('consciousness_peaks', [])
            for step_val, peak_ci in peaks:
                # Find approximate x position
                for i, e in enumerate(timeline):
                    if abs(e - peak_ci) < 0.01:
                        x = 10 + int(i * (graph_w / max(len(timeline) - 1, 1)))
                        y = y_base + graph_h - int(peak_ci * graph_h * 0.9) - 2
                        y = max(y_base + 2, y)
                        pg.draw.circle(self.screen, COL_CI_PEAK, (x, y), 3)
                        break

            # Label
            lbl = self.font_sm.render('CI TIMELINE', True, (80, 100, 140))
            self.screen.blit(lbl, (12, y_base + 1))

        # ── CI value bar (right side) ──
        ci_color = self._ci_color(ci)

        # Label
        ci_txt = self.font.render(f'CONSCIOUSNESS: {ci:.3f}', True, ci_color)
        self.screen.blit(ci_txt, (bar_x, y_base))

        # Bar background
        bar_y = y_base + 18
        pg.draw.rect(self.screen, (8, 10, 25),
                     (bar_x, bar_y, bar_w, bar_h))
        pg.draw.rect(self.screen, (30, 40, 70),
                     (bar_x, bar_y, bar_w, bar_h), 1)

        # Bar fill with gradient
        fill_w = int(bar_w * min(ci, 1.0))
        if fill_w > 0:
            for x in range(fill_w):
                t = x / bar_w
                c = self._ci_color(t)
                pg.draw.line(self.screen, c,
                             (bar_x + x, bar_y + 1),
                             (bar_x + x, bar_y + bar_h - 2), 1)

    def _draw_consciousness_sidebar(self):
        """Draw PHI/GWT/SLF/CMP bars below existing sidebar."""
        d = self.data
        pg = self.pg
        ci = d.get('consciousness_ci', 0.0)

        if ci == 0.0 and d.get('consciousness_phi', 0.0) == 0.0:
            return  # no data yet

        x_start = 710
        bar_w = 70
        bar_h = 12

        # Position below existing sidebar (14 bars × 20px + header)
        y = 50 + len(SIDEBAR_BARS) * (bar_h + 8) + 20

        # Header
        header = self.font_sm.render('CONSCIOUSNESS', True, COL_HUD)
        self.screen.blit(header, (x_start, y - 14))
        y += 4

        metrics = [
            ('PHI', d.get('consciousness_phi', 0.0)),
            ('GWT', d.get('consciousness_gw', 0.0)),
            ('SLF', d.get('consciousness_self', 0.0)),
            ('CMP', d.get('consciousness_cmplx', 0.0)),
        ]

        for label, val in metrics:
            val = max(0.0, min(1.0, val))
            color = self._ci_color(val)

            # Label
            lbl_color = color if val > 0.05 else (60, 65, 90)
            lbl = self.font_sm.render(label, True, lbl_color)
            self.screen.blit(lbl, (x_start, y))

            # Bar background
            bx = x_start + 32
            pg.draw.rect(self.screen, (12, 14, 28),
                         (bx, y, bar_w, bar_h))
            pg.draw.rect(self.screen, (30, 35, 60),
                         (bx, y, bar_w, bar_h), 1)

            # Bar fill
            fill_w = int(bar_w * val)
            if fill_w > 0:
                for x in range(fill_w):
                    t = x / bar_w
                    c = self._ci_color(t)
                    pg.draw.line(self.screen, c,
                                 (bx + x, y + 1),
                                 (bx + x, y + bar_h - 2), 1)

            # Value text
            val_color = color if val > 0.1 else (55, 60, 85)
            val_txt = self.font_sm.render(f'{val:.2f}', True, val_color)
            self.screen.blit(val_txt, (bx + bar_w + 4, y))

            y += bar_h + 8

    # ── Render Frame ──────────────────────────────────────────────────────

    def render_frame(self):
        """Render one complete frame with the full pipeline."""
        # Timing
        now = time.monotonic()
        dt = now - self.last_time if self.last_time else 1.0 / 30.0
        dt = min(dt, 0.1)  # cap at 100ms
        self.last_time = now
        self.frame_time += dt
        self.dash_offset += dt * 40.0  # dash animation speed

        # 1. Smooth & pulse
        self._smooth_intensities(dt)
        self._compute_pulse()

        # 2. Background
        self.screen.fill(COL_BG)

        # 3. Hex grid
        self.screen.blit(self._hex_grid, (0, 0))

        # 4. Brain silhouette
        self.screen.blit(self._brain_sil, (0, 0))

        # 5. Dashed connections
        self._draw_connections()

        # 6. Particles
        conn_intensities = self._get_conn_intensities()
        self.particles.update(dt, conn_intensities)
        self.particles.draw(self.screen, self.pg, CONNECTIONS)

        # 7. Region glows
        self._draw_regions()

        # 8. Scanlines
        self.screen.blit(self._scanlines, (0, 0),
                         special_flags=self.pg.BLEND_RGBA_SUB)

        # 9. HUD + sidebar
        self._draw_hud()
        self._draw_sidebar()

        # 10. Consciousness overlay (if data present)
        self._draw_consciousness()
        self._draw_consciousness_sidebar()

        # 11. Flip
        self.pg.display.flip()

    def _get_conn_intensities(self):
        """Get source-region intensity for each connection."""
        result = []
        for src_name, dst_name in CONNECTIONS:
            src_idx = _REGION_IDX[src_name]
            result.append(self.display_intensity[src_idx])
        return result

    # ── Connections (animated dashes) ─────────────────────────────────────

    def _draw_connections(self):
        """Draw animated dashed lines between connected regions."""
        pg = self.pg
        dash_len = 8
        gap_len = 5
        segment = dash_len + gap_len

        for ci, (src_name, dst_name) in enumerate(CONNECTIONS):
            src_idx = _REGION_IDX[src_name]
            dst_idx = _REGION_IDX[dst_name]
            src = REGIONS[src_idx]
            dst = REGIONS[dst_idx]
            intensity = self.display_intensity[src_idx]

            sx, sy = src[1], src[2]
            dx, dy = dst[1] - sx, dst[2] - sy
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1:
                continue

            # Color
            if intensity < 0.01:
                color = (20, 25, 45)
                width = 1
            else:
                sr, sg, sb = src[4]
                t = intensity * 0.8
                color = (
                    int(sr * t + 20 * (1 - t)),
                    int(sg * t + 25 * (1 - t)),
                    int(sb * t + 45 * (1 - t)),
                )
                width = 2

            # Animated dash offset — proportional to activity
            speed_mult = 0.3 + intensity * 0.7
            offset = (self.dash_offset * speed_mult) % segment

            # Walk along line drawing dashes
            ux, uy = dx / length, dy / length
            pos = -offset  # start before 0 so dashes flow in
            while pos < length:
                start = max(pos, 0)
                end = min(pos + dash_len, length)
                if end > start:
                    x1 = int(sx + ux * start)
                    y1 = int(sy + uy * start)
                    x2 = int(sx + ux * end)
                    y2 = int(sy + uy * end)
                    pg.draw.line(self.screen, color, (x1, y1), (x2, y2),
                                 width)
                pos += segment

    # ── Region Glows ──────────────────────────────────────────────────────

    def _draw_regions(self):
        """Draw all brain regions with gaussian glow from cache."""
        for idx, (name, x, y, radius, color, group) in enumerate(REGIONS):
            intensity = self.display_intensity[idx]

            # Dim base ring (always visible)
            dim = (color[0] // 6, color[1] // 6, color[2] // 6)
            self.pg.draw.circle(self.screen, dim, (x, y), radius, 1)

            if intensity > 0.015:
                surf = self.glow_cache.get(idx, intensity)
                if surf is not None:
                    blit_x = x - surf.get_width() // 2
                    blit_y = y - surf.get_height() // 2
                    self.screen.blit(surf, (blit_x, blit_y),
                                     special_flags=self.pg.BLEND_ADD)

            # Region label
            label = name.replace('_L', ' L').replace('_R', ' R')
            if intensity > 0.1:
                text_color = (
                    min(color[0] + 80, 255),
                    min(color[1] + 80, 255),
                    min(color[2] + 80, 255),
                )
            else:
                text_color = (40, 45, 70)
            txt = self.font_sm.render(label, True, text_color)
            self.screen.blit(txt, (x - txt.get_width() // 2,
                                   y + radius + 3))

    # ── HUD ───────────────────────────────────────────────────────────────

    def _draw_hud(self):
        """Draw top and bottom info bars with glow styling."""
        d = self.data
        pg = self.pg

        # ── Top bar ──
        self.screen.blit(self._title_glow, (4, 2))

        t_sim = d.get('t_sim', 0.0)
        t_txt = self.font.render(f't={t_sim:.3f}s', True, COL_HUD)
        self.screen.blit(t_txt, (self.WIDTH - t_txt.get_width() - 10, 10))

        # Thin separator with gradient feel
        for i in range(self.WIDTH):
            brightness = int(40 * (1.0 - abs(i - self.WIDTH / 2) /
                                   (self.WIDTH / 2)) + 15)
            pg.draw.line(self.screen,
                         (brightness // 3, brightness // 2, brightness),
                         (i, 32), (i, 32), 1)

        # ── Bottom bar ──
        y_bot = self.HEIGHT - 22

        # Bottom separator
        for i in range(self.WIDTH):
            brightness = int(40 * (1.0 - abs(i - self.WIDTH / 2) /
                                   (self.WIDTH / 2)) + 15)
            pg.draw.line(self.screen,
                         (brightness // 3, brightness // 2, brightness),
                         (i, y_bot - 6), (i, y_bot - 6), 1)

        # Mode
        mode = d.get('mode', 'walking')
        mode_color = MODE_COLORS.get(mode, COL_HUD)
        mode_txt = self.font.render(f'MODE: {mode.upper()}', True,
                                    mode_color)
        self.screen.blit(mode_txt, (10, y_bot))

        # Stimulus
        stim = d.get('stimulus', '')
        stim_txt = self.font.render(f'STIM: {stim}', True, COL_HUD)
        self.screen.blit(stim_txt, (170, y_bot))

        # Drive
        drv = d.get('drive', [0.0, 0.0])
        drv_txt = self.font.render(
            f'DRIVE L={drv[0]:.2f} R={drv[1]:.2f}', True, COL_HUD)
        self.screen.blit(drv_txt, (330, y_bot))

        # Threat indicator
        threat = d.get('threat_asym', 0.0)
        if mode == 'escape' and abs(threat) > 0.01:
            if threat > 0:
                thr_str = f'THREAT: --> RIGHT (+{threat:.2f})'
            else:
                thr_str = f'THREAT: <-- LEFT ({threat:.2f})'
            thr_txt = self.font.render(thr_str, True, COL_ESCAPE)
            self.screen.blit(thr_txt, (530, y_bot))

        # Flight indicator
        flight_state = d.get('flight_state', 'grounded')
        if flight_state != 'grounded':
            flt_alt = d.get('flight_alt', 0.0)
            flt_wf = d.get('flight_wing_freq', 0.0)
            flt_txt = self.font.render(
                f'ALT={flt_alt:.1f}mm WING={flt_wf:.0f}Hz', True, COL_FLIGHT)
            self.screen.blit(flt_txt, (530, y_bot))
        else:
            # Wing song indicator (only when not in flight)
            wing_song = d.get('wing_song', 'silent')
            if wing_song != 'silent':
                wing_freq = d.get('wing_freq', 0)
                ws_txt = self.font.render(
                    f'SONG: {wing_song} {wing_freq:.0f}Hz', True, COL_WING)
                self.screen.blit(ws_txt, (530, y_bot))

        # Ball distance
        ball_x = d.get('ball_x', None)
        if ball_x is not None:
            ball_txt = self.font_sm.render(
                f'BALL:{ball_x:.0f}mm', True, (70, 80, 120))
            self.screen.blit(ball_txt, (self.WIDTH - 90, y_bot + 2))

    # ── Sidebar ───────────────────────────────────────────────────────────

    def _draw_sidebar(self):
        """Draw DN group activity bars on right side with gradient fill."""
        d = self.data
        pg = self.pg
        x_start = 710
        bar_w = 70
        bar_h = 12
        y = 50

        # Header with glow
        header = self.font_sm.render('DN ACTIVITY', True, COL_HUD)
        # Subtle glow behind header
        hdr_glow = self.font_sm.render('DN ACTIVITY', True,
                                       (COL_HUD[0] // 4,
                                        COL_HUD[1] // 4,
                                        COL_HUD[2] // 4))
        self.screen.blit(hdr_glow, (x_start - 1, y - 17))
        self.screen.blit(hdr_glow, (x_start + 1, y - 17))
        self.screen.blit(header, (x_start, y - 16))

        for label, key, color in SIDEBAR_BARS:
            val = _clamp(d.get(key, 0.0))

            # Label
            lbl_color = color if val > 0.1 else (60, 65, 90)
            lbl = self.font_sm.render(label, True, lbl_color)
            self.screen.blit(lbl, (x_start, y))

            # Bar background
            bar_x = x_start + 32
            pg.draw.rect(self.screen, (12, 14, 28),
                         (bar_x, y, bar_w, bar_h))
            pg.draw.rect(self.screen, (30, 35, 60),
                         (bar_x, y, bar_w, bar_h), 1)

            # Bar fill with gradient
            fill_w = int(bar_w * val)
            if fill_w > 0:
                grad_surf = self._bar_gradients.get(key)
                if grad_surf is not None:
                    # Clip to fill_w
                    self.screen.blit(grad_surf, (bar_x, y),
                                     area=pg.Rect(0, 0, fill_w, bar_h))

            # Value text
            val_color = color if val > 0.3 else (55, 60, 85)
            val_txt = self.font_sm.render(f'{val:.2f}', True, val_color)
            self.screen.blit(val_txt, (bar_x + bar_w + 4, y))

            y += bar_h + 8


# ============================================================================
# Monitor process entry point
# ============================================================================

def _monitor_loop(queue):
    """Entry point for the brain monitor child process."""
    import sys
    import traceback
    try:
        import pygame
        import numpy  # noqa: F401  — ensure available for GlowCache
        pygame.init()
        print("[BrainMonitor] pygame initialized (v2 futuristic)", flush=True)

        renderer = BrainRenderer(pygame)
        renderer.init_display()
        print("[BrainMonitor] window open", flush=True)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            # Drain queue — use only the latest frame
            latest = None
            while True:
                try:
                    latest = queue.get_nowait()
                except Exception:
                    break
            if latest == 'STOP':
                break
            if latest is not None:
                renderer.update_data(latest)

            renderer.render_frame()
            renderer.clock.tick(30)

        pygame.quit()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()


# ============================================================================
# Public API — used by fly_embodied.py (unchanged)
# ============================================================================

class BrainMonitorProcess:
    """Manages the brain monitor child process."""

    def __init__(self):
        self.queue = mp.Queue(maxsize=10)
        self.process = None

    def start(self):
        """Launch the monitor in a separate process."""
        self.process = mp.Process(
            target=_monitor_loop, args=(self.queue,), daemon=True)
        self.process.start()

    def send(self, data_dict):
        """Send neural data to monitor (non-blocking, drops if full)."""
        try:
            self.queue.put_nowait(data_dict)
        except Exception:
            pass  # queue full — skip this frame

    def stop(self):
        """Signal the monitor to shut down."""
        if self.process is not None and self.process.is_alive():
            try:
                self.queue.put_nowait('STOP')
            except Exception:
                pass
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()

    def is_alive(self):
        return self.process is not None and self.process.is_alive()
