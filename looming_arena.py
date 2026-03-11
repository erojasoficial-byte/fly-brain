"""
Looming Arena: Natural environment with a dark sphere approaching the fly.

Features a realistic ground surface, gradient sky, directional sunlight,
and atmospheric fog. The sphere approaches once (single loom) — when it
passes the fly, it disappears, letting the GF response decay naturally.
"""

import numpy as np
from flygym.arena import BaseArena


class LoomingArena(BaseArena):
    """Arena with natural visuals and a looming threat sphere.

    Parameters
    ----------
    ball_radius : float
        Radius of the dark sphere in mm. Default 5.0.
    approach_speed : float
        Speed of approach in mm/s. Default 50.0.
    start_distance : float
        Initial distance of sphere from origin along +x in mm. Default 80.0.
    ball_height : float
        Height of sphere center above ground in mm. Default 2.0.
    """

    def __init__(self, ball_radius=5.0, approach_speed=50.0,
                 start_distance=80.0, ball_height=2.0, approach_angle=0.0,
                 taste_zones=None, odor_sources=None, ground_size=100):
        super().__init__()

        self.ball_radius = ball_radius
        self.approach_speed = approach_speed
        self.start_distance = start_distance
        self.ball_height = ball_height
        self.approach_angle = np.radians(approach_angle)
        self.ground_size = ground_size
        self.curr_time = 0.0

        # Starting position
        self.ball_start = np.array([
            start_distance * np.cos(self.approach_angle),
            -start_distance * np.sin(self.approach_angle),
            ball_height,
        ])
        self.ball_pos = self.ball_start.copy()
        self._passed = False

        # Approach direction (toward origin, horizontal)
        direction = -self.ball_start.copy()
        direction[2] = 0.0
        norm = np.linalg.norm(direction[:2])
        self.approach_dir = (direction / norm if norm > 0
                             else np.array([1.0, 0.0, 0.0]))

        # ══════════════════════════════════════════════════════
        # VISUAL ENVIRONMENT
        # ══════════════════════════════════════════════════════

        # ── Skybox (blue gradient) ──
        self.root_element.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=(0.35, 0.55, 0.8),   # sky blue
            rgb2=(0.85, 0.9, 1.0),    # pale horizon
            width=512,
            height=512,
        )

        # ── Headlight (warm ambient + directional) ──
        headlight = self.root_element.visual.headlight
        headlight.ambient = (0.35, 0.35, 0.4)
        headlight.diffuse = (0.7, 0.7, 0.65)
        headlight.specular = (0.3, 0.3, 0.3)

        # ── Sunlight (directional, warm) ──
        self.root_element.worldbody.add(
            "light",
            name="sun",
            pos=(0, 0, 200),
            dir=(0.4, 0.3, -1.0),
            diffuse=(1.0, 0.95, 0.85),
            specular=(0.5, 0.5, 0.4),
            castshadow=True,
            directional=True,
        )

        # ── Ground plane (natural earth tones) ──
        ground_tex = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=(0.28, 0.38, 0.2),   # dark green-brown
            rgb2=(0.35, 0.42, 0.25),  # slightly lighter green
        )
        ground_mat = self.root_element.asset.add(
            "material",
            name="ground_mat",
            texture=ground_tex,
            texrepeat=(30, 30),
            reflectance=0.05,
            rgba=(1.0, 1.0, 1.0, 1.0),
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=ground_mat,
            size=[self.ground_size, self.ground_size, 1],
            friction=(1, 0.005, 0.0001),
            conaffinity=0,
        )
        self.friction = (1, 0.005, 0.0001)

        # ── Looming sphere (dark, menacing, slightly reflective) ──
        ball_mat = self.root_element.asset.add(
            "material",
            name="threat_ball",
            rgba=(0.05, 0.05, 0.08, 1.0),
            reflectance=0.15,
            specular=0.5,
            shininess=0.8,
        )
        self.object_body = self.root_element.worldbody.add(
            "body",
            name="looming_sphere",
            mocap=True,
            pos=self.ball_start.tolist(),
            gravcomp=1,
        )
        self.object_body.add(
            "geom",
            name="looming_ball",
            type="sphere",
            size=(ball_radius,),
            material=ball_mat,
        )
        # ── Taste zones (glowing floor patches + labels) ──
        _TASTE_LABELS = {'sugar': 'AZUCAR', 'bitter': 'VENENO'}
        if taste_zones:
            _TASTE_COLORS = {
                'sugar':  (0.15, 0.75, 0.15, 0.5),
                'bitter': (0.75, 0.12, 0.12, 0.5),
            }
            for i, zone in enumerate(taste_zones):
                rgba = _TASTE_COLORS.get(zone.taste, (0.5, 0.5, 0.5, 0.4))
                mat = self.root_element.asset.add(
                    "material",
                    name=f"taste_mat_{i}",
                    rgba=rgba,
                    reflectance=0.15,
                    emission=0.3,
                )
                self.root_element.worldbody.add(
                    "geom",
                    name=f"taste_zone_{i}_{zone.taste}",
                    type="cylinder",
                    size=(zone.radius, 0.02),
                    pos=(zone.center[0], zone.center[1], 0.021),
                    material=mat,
                    conaffinity=0,
                    contype=0,
                )
                # Floating label site above zone
                label = _TASTE_LABELS.get(zone.taste, zone.taste.upper())
                self.root_element.worldbody.add(
                    "site",
                    name=label,
                    pos=(zone.center[0], zone.center[1], 4.0),
                    size=(0.5,),
                    rgba=rgba[:3] + (1.0,),
                    group=4,
                )

        # ── Odor sources (glowing spheres with halos + labels) ──
        _ODOR_LABELS = {'attractive': 'COMIDA', 'repulsive': 'PELIGRO'}
        if odor_sources:
            _ODOR_COLORS = {
                'attractive': (0.2, 0.85, 0.3, 0.7),   # green glow
                'repulsive':  (0.85, 0.2, 0.85, 0.7),   # purple glow
            }
            for i, src in enumerate(odor_sources):
                rgba = _ODOR_COLORS.get(
                    src.odor_type, (0.5, 0.5, 0.5, 0.5))
                # Solid core
                mat_core = self.root_element.asset.add(
                    "material",
                    name=f"odor_core_{i}",
                    rgba=rgba,
                    reflectance=0.3,
                    emission=0.5,
                    shininess=0.9,
                )
                self.root_element.worldbody.add(
                    "geom",
                    name=f"odor_source_{i}_{src.odor_type}",
                    type="sphere",
                    size=(1.2,),
                    pos=(src.position[0], src.position[1], 1.5),
                    material=mat_core,
                    conaffinity=0,
                    contype=0,
                )
                # Translucent halo
                halo_rgba = (rgba[0], rgba[1], rgba[2], 0.15)
                mat_halo = self.root_element.asset.add(
                    "material",
                    name=f"odor_halo_{i}",
                    rgba=halo_rgba,
                    emission=0.8,
                )
                self.root_element.worldbody.add(
                    "geom",
                    name=f"odor_halo_{i}_{src.odor_type}",
                    type="sphere",
                    size=(3.0,),
                    pos=(src.position[0], src.position[1], 1.5),
                    material=mat_halo,
                    conaffinity=0,
                    contype=0,
                )
                # Floating label site above source
                label = _ODOR_LABELS.get(src.odor_type, src.odor_type.upper())
                self.root_element.worldbody.add(
                    "site",
                    name=label,
                    pos=(src.position[0], src.position[1], 6.0),
                    size=(0.5,),
                    rgba=rgba[:3] + (1.0,),
                    group=4,
                )

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return 0.0

    def step(self, dt, physics, *args, **kwargs):
        """Move sphere toward fly; stop far behind after passing."""
        if not self._passed:
            self.ball_pos[:3] += self.approach_dir * self.approach_speed * dt

            behind = np.dot(self.ball_pos[:2], self.ball_start[:2])
            dist_to_origin = np.linalg.norm(self.ball_pos[:2])
            if behind < 0 and dist_to_origin > self.ball_radius * 2:
                self._passed = True
                self.ball_pos = np.array([0.0, 0.0, -100.0])

        physics.bind(self.object_body).mocap_pos = self.ball_pos
        self.curr_time += dt
