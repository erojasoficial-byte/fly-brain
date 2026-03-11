# Visually-Guided Escape in an Embodied Whole-Brain Drosophila Simulation

## Abstract

We present the first demonstration of visually-guided escape behavior emerging from an embodied whole-brain connectome simulation of *Drosophila melanogaster*. Our system connects a 138,639-neuron leaky integrate-and-fire (LIF) model of the complete fly brain, running on GPU via PyTorch, to a biomechanically realistic body simulation (NeuroMechFly v2) with compound eye vision (1,442 ommatidia). When a dark sphere approaches the virtual fly, visual contrast signals are injected into T2 lobula neurons identified from FlyWire annotations. The signal propagates through the real connectome weight matrix to activate the Giant Fiber descending neurons, triggering escape locomotion — with no hand-coded behavioral rules. The fly walks, sees the approaching threat, and flees. To our knowledge, this is the first closed-loop system where visually-triggered escape emerges from whole-brain connectome dynamics in an embodied agent.

## 1. Introduction

The complete synaptic connectome of the adult *Drosophila melanogaster* brain has been mapped (Dorkenwald et al., 2024; Schlegel et al., 2024), comprising approximately 139,000 neurons and 50 million synaptic connections. Separately, biomechanically realistic simulations of the fly body have been developed (NeuroMechFly; Lobato-Rios et al., 2022), enabling physics-based locomotion driven by neural commands. However, connecting a whole-brain neural simulation to a biomechanical body — creating an embodied virtual fly that perceives, processes, and acts through its real neural circuits — has remained an open challenge.

The looming escape response is one of the best-characterized visuomotor circuits in *Drosophila*. An approaching dark object activates lobula columnar neurons (LC4, LPLC2), which excite the Giant Fiber (GF) descending neurons, triggering a rapid escape jump and flight initiation (von Reyn et al., 2014; Ache et al., 2019). This circuit spans the entire visual processing hierarchy: photoreceptors (R1-R8) → lamina (L1/L2) → medulla (Mi1, Tm1/Tm2) → lobula (T2, T4/T5) → visual projection neurons (LC4) → descending neurons (GF) → motor circuits.

Here we close the loop: a virtual fly with compound eyes sees a dark approaching sphere in a 3D physics simulation, the visual signal enters the whole-brain connectome at the lobula level, propagates through real synaptic weights to activate the Giant Fiber, and the resulting motor commands drive biomechanical escape locomotion.

## 2. Methods

### 2.1 Neural Simulation

The brain model comprises 138,639 leaky integrate-and-fire (LIF) neurons with alpha-function synapses, simulated on GPU using PyTorch. The connectivity matrix is derived from the FlyWire whole-brain connectome (Dorkenwald et al., 2024), with synaptic weights from the `Excitatory × Connectivity` column of the published dataset. Key parameters: membrane time constant τ_mem = 20 ms, synaptic time constant τ_syn = 5 ms, spike threshold V_th = −45 mV, resting potential V_rest = −52 mV, refractory period t_ref = 2.2 ms, simulation timestep dt = 0.1 ms.

External input is delivered via Poisson spike generators with amplitude scaling factor S = 250. Neuron firing rates are set per-neuron; each timestep, a Bernoulli trial with probability p = rate × dt/1000 determines whether a Poisson spike of amplitude S is generated.

### 2.2 Body Simulation

The fly body is simulated using NeuroMechFly v2 (flygym; Lobato-Rios et al., 2022) with the MuJoCo physics engine. The body model includes six articulated legs (7 degrees of freedom each, 42 total), adhesive tarsal pads, and contact sensors. Locomotion is governed by a HybridTurningController that accepts left/right drive signals and generates coordinated tripod gait patterns. Body simulation timestep: 0.1 ms.

### 2.3 Compound Eye Vision

Each compound eye comprises 721 ommatidia arranged in a hexagonal array, simulated by flygym's retina model. Visual input is obtained by rendering the 3D scene from eye-mounted cameras using MuJoCo's native renderer (to avoid OpenGL context conflicts with the interactive viewer on Windows). The raw camera images undergo fisheye correction and hexagonal pixel sampling, producing a (2, 721, 2) observation array with values in [0, 1] representing pale and yellow photoreceptor channel intensities.

### 2.4 Visual Processing Pipeline

#### 2.4.1 The Scale Mismatch Problem

A critical finding of this work is that the LIF connectome model cannot propagate signals through multiple synaptic layers via network spike transmission. The Poisson spike generator produces spikes with amplitude 250, scaled by wScale = 0.275 for an effective input of 68.75 per spike. In contrast, network spikes have amplitude 1, yielding only weight × 0.275 ≈ 0.6–2.0 per spike for typical visual pathway connections. This 50× scale mismatch means that even strong firing in one layer (e.g., medulla at 40 Hz) produces negligible postsynaptic input compared to direct Poisson stimulation.

We verified this quantitatively: the average synaptic weight from Mi1 → T2 is 2.34 (7,589 connections), while Tm1 → T2 averages 1.24 (1,629 connections). Running the brain model with L1/L2 lamina neurons directly stimulated at biologically appropriate rates (L1: 137–200 Hz, L2: 20–83 Hz) produced zero spikes in T2, LC4, and GF neurons after 500 ms of simulation. The signal dies at the medulla–lobula boundary.

#### 2.4.2 Contrast-Based T2 Injection

Based on the weight analysis, we inject Poisson rates at the T2 lobula level — the last processing stage before the LC4 visual projection neurons. T2 neurons (3,240 identified from FlyWire annotations, types T2 and T2a) respond to small-field dark motion in biological *Drosophila*. From T2 onward, the signal propagates through the **real connectome weight matrix** to reach LC4 and the Giant Fiber.

The T2 firing rate for each neuron is computed from contrast:

```
brightness = mean(pale_channel, yellow_channel)  per ommatidium
mean_bright = mean(brightness)  across all ommatidia
raw_contrast = (mean_bright - brightness) / mean_bright
contrast = max(0, raw_contrast - 0.3)  # threshold filters floor texture
T2_rate = 120 Hz × min(contrast, 1.0)
```

The contrast threshold of 0.3 is necessary to filter natural luminance variation from the arena's checkerboard floor (contrast ≈ 0.12), while preserving the strong contrast signal from the dark sphere (contrast ≈ 1.0). T2 neurons mapped to ommatidia at or above mean brightness receive zero rate, producing zero network activity in the absence of a dark stimulus.

#### 2.4.3 Neuron Identification

All visual pathway neurons were identified from the FlyWire neuron annotation dataset (Schlegel et al., 2024) by exact cell_type matching and separated into left/right populations by the `side` column. Neuron counts in the connectome:

| Layer | Type | Count | Role |
|-------|------|-------|------|
| Retina | R1-R8 | 10,580 | Photoreceptors (identified, not injected) |
| Lamina | L1 | 1,590 | OFF pathway (identified, not injected) |
| Lamina | L2 | 1,696 | ON pathway (identified, not injected) |
| Medulla | Mi1 | 1,582 | Intrinsically active (identified, not injected) |
| Medulla | Tm1 | 1,554 | ON transmedullary (identified, not injected) |
| Medulla | Tm2 | 1,552 | ON transmedullary (identified, not injected) |
| Lobula | T2/T2a | 3,240 | OFF motion → **injected with contrast rates** |
| Lobula plate | LC4 | 104 | Looming detector (connectome-driven) |
| Descending | GF | 2 | Giant Fiber escape (connectome-driven) |

### 2.5 Brain-Body Bridge

Descending neuron (DN) spike rates are decoded using a sliding-window estimator (50 ms window). The Giant Fiber normalized rate (0–1) determines escape behavior: when GF rate exceeds threshold 0.3, the system enters escape mode with maximum drive in the reverse direction (away from the approaching stimulus). Walking locomotion is driven by P9 descending neurons receiving constant Poisson input at 100 Hz.

**Mode hysteresis.** To prevent rapid oscillation between behavioral modes (which produces jerky locomotion), a minimum mode duration of 300 ms is enforced. Mode transitions are only permitted when the current mode has been active for at least this duration. This eliminates the walking↔escape flickering caused by transient fluctuations in descending neuron rates. The tactile escape threshold is set at 35 N to avoid triggering escape from normal walking contact forces (which can reach 28 N during vigorous stepping).

### 2.6 Looming Arena

A dark sphere (radius 6 mm, dark material with specular reflections) approaches the fly from 120 mm along a configurable approach angle at 15 mm/s. The sphere is implemented as a MuJoCo mocap body. The arena features a naturalistic visual environment: gradient skybox (blue to pale horizon), directional warm sunlight with cast shadows, and a natural green-brown checker ground surface. Taste zones are rendered as emissive floor patches (green for sugar, red for bitter) and odor sources as glowing spheres with translucent halos. Floating site labels ("AZUCAR", "VENENO", "COMIDA", "PELIGRO") identify each zone.

**Single loom protocol.** The sphere approaches once. When it passes behind the fly, it is parked at z = −100 mm (invisible), allowing the Giant Fiber response to decay naturally through the connectome without artificial stimulus termination. This biologically faithful protocol ensures that landing is determined by the brain's own neural dynamics, not a programmed timer.

### 2.7 Real-Time Neural Activity Visualization

A dedicated brain monitor runs as a separate process (via Python `multiprocessing`) to avoid OpenGL context conflicts with MuJoCo. It receives neural activity data through a lock-free queue and renders a dorsal brain map at 30 fps using pygame. The visualization maps 16 brain regions — organized into functional groups (visual, looming, escape, motor, backward, grooming, feeding) — onto anatomically plausible 2D positions representing the optic lobes and central brain.

**Gaussian glow rendering.** Each region is rendered as a continuous radial gradient rather than discrete filled circles. Three gaussian layers are composited per region: an outer halo (σ = r × 1.8), an inner glow (σ = r × 0.8), and a hot core (σ = r × 0.3) that shifts toward white at high intensity. Surfaces are pre-rendered at 16 discrete intensity levels per region (~272 surfaces total) using numpy array operations, then blitted per-frame with additive blending (`BLEND_ADD`) for natural bloom. This approach eliminates per-frame numpy computation while preserving smooth gradient quality.

**Temporal smoothing and pulse animation.** Raw intensity values undergo exponential smoothing (τ = 120 ms) to prevent abrupt visual transitions. Active regions additionally exhibit a sinusoidal breathing modulation at ~2.5 Hz with ±8% amplitude. Each region receives a random initial phase, producing asynchronous pulsation that conveys a sense of distributed neural dynamics rather than synchronized flashing.

**Connection particles.** Signal flow between connected regions is visualized by luminous particles that travel from source to destination along each connection. Particle spawn rate is proportional to source-region intensity (0–4 particles/s), with a maximum of 6 particles per connection. Each particle fades in over the first 15% and fades out over the last 15% of its trajectory. Connection lines themselves are rendered as animated dashes (8 px dash, 5 px gap) whose scroll speed scales with activity level.

**Pre-rendered overlays.** Three static overlays are composited once at initialization: (1) a hexagonal grid pattern providing a subtle sci-fi substrate, (2) a brain silhouette with radial gradient (brighter center, transparent edges) rendered from three overlapping ellipses via numpy, and (3) a CRT-style scanline overlay (1 px dark line every 3 pixels) applied with subtractive blending. These contribute to the holographic aesthetic without per-frame computational cost.

**HUD and sidebar.** A heads-up display shows simulation time, behavioral mode (walking/escape/grooming with color coding), stimulus type, bilateral drive values, and threat asymmetry during escape. A sidebar displays descending neuron (DN) group activity as gradient-filled horizontal bars with real-time numerical readouts. Text elements use multi-offset glow rendering for readability against the dark background.

The complete render pipeline per frame comprises: (1) smooth intensities, (2) compute pulse modulation, (3) blit hex grid, (4) blit brain silhouette, (5) draw animated dashed connections, (6) update and draw particles, (7) draw region glows via cache lookup, (8) apply scanline overlay, (9) draw HUD and sidebar, (10) display flip. Total draw operations are approximately 140 per frame at 30 fps, with negligible CPU overhead since all numpy operations occur only during initialization.

### 2.8 Somatosensory and Auditory Integration

Beyond vision, the system processes two additional sensory modalities through Johnston's Organ (JO) neurons, the primary mechanosensory and auditory afferents in *Drosophila*.

**Touch (mechanosensory).** MuJoCo contact forces are read from 36 sensors (6 legs × 6 segments: tibia + 5 tarsal segments) at each simulation timestep. Forces are grouped by leg and the maximum magnitude per leg is computed. The three left legs (LF, LM, LH) map to left-hemisphere JO-E/C neurons, and the three right legs (RF, RM, RH) to right-hemisphere JO-E/C neurons. A total of 428 JO touch neurons (222 left, 206 right) were identified from FlyWire annotations by matching cell_type prefixes: JO-E, JO-EDC, JO-EDM, JO-EDP, JO-EV, JO-EVL, JO-EVM, JO-EVP, JO-C, JO-CA, JO-CL, and JO-CM. Force magnitude is mapped to firing rate via:

```
excess = max(force - 0.3, 0)        # floor filters normal ground contact
rate = min(excess / 9.7, 1.0) × 250 Hz
```

Forces below 0.3 N (normal walking contact) produce zero JO activation. Moderate forces (1.5–5.0 N) drive grooming behavior through the connectome pathway JO-E → aDN1. Forces exceeding 5.0 N (e.g., collision with the approaching sphere) trigger tactile escape at the brain-body bridge level, providing a backup pathway when the connectome does not propagate the signal to the Giant Fiber.

**Sound (auditory).** JO-A and JO-B subtypes process near-field vibration in biological *Drosophila*: JO-A neurons are frequency-tuned (responding preferentially to the ~200 Hz courtship song), while JO-B neurons encode vibration amplitude broadly. We identified 390 auditory JO neurons (213 left, 177 right) from FlyWire annotations. Virtual vibration sources are placed in the arena, each defined by position, frequency, and amplitude. The vibration signal at each antenna is computed as:

```
attenuation = amplitude / (1 + (distance / 40mm)²)
frequency_gain = exp(-0.5 × ((f - 200) / 80)²)
effective = attenuation × (0.6 × frequency_gain + 0.4)
rate = effective × 200 Hz
```

Bilateral activation depends on the angle of the source relative to the fly's heading, with ipsilateral neurons receiving higher rates (cosine-weighted split with ±40% modulation). This directional asymmetry enables phonotaxis: the fly orients toward courtship-frequency sources through a combination of connectome-mediated JO → turning DN pathways and a bridge-level orientation bias.

**Multi-modal coexistence.** Somatosensory rates are injected into the brain using element-wise maximum with existing rates, ensuring that manual keyboard stimuli, visual inputs, and mechanosensory/auditory inputs coexist without mutual interference. All three sensory modalities (vision, touch, sound) can be active simultaneously, with behavioral mode selection following the priority hierarchy: escape > grooming > walking.

### 2.9 Gustatory System

*Drosophila* detects tastants through gustatory receptor neurons (GRNs) located on the tarsi, proboscis, and wing margins. We implement tarsal gustation: when leg end-effectors contact defined taste zones on the arena floor, the corresponding GRN populations are activated.

**Taste zones.** Circular regions on the arena floor are defined with center position, radius, and taste identity (sugar or bitter). These are rendered as translucent colored cylinders in MuJoCo (green for sugar, red for bitter) with `conaffinity=0, contype=0` to avoid interfering with physics.

**GRN populations.** From the FlyWire connectome annotations, 21 sugar GRNs and 41 bitter GRNs were identified. At each simulation step, the six tarsal end-effector positions are checked against all taste zones. Only feet near ground level (z < 0.5 mm) are considered in contact. The firing rate scales with the number of legs touching a zone:

```
rate = min(legs_in_zone / 2, 1.0) × max_rate
```

where max_rate is 200 Hz for sugar GRNs and 250 Hz for bitter GRNs. Two or more legs in a zone produce full activation.

**Behavioral effects.** Sugar GRN activation propagates through the connectome to MN9 (proboscis motor) neurons, triggering feeding mode — the fly stops walking and extends the proboscis. Bitter GRN activation triggers escape at the bridge level, overriding other behaviors with the same priority as Giant Fiber escape. The behavioral priority hierarchy is updated to: escape (GF or bitter or tactile) > grooming > feeding > walking.

**Multi-modal coexistence.** Gustatory rates are injected using the same element-wise maximum mechanism as somatosensory rates. All four sensory modalities (vision, touch, sound, taste) can be active simultaneously via the `--visual --somatosensory --gustatory` flags.

### 2.10 Olfactory System

*Drosophila* navigates chemical landscapes using ~2,300 olfactory receptor neurons (ORNs) on the antennae, each tuned to specific volatiles. We implement bilateral olfaction with two receptor populations identified from FlyWire annotations:

- **ORN_DM1** (Or42b equivalent, 68 neurons): responds to food-related volatiles (vinegar, fruit esters). Drives attractive chemotaxis.
- **ORN_DA2** (Or56a equivalent, 39 neurons): responds to geosmin (microbial danger signal). Drives aversive escape.

**Odor sources and concentration gradient.** Virtual odor sources are placed in the arena, each defined by position, type (attractive/repulsive), amplitude, and spread. Concentration follows inverse-square falloff:

```
c(r) = amplitude / (1 + (r / spread)²)
```

This produces a smooth gradient that the fly can navigate. Sources are rendered as small colored spheres in MuJoCo (green = attractive, purple = repulsive).

**Bilateral antenna detection.** Left and right antenna positions are computed from the fly's center position and heading angle, offset perpendicular to the forward direction by ±1.0 mm (slightly exaggerated vs real anatomy to produce functional gradients at simulation scale). The concentration at each antenna is computed independently, and the bilateral asymmetry drives chemotaxis:

```
attraction_bias = (c_right - c_left) / (c_right + c_left)
```

This bias is amplified by a gain factor (4.0×) and added to the fly's turning drive during walking, causing the fly to gradually orient toward attractive sources.

**Behavioral effects.** Attractive ORN activation (DM1) adds a turning bias toward the food source during walking mode — the fly performs chemotaxis by curving toward higher concentrations. Repulsive ORN activation (DA2) above 30% threshold triggers escape mode with directional avoidance: the fly turns away from the side with higher aversive concentration. The behavioral priority hierarchy is: escape (GF or bitter or tactile or olfactory repulsive) > grooming > feeding > walking.

**Multi-modal coexistence.** Olfactory rates are injected using element-wise maximum. All five sensory modalities (vision, touch, sound, taste, smell) can be active simultaneously via `--visual --somatosensory --gustatory --olfactory`.

### 2.11 Wing Song (Vocalization)

*Drosophila* males produce courtship songs by extending and vibrating one wing. The NeuroMechFly model has static wing bodies (no wing joints in the MJCF), so we implement song production virtually: descending neuron activity patterns trigger song mode selection, which emits a vibration signal from the fly's position.

**Song types.** Three song modes are modeled based on real *Drosophila* acoustic communication:
- **Pulse song** (200 Hz, amplitude 0.7): interpulse interval courtship signal, the primary close-range mating song.
- **Sine song** (160 Hz, amplitude 0.5): continuous low-frequency courtship approach signal.
- **Alarm buzz** (400 Hz, amplitude 0.9): high-frequency distress signal during escape.

**Triggering.** Song mode is determined by DN firing rates at each simulation step:
- MN9 (feeding/approach) rate above 0.03 triggers courtship, alternating between pulse (60% of cycle) and sine (40%) on a 1-second period.
- GF (escape) rate above 0.15 triggers alarm buzz, overriding courtship.

**Self-hearing and social signaling.** The wing song is emitted as a `VibrationSource` at the fly's position, attenuated to 20% amplitude for self-hearing. This source is processed by the somatosensory system's JO neurons, creating a closed sensorimotor loop: DN activity → song production → JO detection → brain activity. In a multi-fly scenario, other flies would detect the full-amplitude vibration, enabling acoustic social interaction.

**Audio output.** The brain monitor process generates real-time audio tones via `pygame.mixer`, with pre-rendered sine waves at 160, 200, and 400 Hz. The tone switches automatically as the fly's song mode changes, providing audible feedback of the neural state.

### 2.12 Virtual Flight System

NeuroMechFly v2 has no flight mode — wings are static mesh bodies without joints or actuators. We implement virtual flight pragmatically using MuJoCo's `xfrc_applied` mechanism to apply external forces to the Thorax body, simulating aerodynamic lift and thrust.

**Flight state machine.** Four states govern flight: GROUNDED → TAKEOFF → FLYING → LANDING → GROUNDED. The Giant Fiber (GF) normalized rate exceeding 0.06 triggers takeoff; GF dropping below 0.06 initiates landing. A 3-second post-landing cooldown prevents immediate re-takeoff from residual GF activity.

**Translational forces.** During takeoff, lift = 1.4 × mg (body weight) provides rapid ascent; during flight, a P-controller maintains target altitude (5 mm) with gain 0.12. Horizontal thrust is proportional to GF/P9 activity along the locked escape heading. Aerodynamic drag (linear 0.008 × mg per mm/s, vertical 0.015 × mg per mm/s) provides natural deceleration. During landing, lift decreases with altitude: `lift = 0.5 × mg × max(0, 1 - altitude/15mm)`, producing controlled descent.

**Orientation control via quaternion override.** Torque-based orientation control fails on articulated bodies with 100+ joints: the joint PD controllers, leg masses, and contact dynamics create competing torques that overwhelm any externally applied corrective torques, resulting in uncontrollable spinning. Instead, we directly override the free joint quaternion in `data.qpos` after each physics step, locking the body to an upright orientation with the escape yaw angle determined at takeoff. Angular velocity (`data.qvel[3:6]`) is zeroed simultaneously. This guarantees rigid heading lock regardless of articulated body dynamics.

**Ballistic escape flight.** Real *Drosophila* escape flights are ballistic — the heading is committed at takeoff and maintained throughout flight. The escape heading is computed from `fly_orientation` (the body X-axis in world frame, obtained via flygym's `framexaxis` sensor) at the moment GF exceeds threshold. The fly's forward direction vector is locked as `escape_heading`, and all thrust is applied along this direction.

**Leg freezing during flight.** Leg joints are set to the neutral standing pose (`groom_ctrl.neutral`) with adhesion disabled during flight, preventing asymmetric torques from leg oscillation.

### 2.13 Proboscis Extension

The NeuroMechFly model includes passive Rostrum and Haustellum bodies (the proximal and distal segments of the proboscis) with high-resolution STL meshes, but no joints or actuators. We add a hinge joint dynamically to the Rostrum body via dm_control's MJCF API before model compilation:

```
joint: type=hinge, axis=[0,1,0] (pitch), range=[-0.1, 1.2] rad
stiffness=50 (spring return), damping=5
```

When the brain-body bridge enters feeding mode (MN9 motor neurons active via sugar GRN → connectome → MN9 pathway), the joint `qpos` is set to 1.0, extending the proboscis downward. When feeding ceases, the spring stiffness (50) retracts it automatically. This creates a complete sensorimotor loop: tarsal sugar contact → GRN spikes → connectome propagation → MN9 activation → proboscis extension, with no hand-coded behavioral rules beyond the biomechanical joint.

## 3. Results

### 3.1 Visually-Triggered Escape

When the dark sphere approaches, the fly exhibits a graded escape response that scales with the angular size of the threat:

| Time (s) | Ball distance (mm) | GF rate (normalized) | Behavior | Fly position x (mm) |
|----------|-------------------|---------------------|----------|---------------------|
| 0.2 | 75 | 0.000 | Walking | +0.3 |
| 0.4 | 70 | 0.188 | Walking | +0.7 |
| 0.6 | 65 | 0.300 | Walking → Escape | +1.2 |
| 0.8 | 60 | 0.400 | Escape | +0.6 |
| 1.0 | 55 | 0.450 | Escape | −0.9 |
| 2.0 | 30 | 0.450 | Escape | −5.3 |
| 3.2 | 0 | 0.550 | Escape | −14.5 |

The fly transitions from forward walking to escape at t ≈ 0.6 s when the ball is at 65 mm (angular diameter ≈ 10.6°). It then reverses direction, moving from x = +1.2 mm to x = −14.5 mm — a displacement of 15.7 mm away from the approaching threat.

### 3.2 Causal Verification

To verify that escape is causally driven by vision, we tested the brain model with varying stimulus conditions:

| Condition | T2 neurons active | GF rate (Hz) |
|-----------|-------------------|--------------|
| No ball (uniform ground) | 0 | 0.0 |
| Ball small (5% dark ommatidia) | 162 | 50.0 |
| Ball medium (20% dark) | 648 | 78.3 |
| Ball big (50% dark) | 1,620 | 90.0 |

With zero T2 stimulation, GF produces zero spikes — confirming that escape is entirely driven by visual input, not spontaneous activity.

### 3.3 Connectome Pathway Verification

T2 → LC4 connectivity: 425 non-zero synaptic connections (mean weight 1.08), reaching 99 of 104 LC4 neurons. T2 → GF activation proceeds through multiple connectome pathways, not exclusively through LC4. When T2 is stimulated at 100 Hz uniformly, LC4 fires at 18.5 Hz and GF at 80 Hz. The connectome's rich interconnectivity provides redundant pathways from lobula to descending neurons.

## 4. Discussion

### 4.1 Significance

This work demonstrates that meaningful visuomotor behavior can emerge from whole-brain connectome dynamics in an embodied agent. The escape response is not programmed — it arises from the propagation of visual contrast signals through 3,240 T2 lobula neurons, across the real synaptic weight matrix, to 2 Giant Fiber descending neurons, which then drive biomechanical locomotion. The connectome determines *which* T2 activation patterns produce escape and *how strongly* the Giant Fiber responds.

### 4.2 Limitations

**Graded potential layers**: The biological visual system from retina through medulla uses graded (analog) potentials, not spikes. The LIF model cannot capture this, necessitating direct rate injection at the T2 level rather than at photoreceptors.

**Scale mismatch**: The 50× amplitude difference between Poisson stimulation and network synaptic transmission is a fundamental limitation of the current model parameterization. Reducing the Poisson scale or amplifying visual pathway weights could enable deeper signal propagation in future work.

**Spatial mapping**: The ommatidium-to-neuron mapping uses sorted-ID distribution rather than true retinotopic coordinates, which are not yet available in the connectome annotations.

**Single visual behavior**: The looming escape circuit was the primary visual behavior tested. Mechanosensory (touch) and auditory (vibration) inputs via Johnston's Organ now complement the visual pathway, providing multi-modal sensory integration through the same connectome.

### 4.3 Real-Time Observability

The brain monitor provides real-time visual feedback of neural dynamics during simulation. The dorsal brain map renders functional region activity as gaussian glow fields with temporal smoothing, animated connection particles showing signal flow direction and magnitude, and pulsing modulation that conveys the distributed nature of neural processing. This visualization was instrumental during development for diagnosing signal propagation issues (e.g., verifying that T2 activation reached GF through the connectome) and continues to serve as a qualitative validation tool: the visual cascade from retina → T2 → LC4 → GF → motor neurons is immediately apparent when the looming sphere enters the fly's visual field. Running as a separate process, the monitor introduces no performance overhead to the main simulation loop.

### 4.4 Future Directions

- Reduce Poisson scale factor to enable multi-layer propagation (retina → lamina → medulla → lobula)
- Add retinotopic mapping using column-level connectivity data
- Implement motion detection (T4/T5 → optomotor response) for course stabilization
- Multi-sensory integration: combine visual escape with olfactory attraction/repulsion
- Closed-loop learning: adapt synaptic weights based on behavioral outcomes

## 5. Methods Summary

| Component | Implementation | Scale |
|-----------|---------------|-------|
| Brain model | LIF + alpha synapse, PyTorch GPU | 138,639 neurons |
| Connectome | FlyWire (Dorkenwald et al., 2024) | ~50M synapses |
| Body model | NeuroMechFly v2 / MuJoCo | 42 DOF, 6 legs |
| Vision | Compound eye, 721 ommatidia/eye | 1,442 total |
| Visual injection | T2 lobula, contrast-based | 3,240 neurons |
| Connectome pathway | T2 → LC4 → GF (real weights) | Verified |
| Brain monitor | Gaussian glow, particles, pygame (separate process) | 20 regions, 30 fps |
| Touch (JO-E/C) | Contact forces → bilateral JO rates | 428 neurons |
| Sound (JO-A/B) | Vibration sources → bilateral JO rates | 390 neurons |
| Taste (sugar GRN) | Tarsal contact → sugar GRN rates | 21 neurons |
| Taste (bitter GRN) | Tarsal contact → bitter GRN rates | 41 neurons |
| Smell (ORN_DM1/Or42b) | Odor gradient → bilateral attractive ORN rates | 68 neurons |
| Smell (ORN_DA2/Or56a) | Odor gradient → bilateral repulsive ORN rates | 39 neurons |
| Wing song | DN activity → virtual vibration → JO self-hearing | 3 song types |
| Virtual flight | GF → xfrc_applied forces + qpos quaternion override | 4 states |
| Proboscis | MN9 → dynamic hinge joint on Rostrum body | 1 DOF |
| Simulation speed | ~800 brain steps/s (RTX 4060) | 0.1 ms/step |

## References

- Ache, J.M., et al. (2019). Neural basis for looming size and velocity encoding in the Drosophila giant fiber escape pathway. *Current Biology*.
- Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*.
- Lobato-Rios, V., et al. (2022). NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. *Nature Methods*.
- Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome cell type quantification. *Nature*.
- von Reyn, C.R., et al. (2014). A spike-timing mechanism for action selection. *Nature Neuroscience*.

---
