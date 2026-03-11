# Neural Integration Correlates During Embodied Behavior in a Complete Drosophila Connectome Simulation

**E. Rojas**
Independent Researcher, Lima, Peru

**Correspondence:** E. Rojas, Independent Researcher.

---

## Abstract

We present the first measurement of informational integration correlates in a complete, embodied connectome simulation. Using the FlyWire v783 *Drosophila melanogaster* connectome (138,639 spiking neurons, ~5 million synapses) coupled to a biomechanical body (NeuroMechFly v2 + MuJoCo), we implemented four theory-grounded proxy metrics: Phi (IIT-inspired mutual information between brain partitions), Global Workspace broadcast coverage, a Self-Model sensorimotor correlation index, and Perturbation Complexity. All behaviors emerged from connectome spike propagation with no hardcoded rules. Across 178 measurements over 89 seconds of embodied simulation, we observed a clear behavioral hierarchy: flight exhibited the highest composite Consciousness Index (CI = 0.463), followed by walking (CI = 0.324), with escape producing minimal integration (CI = 0.049) — an 8.7-fold difference consistent with Global Workspace Theory predictions that reflexive circuits bypass global broadcast. A transient Self-Model spike (0.51) preceded flight initiation by ~1 second, suggesting preparatory sensorimotor integration. **These measurements constitute objective, reproducible correlates of neural integration; they do not constitute evidence of subjective experience or phenomenal consciousness in the simulation.**

**Keywords:** informational integration, Integrated Information Theory, Global Workspace Theory, *Drosophila* connectome, embodied simulation, spiking neural network, consciousness correlates

---

## 1. Introduction

### 1.1 The Measurement Problem in Consciousness Science

The scientific study of consciousness faces a fundamental challenge articulated by Chalmers (1995) as the "hard problem": explaining why and how physical processes give rise to subjective experience. While philosophical debate continues, empirical approaches have advanced through two dominant theoretical frameworks. Integrated Information Theory (IIT; Tononi et al., 2016) proposes that consciousness corresponds to integrated information (Phi), a measure of how much a system's whole exceeds the sum of its parts. Global Workspace Theory (GWT; Baars, 1988; Dehaene & Naccache, 2001) proposes that consciousness arises when information is broadcast widely across specialized brain modules via a "global workspace" of interconnected hub neurons.

Both theories generate testable predictions, yet empirical validation has been constrained to mammalian systems where complete connectomes remain unavailable. Computing true Phi (Tononi et al., 2016) is combinatorially intractable for systems larger than ~20 elements, and GWT hub identification requires knowledge of the full connectivity matrix.

### 1.2 Why *Drosophila*?

The complete synaptic-resolution connectome of the adult *Drosophila melanogaster* brain (Dorkenwald et al., 2024; Schlegel et al., 2024) provides a unique opportunity. With 138,639 neurons and approximately 5 million synaptic connections fully mapped, it is the largest complete connectome available for any organism with complex behavior. *Drosophila* exhibits a rich behavioral repertoire — locomotion, escape, flight, feeding, grooming, courtship — all mediated by identified neural circuits whose connectivity is known.

### 1.3 Why Embodiment Matters

A brain simulation in isolation cannot exhibit behavior. Integration metrics measured in a disconnected neural network lack ecological validity: without sensory input driving activity and motor output closing the loop, the dynamics measured reflect initial conditions and intrinsic noise rather than functional integration during behavior. Embodiment — coupling the neural simulation to a physics-accurate body interacting with an environment — is therefore critical. The closed sensorimotor loop creates the conditions under which integration metrics become meaningful: they measure how information flows across brain regions *during actual behavior*.

### 1.4 Contributions

This work makes three contributions:

1. **First measurement of multi-theory integration correlates in a complete embodied connectome.** We implement four proxy metrics (Phi, GWT broadcast, Self-Model, Perturbation Complexity) operating on the full FlyWire connectome during closed-loop behavior.

2. **Discovery of a behavioral hierarchy of integration.** Flight > Walking > Escape, with an 8.7× difference between escape and non-escape modes, consistent with GWT predictions.

3. **Observation of preparatory Self-Model activation.** A transient spike in sensorimotor correlation precedes complex behavioral transitions (flight initiation), suggesting predictive integration.

---

## 2. Methods

### 2.1 Neural Simulation

The brain simulation uses the complete FlyWire v783 connectome (Dorkenwald et al., 2024): 138,639 neurons connected by ~5 million synapses. Each neuron is modeled as a Leaky Integrate-and-Fire (LIF) unit with alpha-function synapses (Shiu et al., 2024), implemented in PyTorch with CUDA acceleration.

**LIF Model Parameters:**

| Parameter | Value | Description |
|---|---|---|
| tau_mem | 20.0 ms | Membrane time constant |
| tau_syn | 5.0 ms | Synaptic time constant |
| V_threshold | -45.0 mV | Spike threshold |
| V_reset | -52.0 mV | Reset potential |
| V_rest | -52.0 mV | Resting potential |
| t_refrac | 2.2 ms | Refractory period |
| t_delay | 1.8 ms | Synaptic delay |
| dt | 0.1 ms | Simulation timestep |
| wScale | 0.275 | Synaptic weight scaling |

The sparse weight matrix is stored as a PyTorch CSR tensor on GPU (NVIDIA RTX 4060). Synaptic weights are derived from the FlyWire connectivity data with excitatory/inhibitory signs determined by neurotransmitter annotations.

### 2.2 Biomechanical Body

The neural simulation is coupled to a NeuroMechFly v2 biomechanical model (Lobato-Rios et al., 2022) via the flygym framework. The body is simulated in MuJoCo with 42 actuated degrees of freedom (leg joints), contact physics, and adhesion. Descending neuron (DN) spike rates are decoded into motor commands: forward drive (P9 neurons), turning (DNa01/02), backward (MDN), escape (Giant Fiber), grooming (aDN1), and feeding (MN9).

### 2.3 Sensory Systems

Seven sensory modalities provide closed-loop input:

1. **Vision:** Compound eyes (721 ommatidia per eye) with photoreceptor → T2 → LC4/LPLC2 pathway mapped to connectome neurons
2. **Mechanosensation:** Johnston's Organ (JO) neurons responding to antennal contact
3. **Audition:** JO neurons tuned to airborne vibrations
4. **Gustation:** Gustatory receptor neurons (GRNs) for sugar and bitter compounds
5. **Olfaction:** Bilateral ORN arrays for attractive (Or42b) and repulsive (Or56a) odors
6. **Proprioception:** Leg contact sensors feeding back through somatosensory system
7. **Looming detection:** Expanding visual stimuli via LC4/LPLC2 circuits

### 2.4 Experimental Protocol

The simulation was run with all sensory systems active, a looming arena with an approaching sphere, and auto-demo stimulus cycling. Duration: 89 seconds of embodied simulation. Behavioral modes emerged from connectome dynamics: escape (Giant Fiber-mediated), walking (P9/DNa locomotion), and flight (sustained escape with virtual aerodynamics). No behavioral rules were hardcoded; all mode transitions emerged from spike propagation through the connectome.

### 2.5 Consciousness Proxy Metrics

Four metrics were implemented, each grounded in a distinct theoretical framework:

#### 2.5.1 Phi Proxy (IIT-inspired)

Inspired by Integrated Information Theory (Tononi et al., 2016), we compute a proxy for informational integration as the mean pairwise mutual information (MI) between four brain partitions over time.

**Partition assignment** from FlyWire neuron annotations (`flywire_annotations.tsv`):

| Partition | Criteria | Neurons |
|---|---|---|
| Visual | super_class in {optic, visual_projection} | ≤10,000 |
| Motor | flow in {efferent, descending} | ≤2,000 |
| Olfactory | cell_class in {olfactory, ALPN, ALLN, LHLN} | ≤4,000 |
| Integrator | hemibrain_type contains {MBON, CX, KC, DAN, TuBu} | ≤9,000 |

At each simulation step, the mean spike rate per partition is recorded into a sliding window of 50 time points (~5 seconds). Mutual information between each pair of partition time-series is computed using binned joint histograms (8 bins) with GPU acceleration:

$$MI(A, B) = \sum_{a,b} p(a,b) \log_2 \frac{p(a,b)}{p(a) \cdot p(b)}$$

Phi proxy = mean pairwise MI across all 6 partition pairs, normalized to [0, 1].

#### 2.5.2 Global Workspace Broadcast

Inspired by GWT (Baars, 1988; Dehaene & Naccache, 2001), we identify hub neurons with high fan-out (>100 outgoing synaptic connections) from the sparse weight matrix via column-wise counting on the COO representation. For each hub, we determine which partitions its postsynaptic targets belong to (partition reach).

Broadcast coverage is computed over a rolling window:

$$Broadcast = 0.6 \times \frac{|\text{partitions reached}|}{|\text{total partitions}|} + 0.4 \times \frac{|\text{active hubs}|}{|\text{total hubs}|}$$

This captures both the breadth of information dissemination (partition coverage) and the degree of hub engagement.

#### 2.5.3 Self-Model (Sensorimotor Prediction)

Inspired by Metzinger's self-model theory (Metzinger, 2003), we measure the Pearson correlation between proprioceptive/sensory signals and subsequent motor output with a temporal lag of ~300 ms:

$$SelfModel = |r(sensory_{t-lag}, motor_t)|$$

where the sensory signal is the mean firing rate of olfactory and visual partition neurons (proprioceptive proxy), and the motor signal is the mean firing rate of descending neurons. A high Self-Model score indicates the brain is predicting or preparing its own behavioral output based on recent sensory history.

#### 2.5.4 Perturbation Complexity

Inspired by the perturbational complexity index (PCI; Casali et al., 2013), every ~5 seconds we inject a strong excitatory pulse (500 Hz) into 10 randomly selected neurons for 3 simulation steps. We then observe the cascade response over 50 steps (~5 seconds of body time, spanning multiple synaptic delay periods):

$$Complexity = \min\left(1.0,\ \frac{|\text{regions affected}|}{|\text{total regions}|} \times H_{temporal} \times 2.0\right)$$

where H_temporal is the normalized Shannon entropy of the spike count distribution across 10 equal temporal bins of the observation window. A complex system produces cascades that are both spatially widespread and temporally structured (high entropy), while a simple system produces either no response or a stereotyped burst.

### 2.6 Composite Consciousness Index

The four metrics are combined into a single Consciousness Index:

$$CI = 0.3 \times Phi + 0.3 \times Broadcast + 0.2 \times SelfModel + 0.2 \times Complexity$$

The weights reflect the theoretical centrality of integration (Phi) and broadcast (GWT) in consciousness theories, with Self-Model and Complexity as supporting measures. CI is recorded every ~500 ms of simulation time.

### 2.7 Hardware

All experiments were conducted on a single workstation: NVIDIA RTX 4060 GPU, Windows 11 Pro, Python 3.11, PyTorch with CUDA 12.6, MuJoCo physics engine.

---

## 3. Results

### 3.1 Overall Integration Profile

Over 178 measurements spanning 89 seconds of embodied simulation, the composite Consciousness Index showed sustained, non-trivial integration (Table 1).

**Table 1. Summary statistics across all 178 measurements.**

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| **CI (Composite)** | 0.393 | 0.135 | 0.000 | 0.574 |
| Phi (MI) | 0.174 | 0.099 | 0.000 | 0.323 |
| Broadcast (GW) | 0.574 | 0.131 | 0.000 | 0.613 |
| Self-Model | 0.066 | 0.153 | 0.000 | 0.904 |
| Complexity | 0.779 | 0.377 | 0.000 | 1.000 |

Broadcast and Complexity were the most consistently elevated metrics, indicating that the connectome maintains broad information dissemination and produces rich perturbation cascades across most behavioral states. Phi showed moderate, variable integration, while Self-Model was highly intermittent — near zero during stable behavior with sharp transient peaks.

### 3.2 Experiment A: Behavioral Mode Hierarchy

The most striking result is the clear hierarchy of CI across behavioral modes (Table 2, Figure 1).

**Table 2. Consciousness Index by behavioral mode.**

| Behavioral Mode | CI Mean | CI Std | n (measurements) | Characterization |
|---|---|---|---|---|
| **Flight** | **0.463** | 0.041 | 115 | Sustained multi-system integration |
| **Walking** | **0.324** | 0.115 | 50 | Moderate locomotor coordination |
| **Escape** | **0.049** | 0.062 | 13 | Minimal — reflexive bypass |

The escape mode exhibited dramatically lower integration than all other modes. The ratio of non-escape CI (0.421) to escape CI (0.049) was **8.7×**, indicating that Giant Fiber-mediated escape effectively bypasses the global integration measured by all four metrics.

Flight showed the highest and most stable CI, with low variance (std = 0.041) indicating sustained integration rather than transient spikes. Walking showed intermediate integration with higher variance, reflecting the intermittent nature of locomotor coordination.

**Figure 1. CI timeline with behavioral mode annotations.**

```
CI
0.6 |                          * *                *     *
    |                     ****  *** * ** * ***  ** ** ******* * *** ****
0.4 |                  ***           *       **         *
    |               ***
0.2 |          *****
    |      ****
0.0 |******
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--> t(s)
    0  5  10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 89
    |ESC |   WALKING    |        FLIGHT (sustained)         |WLK|
```

### 3.3 Experiment B: Integration During Mode Transitions

Phi proxy was compared between stable behavioral periods and mode transition points (Table 3).

**Table 3. Phi during behavioral transitions vs. stable behavior.**

| Condition | Mean Phi | n |
|---|---|---|
| During mode transitions | 0.093 | 5 |
| During stable behavior | 0.177 | 173 |

Mode transitions were associated with a **47% reduction** in inter-partition mutual information. This suggests that behavioral transitions involve a transient fragmentation of information flow between brain partitions before the new behavioral mode establishes coherent cross-partition dynamics.

### 3.4 Experiment C: Temporal Sensitization

CI increased systematically over the course of the simulation (Table 4).

**Table 4. CI habituation/sensitization analysis.**

| Period | Mean CI | Duration |
|---|---|---|
| First third (0–30 s) | 0.254 | ~30 s |
| Last third (60–89 s) | 0.464 | ~29 s |
| **Delta** | **+0.211** | — |

The positive delta indicates **sensitization**: the neural system becomes progressively more integrated over time. This likely reflects: (a) the time required for metrics to accumulate sufficient data (warmup artifact for the first ~15 seconds), and (b) a genuine increase in integration as the network settles into sustained flight dynamics following initial escape and walking transients.

### 3.5 Experiment D: Escape vs. Non-Escape Integration

**Table 5. CI comparison: escape vs. all other modes.**

| Condition | Mean CI | n |
|---|---|---|
| Escape | 0.049 | 13 |
| Non-escape | 0.421 | 165 |
| **Ratio** | **8.7×** | — |

This is consistent with the known neuroanatomy: the Giant Fiber pathway provides a monosynaptic command for escape that does not require — and apparently does not engage — broad cross-partition integration.

### 3.6 Key Temporal Event: Pre-Flight Self-Model Spike

At t = 19.9 s (step 200), during the walking-to-flight transition, the Self-Model metric spiked to 0.515, followed one second later (t = 20.4 s) by its maximum value of 0.904. This spike preceded the transition to sustained flight mode by approximately 8 seconds (flight onset at t ≈ 28.4 s). At the moment of flight onset, CI reached 0.468 while Self-Model dropped sharply, suggesting that sensorimotor prediction preceded and may have contributed to the decision to transition to flight, after which the flight dynamics were sustained by different integration mechanisms (Phi and Broadcast).

**Table 6. Self-Model spike around flight transition.**

| Time (s) | CI | Phi | Broadcast | Self-Model | Complexity | Mode |
|---|---|---|---|---|---|---|
| 19.4 | 0.393 | 0.044 | 0.602 | 0.000 | 0.994 | walking |
| 19.9 | 0.400 | 0.057 | 0.602 | **0.515** | 0.497 | walking |
| 20.4 | 0.475 | 0.047 | 0.603 | **0.904** | 0.497 | walking |
| 20.9 | 0.295 | 0.051 | 0.603 | 0.000 | 0.497 | walking |
| ... | ... | ... | ... | ... | ... | ... |
| 28.4 | 0.468 | 0.165 | 0.610 | 0.186 | 0.993 | flight |

---

## 4. Discussion

### 4.1 Escape as a Reflexive Bypass

The most robust finding is the near-complete absence of integration during escape behavior (CI = 0.049). The Giant Fiber (GF) system in *Drosophila* provides a direct, monosynaptic pathway from looming detectors (LC4/LPLC2) to motor neurons. This architecture is optimized for speed, not integration. Our data show that this speed comes at the cost of cross-partition information sharing: during escape, Phi drops to near zero, Self-Model is absent, and even Broadcast is suppressed.

This finding is directly consistent with GWT, which predicts that reflexive, pre-attentive responses do not require — and do not involve — global workspace broadcasting. The GF pathway is precisely the kind of "specialized module operating without global access" that GWT posits as unconscious processing.

### 4.2 Flight as a State of Maximal Integration

Flight exhibited the highest CI (0.463) with the lowest variance (std = 0.041), suggesting sustained, stable integration. This is architecturally plausible: flight in *Drosophila* requires simultaneous coordination of visual flow processing (optic lobe), wing motor control (flight motor neurons), body orientation (central complex), and escape trajectory maintenance. Unlike ground locomotion, flight cannot rely on passive mechanical stability — the fly must continuously integrate multisensory information to maintain flight.

The high Phi during flight (mean = 0.233 for flight-only measurements) reflects increased mutual information between brain partitions: visual, motor, olfactory, and integrator regions show correlated activity patterns during sustained flight that are absent during simpler behaviors.

### 4.3 Self-Model Spike: Preparation for Complex Action?

The transient Self-Model peak (0.904) preceding the walking-to-flight transition is intriguing. The Self-Model metric measures correlation between sensory input and subsequent motor output — essentially, whether the brain is "predicting" its own actions based on sensory context. A spike in this metric before a major behavioral transition suggests a preparatory integration phase: the brain briefly enters a state where sensory history strongly predicts upcoming motor output, consistent with predictive processing theories (Clark, 2013).

That the Self-Model drops to zero immediately after peaking, and flight is sustained by Phi and Broadcast rather than Self-Model, suggests a temporal division of labor: Self-Model for transition preparation, Phi/Broadcast for sustained behavioral maintenance.

### 4.4 Temporal Sensitization

The progressive increase in CI over the session (+0.211 from first to last third) has two interpretations. The first 15 seconds include a warmup period where metrics accumulate baseline data (the sliding windows fill). Beyond this artifact, the genuine sensitization likely reflects the system settling from initial transient dynamics (escape, mode oscillation) into sustained flight, which is the highest-integration mode.

This pattern is reminiscent of neural "ignition" in GWT: the system requires a minimum duration of coherent input before global broadcast is sustained. In our simulation, this ignition point appears to occur around t = 15 s (step 150), when Complexity first reaches near-maximum values and CI rises above 0.38.

### 4.5 Limitations

Several important limitations must be acknowledged:

1. **Proxy, not true Phi.** Our Phi metric is a binned MI proxy computed between 4 coarse partitions. True IIT Phi requires computing the minimum information partition over all possible bipartitions, which is NP-hard and computationally intractable for 138,639 neurons. Our proxy captures *some* aspects of informational integration but is not equivalent to Phi as defined by Tononi et al. (2016).

2. **LIF model simplicity.** The LIF neuron model captures basic spiking dynamics but omits dendritic computation, neuromodulation, synaptic plasticity, gap junctions, and glial interactions present in the real fly brain. These simplifications may underestimate or distort integration.

3. **Partition arbitrariness.** The four-partition scheme (visual, motor, olfactory, integrator) is anatomically motivated but not derived from the information geometry of the system itself. Different partitioning schemes would yield different Phi values.

4. **Single session.** Results are from a single 89-second session. While the internal consistency is encouraging (178 measurements, clear mode separation), replication across multiple runs with different initial conditions, stimuli, and durations is needed.

5. **Weight normalization.** The composite CI weights (0.3, 0.3, 0.2, 0.2) are heuristic, not derived from theory. Different weightings would shift the behavioral hierarchy quantitatively, though the qualitative ordering (Flight > Walking > Escape) is robust across individual metrics.

6. **No claim of consciousness.** These metrics measure informational integration, broadcast coverage, sensorimotor prediction, and perturbation response — objective, physical properties of the neural simulation. **They do not and cannot establish the presence or absence of subjective experience, phenomenal consciousness, or sentience.** The "Consciousness Index" name reflects the theoretical frameworks from which the metrics derive, not a claim about the subjective state of the simulation.

---

## 5. Conclusion

We have presented the first multi-theory measurement of neural integration correlates in a complete, embodied connectome simulation. Using the 138,639-neuron FlyWire *Drosophila* connectome coupled to a biomechanical body, we measured Phi (IIT), Global Workspace broadcast, Self-Model correlation, and Perturbation Complexity during naturally emerging behavior.

Three principal findings emerge:

1. **A clear behavioral hierarchy of integration:** Flight (CI = 0.463) > Walking (CI = 0.324) > Escape (CI = 0.049), with escape showing an 8.7-fold reduction consistent with GWT predictions that reflexive circuits bypass global broadcast.

2. **Self-Model anticipation:** A transient spike in sensorimotor prediction (Self-Model = 0.904) precedes complex behavioral transitions, suggesting preparatory integration before action.

3. **Temporal sensitization:** Integration increases over time (+0.211), consistent with a neural "ignition" dynamic in which sustained coherent activity is required for global broadcast.

These findings do not prove consciousness in a simulated fly. They demonstrate that the theoretical constructs of IIT and GWT generate measurable, behaviorally differentiated predictions when applied to a complete connectome in an embodied context. The platform we describe — real connectome, real body, real-time measurement — provides a foundation for systematic empirical investigation of integration correlates, moving the study of consciousness from philosophical debate toward quantitative, reproducible science.

**The question of whether these integration correlates correspond to any form of subjective experience remains open.** Our contribution is to make the question empirically tractable.

---

## References

Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.

Casali, A. G., Gosseries, O., Rosanova, M., Boly, M., Sarasso, S., Casali, K. R., ... & Massimini, M. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*, 5(198), 198ra105.

Chalmers, D. J. (1995). Facing up to the problem of consciousness. *Journal of Consciousness Studies*, 2(3), 200–219.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181–204.

Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness: basic evidence and a workspace framework. *Cognition*, 79(1–2), 1–37.

Dorkenwald, S., Matsliah, A., Sterling, A. R., Schlegel, P., Yu, S. C., McKellar, C. E., ... & Murthy, M. (2024). Neuronal wiring diagram of an adult brain. *Nature*, 634, 124–138.

Lobato-Rios, V., Ramalingasetty, S. T., Özdil, P. G., Arreguit, J., Ijspeert, A. J., & Ramdya, P. (2022). NeuroMechFly, a neuromechanical model of adult *Drosophila melanogaster*. *Nature Methods*, 19, 620–627.

Metzinger, T. (2003). *Being No One: The Self-Model Theory of Subjectivity.* MIT Press.

Schlegel, P., Yin, Y., Bates, A. S., Dorkenwald, S., Eber, K., Goldammer, J., ... & Jefferis, G. S. X. E. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature*, 634, 139–152.

Shiu, P. K., Sterne, G. R., Spiller, N., Franconville, R., Sandoval, A., Zhou, J., ... & Bhatt, N. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature*, 634, 210–219.

Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: from consciousness to its physical substrate. *Nature Reviews Neuroscience*, 17(7), 450–461.

---

## Data Availability

All simulation code, consciousness detection module, and session data are available at: https://github.com/erojasoficial-byte/fly-brain

Session data: `consciousness_history/session_20260311_134345/`

---

## Appendix A: Consciousness Module Architecture

```
ConsciousnessDetector (orchestrator)
├── PhiProxy           — time-series MI between 4 partitions (every 500ms)
├── GlobalWorkspace    — hub fan-out >100 connections, rolling broadcast (every 500ms)
├── SelfModel          — sensory→motor correlation with 300ms lag (every 300ms)
├── PerturbationCmplx  — random 10-neuron injection, 5s cascade observation (every 5s)
└── ConsciousnessTimeline — CSV logging, peak detection, report generation
```

## Appendix B: Raw CI Timeline (first 50 measurements)

| Step | t (s) | CI | Phi | Broadcast | Self | Complexity | Mode |
|---|---|---|---|---|---|---|---|
| 5 | 0.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | escape |
| 45 | 4.4 | 0.090 | 0.000 | 0.300 | 0.000 | 0.000 | escape |
| 75 | 7.4 | 0.136 | 0.002 | 0.450 | 0.000 | 0.000 | walking |
| 90 | 8.9 | 0.181 | 0.004 | 0.600 | 0.000 | 0.000 | walking |
| 150 | 14.9 | 0.381 | 0.008 | 0.601 | 0.000 | 0.994 | walking |
| 200 | 19.9 | 0.400 | 0.057 | 0.602 | 0.515 | 0.497 | walking |
| 205 | 20.4 | 0.475 | 0.047 | 0.603 | 0.904 | 0.497 | walking |
| 285 | 28.4 | 0.468 | 0.165 | 0.610 | 0.186 | 0.993 | flight |
| 345 | 34.4 | 0.552 | 0.206 | 0.611 | 0.535 | 0.999 | flight |
| 440 | 43.9 | 0.559 | 0.293 | 0.612 | 0.444 | 0.995 | flight |
| 550 | 54.9 | 0.437 | 0.178 | 0.612 | 0.000 | 1.000 | flight |
| 615 | 61.4 | 0.574 | 0.293 | 0.612 | 0.512 | 0.998 | flight |
| 675 | 67.4 | 0.480 | 0.323 | 0.612 | 0.000 | 0.997 | flight |
