# Correlatos de Integración Neural Durante Comportamiento Embodied en una Simulación Completa del Conectoma de Drosophila

**E. Rojas**
Investigador Independiente, Lima, Peru

**Correspondencia:** E. Rojas, Investigador Independiente.

---

## Resumen

Presentamos la primera medición de correlatos de integración informacional en una simulación embodied de conectoma completo. Utilizando el conectoma FlyWire v783 de *Drosophila melanogaster* (138,639 neuronas spiking, ~5 millones de sinapsis) acoplado a un cuerpo biomecánico (NeuroMechFly v2 + MuJoCo), implementamos cuatro métricas proxy fundamentadas en teorías: Phi (información mutua inspirada en IIT entre particiones cerebrales), cobertura de broadcast del Global Workspace, un índice de correlación sensorimotor Self-Model, y Complejidad de Perturbación. Todos los comportamientos emergieron de la propagación de spikes a través del conectoma sin reglas programadas. A lo largo de 178 mediciones en 89 segundos de simulación embodied, observamos una jerarquía conductual clara: el vuelo exhibió el mayor Índice de Consciencia compuesto (CI = 0.463), seguido por la marcha (CI = 0.324), con la huida produciendo integración mínima (CI = 0.049) — una diferencia de 8.7 veces consistente con las predicciones de la Teoría del Espacio de Trabajo Global de que los circuitos reflexivos bypasean el broadcast global. Un pico transitorio del Self-Model (0.51) precedió la iniciación del vuelo por ~1 segundo, sugiriendo integración sensorimotor preparatoria. **Estas mediciones constituyen correlatos objetivos y reproducibles de integración neural; no constituyen evidencia de experiencia subjetiva ni consciencia fenoménica en la simulación.**

**Palabras clave:** integración informacional, Teoría de la Información Integrada, Teoría del Espacio de Trabajo Global, conectoma de *Drosophila*, simulación embodied, red neuronal spiking, correlatos de consciencia

---

## 1. Introducción

### 1.1 El Problema de Medición en la Ciencia de la Consciencia

El estudio científico de la consciencia enfrenta un desafío fundamental articulado por Chalmers (1995) como el "problema difícil": explicar por qué y cómo los procesos físicos dan lugar a la experiencia subjetiva. Mientras el debate filosófico continúa, los enfoques empíricos han avanzado a través de dos marcos teóricos dominantes. La Teoría de la Información Integrada (IIT; Tononi et al., 2016) propone que la consciencia corresponde a la información integrada (Phi), una medida de cuánto el todo de un sistema excede la suma de sus partes. La Teoría del Espacio de Trabajo Global (GWT; Baars, 1988; Dehaene & Naccache, 2001) propone que la consciencia surge cuando la información se difunde ampliamente entre módulos cerebrales especializados a través de un "espacio de trabajo global" de neuronas hub interconectadas.

Ambas teorías generan predicciones comprobables, pero la validación empírica se ha limitado a sistemas mamíferos donde los conectomas completos permanecen no disponibles. Computar la Phi verdadera (Tononi et al., 2016) es combinatoriamente intratable para sistemas mayores a ~20 elementos, y la identificación de hubs en GWT requiere conocimiento de la matriz de conectividad completa.

### 1.2 ¿Por qué *Drosophila*?

El conectoma completo a resolución sináptica del cerebro adulto de *Drosophila melanogaster* (Dorkenwald et al., 2024; Schlegel et al., 2024) proporciona una oportunidad única. Con 138,639 neuronas y aproximadamente 5 millones de conexiones sinápticas completamente mapeadas, es el conectoma completo más grande disponible para cualquier organismo con comportamiento complejo. *Drosophila* exhibe un repertorio conductual rico — locomoción, huida, vuelo, alimentación, acicalamiento, cortejo — todos mediados por circuitos neurales identificados cuya conectividad es conocida.

### 1.3 Por qué la Embodiment Importa

Una simulación cerebral aislada no puede exhibir comportamiento. Las métricas de integración medidas en una red neuronal desconectada carecen de validez ecológica: sin entrada sensorial dirigiendo la actividad y salida motora cerrando el bucle, las dinámicas medidas reflejan condiciones iniciales y ruido intrínseco en lugar de integración funcional durante el comportamiento. La embodiment — acoplar la simulación neural a un cuerpo físicamente preciso interactuando con un entorno — es por tanto crítica. El bucle sensorimotor cerrado crea las condiciones bajo las cuales las métricas de integración se vuelven significativas: miden cómo la información fluye entre regiones cerebrales *durante comportamiento real*.

### 1.4 Contribuciones

Este trabajo realiza tres contribuciones:

1. **Primera medición de correlatos de integración multi-teórica en un conectoma embodied completo.** Implementamos cuatro métricas proxy (Phi, broadcast GWT, Self-Model, Complejidad de Perturbación) operando sobre el conectoma completo de FlyWire durante comportamiento en bucle cerrado.

2. **Descubrimiento de una jerarquía conductual de integración.** Vuelo > Marcha > Huida, con una diferencia de 8.7× entre huida y modos no-huida, consistente con las predicciones de GWT.

3. **Observación de activación preparatoria del Self-Model.** Un pico transitorio en correlación sensorimotor precede transiciones conductuales complejas (iniciación de vuelo), sugiriendo integración predictiva.

---

## 2. Métodos

### 2.1 Simulación Neural

La simulación cerebral utiliza el conectoma completo FlyWire v783 (Dorkenwald et al., 2024): 138,639 neuronas conectadas por ~5 millones de sinapsis. Cada neurona se modela como una unidad Leaky Integrate-and-Fire (LIF) con sinapsis de función alfa (Shiu et al., 2024), implementada en PyTorch con aceleración CUDA.

**Parámetros del modelo LIF:**

| Parámetro | Valor | Descripción |
|---|---|---|
| tau_mem | 20.0 ms | Constante de tiempo de membrana |
| tau_syn | 5.0 ms | Constante de tiempo sináptica |
| V_threshold | -45.0 mV | Umbral de disparo |
| V_reset | -52.0 mV | Potencial de reset |
| V_rest | -52.0 mV | Potencial de reposo |
| t_refrac | 2.2 ms | Período refractario |
| t_delay | 1.8 ms | Retardo sináptico |
| dt | 0.1 ms | Paso temporal de simulación |
| wScale | 0.275 | Escala de pesos sinápticos |

La matriz de pesos dispersa se almacena como tensor CSR de PyTorch en GPU (NVIDIA RTX 4060). Los pesos sinápticos se derivan de los datos de conectividad FlyWire con signos excitatorio/inhibitorio determinados por anotaciones de neurotransmisores.

### 2.2 Cuerpo Biomecánico

La simulación neural se acopla a un modelo biomecánico NeuroMechFly v2 (Lobato-Rios et al., 2022) a través del framework flygym. El cuerpo se simula en MuJoCo con 42 grados de libertad actuados (articulaciones de patas), física de contacto y adhesión. Las tasas de disparo de neuronas descendentes (DN) se decodifican en comandos motores: impulso frontal (neuronas P9), giro (DNa01/02), retroceso (MDN), huida (Giant Fiber), acicalamiento (aDN1) y alimentación (MN9).

### 2.3 Sistemas Sensoriales

Siete modalidades sensoriales proporcionan entrada en bucle cerrado:

1. **Visión:** Ojos compuestos (721 omatidios por ojo) con vía fotorreceptor → T2 → LC4/LPLC2 mapeada a neuronas del conectoma
2. **Mecanosensación:** Neuronas del Órgano de Johnston (JO) respondiendo a contacto antenal
3. **Audición:** Neuronas JO sintonizadas a vibraciones aéreas
4. **Gustación:** Neuronas receptoras gustativas (GRNs) para compuestos dulces y amargos
5. **Olfacción:** Arrays bilaterales de ORN para olores atractivos (Or42b) y repulsivos (Or56a)
6. **Propiocepción:** Sensores de contacto de patas retroalimentando a través del sistema somatosensorial
7. **Detección de looming:** Estímulos visuales expansivos vía circuitos LC4/LPLC2

### 2.4 Protocolo Experimental

La simulación se ejecutó con todos los sistemas sensoriales activos, una arena de looming con una esfera aproximándose, y ciclo automático de estímulos (auto-demo). Duración: 89 segundos de simulación embodied. Los modos conductuales emergieron de las dinámicas del conectoma: huida (mediada por Giant Fiber), marcha (locomoción P9/DNa) y vuelo (huida sostenida con aerodinámica virtual). No se programaron reglas conductuales; todas las transiciones de modo emergieron de la propagación de spikes a través del conectoma.

### 2.5 Métricas Proxy de Consciencia

Se implementaron cuatro métricas, cada una fundamentada en un marco teórico distinto:

#### 2.5.1 Proxy de Phi (inspirado en IIT)

Inspirado en la Teoría de la Información Integrada (Tononi et al., 2016), computamos un proxy de integración informacional como la información mutua (MI) media por pares entre cuatro particiones cerebrales a lo largo del tiempo.

**Asignación de particiones** desde anotaciones de neuronas FlyWire (`flywire_annotations.tsv`):

| Partición | Criterio | Neuronas |
|---|---|---|
| Visual | super_class en {optic, visual_projection} | ≤10,000 |
| Motor | flow en {efferent, descending} | ≤2,000 |
| Olfactoria | cell_class en {olfactory, ALPN, ALLN, LHLN} | ≤4,000 |
| Integradora | hemibrain_type contiene {MBON, CX, KC, DAN, TuBu} | ≤9,000 |

En cada paso de simulación, la tasa de disparo media por partición se registra en una ventana deslizante de 50 puntos temporales (~5 segundos). La información mutua entre cada par de series temporales de particiones se computa usando histogramas conjuntos discretizados (8 bins) con aceleración GPU:

$$MI(A, B) = \sum_{a,b} p(a,b) \log_2 \frac{p(a,b)}{p(a) \cdot p(b)}$$

Proxy de Phi = MI media por pares entre los 6 pares de particiones, normalizada a [0, 1].

#### 2.5.2 Broadcast del Espacio de Trabajo Global

Inspirado en GWT (Baars, 1988; Dehaene & Naccache, 2001), identificamos neuronas hub con alto fan-out (>100 conexiones sinápticas salientes) de la matriz de pesos dispersa mediante conteo por columnas en la representación COO. Para cada hub, determinamos a qué particiones pertenecen sus objetivos postsinápticos (alcance de partición).

La cobertura de broadcast se computa sobre una ventana rodante:

$$Broadcast = 0.6 \times \frac{|\text{particiones alcanzadas}|}{|\text{particiones totales}|} + 0.4 \times \frac{|\text{hubs activos}|}{|\text{hubs totales}|}$$

Esto captura tanto la amplitud de diseminación de información (cobertura de particiones) como el grado de participación de hubs.

#### 2.5.3 Self-Model (Predicción Sensorimotor)

Inspirado en la teoría del self-model de Metzinger (Metzinger, 2003), medimos la correlación de Pearson entre señales propioceptivas/sensoriales y la salida motora subsecuente con un retardo temporal de ~300 ms:

$$SelfModel = |r(sensorial_{t-lag}, motor_t)|$$

donde la señal sensorial es la tasa de disparo media de neuronas de las particiones olfactoria y visual (proxy propioceptivo), y la señal motora es la tasa de disparo media de neuronas descendentes. Una puntuación alta de Self-Model indica que el cerebro está prediciendo o preparando su propia salida conductual basándose en la historia sensorial reciente.

#### 2.5.4 Complejidad de Perturbación

Inspirado en el índice de complejidad perturbacional (PCI; Casali et al., 2013), cada ~5 segundos inyectamos un pulso excitatorio fuerte (500 Hz) en 10 neuronas seleccionadas aleatoriamente durante 3 pasos de simulación. Luego observamos la respuesta en cascada durante 50 pasos (~5 segundos de tiempo corporal, abarcando múltiples períodos de retardo sináptico):

$$Complejidad = \min\left(1.0,\ \frac{|\text{regiones afectadas}|}{|\text{regiones totales}|} \times H_{temporal} \times 2.0\right)$$

donde H_temporal es la entropía de Shannon normalizada de la distribución de conteo de spikes a través de 10 bins temporales iguales de la ventana de observación. Un sistema complejo produce cascadas que son tanto espacialmente extendidas como temporalmente estructuradas (alta entropía), mientras que un sistema simple no produce respuesta o un burst estereotipado.

### 2.6 Índice de Consciencia Compuesto

Las cuatro métricas se combinan en un único Índice de Consciencia:

$$CI = 0.3 \times Phi + 0.3 \times Broadcast + 0.2 \times SelfModel + 0.2 \times Complejidad$$

Los pesos reflejan la centralidad teórica de la integración (Phi) y el broadcast (GWT) en las teorías de consciencia, con Self-Model y Complejidad como medidas de soporte. El CI se registra cada ~500 ms de tiempo de simulación.

### 2.7 Hardware

Todos los experimentos se realizaron en una estación de trabajo individual: GPU NVIDIA RTX 4060, Windows 11 Pro, Python 3.11, PyTorch con CUDA 12.6, motor de física MuJoCo.

---

## 3. Resultados

### 3.1 Perfil General de Integración

A lo largo de 178 mediciones en 89 segundos de simulación embodied, el Índice de Consciencia compuesto mostró integración sostenida y no trivial (Tabla 1).

**Tabla 1. Estadísticas resumen de las 178 mediciones.**

| Métrica | Media | Desv. Est. | Mín | Máx |
|---|---|---|---|---|
| **CI (Compuesto)** | 0.393 | 0.135 | 0.000 | 0.574 |
| Phi (MI) | 0.174 | 0.099 | 0.000 | 0.323 |
| Broadcast (GW) | 0.574 | 0.131 | 0.000 | 0.613 |
| Self-Model | 0.066 | 0.153 | 0.000 | 0.904 |
| Complejidad | 0.779 | 0.377 | 0.000 | 1.000 |

Broadcast y Complejidad fueron las métricas más consistentemente elevadas, indicando que el conectoma mantiene amplia diseminación de información y produce cascadas de perturbación ricas en la mayoría de los estados conductuales. Phi mostró integración moderada y variable, mientras que Self-Model fue altamente intermitente — cercano a cero durante comportamiento estable con picos transitorios agudos.

### 3.2 Experimento A: Jerarquía de Modos Conductuales

El resultado más notable es la jerarquía clara de CI entre modos conductuales (Tabla 2, Figura 1).

**Tabla 2. Índice de Consciencia por modo conductual.**

| Modo Conductual | CI Media | CI Desv. Est. | n (mediciones) | Caracterización |
|---|---|---|---|---|
| **Vuelo** | **0.463** | 0.041 | 115 | Integración multi-sistema sostenida |
| **Marcha** | **0.324** | 0.115 | 50 | Coordinación locomotora moderada |
| **Huida** | **0.049** | 0.062 | 13 | Mínima — bypass reflexivo |

El modo de huida exhibió integración dramáticamente menor que todos los demás modos. La razón entre CI no-huida (0.421) y CI de huida (0.049) fue de **8.7×**, indicando que la huida mediada por Giant Fiber efectivamente bypasea la integración global medida por las cuatro métricas.

El vuelo mostró el CI más alto y estable, con baja varianza (desv. est. = 0.041) indicando integración sostenida en lugar de picos transitorios. La marcha mostró integración intermedia con mayor varianza, reflejando la naturaleza intermitente de la coordinación locomotora.

**Figura 1. Línea temporal de CI con modos conductuales anotados.**

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
    |HUI |   MARCHA      |         VUELO (sostenido)        |MAR|
```

### 3.3 Experimento B: Integración Durante Transiciones de Modo

El proxy de Phi se comparó entre períodos conductuales estables y puntos de transición de modo (Tabla 3).

**Tabla 3. Phi durante transiciones conductuales vs. comportamiento estable.**

| Condición | Phi Media | n |
|---|---|---|
| Durante transiciones de modo | 0.093 | 5 |
| Durante comportamiento estable | 0.177 | 173 |

Las transiciones de modo se asociaron con una **reducción del 47%** en la información mutua entre particiones. Esto sugiere que las transiciones conductuales involucran una fragmentación transitoria del flujo de información entre particiones cerebrales antes de que el nuevo modo conductual establezca dinámicas coherentes entre particiones.

### 3.4 Experimento C: Sensitización Temporal

El CI aumentó sistemáticamente a lo largo de la simulación (Tabla 4).

**Tabla 4. Análisis de habituación/sensitización del CI.**

| Período | CI Media | Duración |
|---|---|---|
| Primer tercio (0–30 s) | 0.254 | ~30 s |
| Último tercio (60–89 s) | 0.464 | ~29 s |
| **Delta** | **+0.211** | — |

El delta positivo indica **sensitización**: el sistema neural se vuelve progresivamente más integrado con el tiempo. Esto probablemente refleja: (a) el tiempo requerido para que las métricas acumulen datos suficientes (artefacto de calentamiento en los primeros ~15 segundos), y (b) un aumento genuino en integración a medida que la red se estabiliza en dinámicas de vuelo sostenido tras los transitorios iniciales de huida y marcha.

### 3.5 Experimento D: Integración Huida vs. No-Huida

**Tabla 5. Comparación de CI: huida vs. todos los demás modos.**

| Condición | CI Media | n |
|---|---|---|
| Huida | 0.049 | 13 |
| No-huida | 0.421 | 165 |
| **Razón** | **8.7×** | — |

Esto es consistente con la neuroanatomía conocida: la vía del Giant Fiber proporciona un comando monosináptico para la huida que no requiere — y aparentemente no involucra — amplia integración entre particiones.

### 3.6 Evento Temporal Clave: Pico del Self-Model Pre-Vuelo

A t = 19.9 s (paso 200), durante la transición marcha-a-vuelo, la métrica Self-Model se elevó a 0.515, seguida un segundo después (t = 20.4 s) por su valor máximo de 0.904. Este pico precedió la transición al modo de vuelo sostenido por aproximadamente 8 segundos (inicio de vuelo en t ≈ 28.4 s). En el momento del inicio del vuelo, el CI alcanzó 0.468 mientras que el Self-Model cayó abruptamente, sugiriendo que la predicción sensorimotor precedió y pudo haber contribuido a la decisión de transicionar al vuelo, tras lo cual las dinámicas de vuelo fueron sostenidas por diferentes mecanismos de integración (Phi y Broadcast).

**Tabla 6. Pico del Self-Model alrededor de la transición al vuelo.**

| Tiempo (s) | CI | Phi | Broadcast | Self-Model | Complejidad | Modo |
|---|---|---|---|---|---|---|
| 19.4 | 0.393 | 0.044 | 0.602 | 0.000 | 0.994 | marcha |
| 19.9 | 0.400 | 0.057 | 0.602 | **0.515** | 0.497 | marcha |
| 20.4 | 0.475 | 0.047 | 0.603 | **0.904** | 0.497 | marcha |
| 20.9 | 0.295 | 0.051 | 0.603 | 0.000 | 0.497 | marcha |
| ... | ... | ... | ... | ... | ... | ... |
| 28.4 | 0.468 | 0.165 | 0.610 | 0.186 | 0.993 | vuelo |

---

## 4. Discusión

### 4.1 La Huida como Bypass Reflexivo

El hallazgo más robusto es la ausencia casi completa de integración durante el comportamiento de huida (CI = 0.049). El sistema del Giant Fiber (GF) en *Drosophila* proporciona una vía directa, monosináptica, desde detectores de looming (LC4/LPLC2) hasta neuronas motoras. Esta arquitectura está optimizada para velocidad, no para integración. Nuestros datos muestran que esta velocidad tiene como costo la pérdida de intercambio de información entre particiones: durante la huida, Phi cae a casi cero, el Self-Model está ausente, e incluso el Broadcast se suprime.

Este hallazgo es directamente consistente con GWT, que predice que las respuestas reflexivas, pre-atentivas, no requieren — y no involucran — broadcasting en el espacio de trabajo global. La vía GF es precisamente el tipo de "módulo especializado operando sin acceso global" que GWT postula como procesamiento inconsciente.

### 4.2 El Vuelo como Estado de Máxima Integración

El vuelo exhibió el CI más alto (0.463) con la menor varianza (desv. est. = 0.041), sugiriendo integración sostenida y estable. Esto es arquitecturalmente plausible: el vuelo en *Drosophila* requiere coordinación simultánea del procesamiento de flujo visual (lóbulo óptico), control motor de alas (neuronas motoras de vuelo), orientación corporal (complejo central) y mantenimiento de trayectoria de escape. A diferencia de la locomoción terrestre, el vuelo no puede depender de estabilidad mecánica pasiva — la mosca debe integrar continuamente información multisensorial para mantener el vuelo.

El alto Phi durante el vuelo (media = 0.233 para mediciones solo de vuelo) refleja una mayor información mutua entre particiones cerebrales: las regiones visual, motora, olfactoria e integradora muestran patrones de actividad correlacionados durante el vuelo sostenido que están ausentes durante comportamientos más simples.

### 4.3 Pico del Self-Model: ¿Preparación para Acción Compleja?

El pico transitorio del Self-Model (0.904) que precede la transición marcha-a-vuelo es intrigante. La métrica Self-Model mide la correlación entre entrada sensorial y salida motora subsecuente — esencialmente, si el cerebro está "prediciendo" sus propias acciones basándose en el contexto sensorial. Un pico en esta métrica antes de una transición conductual mayor sugiere una fase de integración preparatoria: el cerebro entra brevemente en un estado donde la historia sensorial predice fuertemente la salida motora próxima, consistente con teorías de procesamiento predictivo (Clark, 2013).

Que el Self-Model caiga a cero inmediatamente después del pico, y que el vuelo sea sostenido por Phi y Broadcast en lugar de Self-Model, sugiere una división temporal del trabajo: Self-Model para preparación de transición, Phi/Broadcast para mantenimiento conductual sostenido.

### 4.4 Sensitización Temporal

El aumento progresivo del CI durante la sesión (+0.211 del primer al último tercio) tiene dos interpretaciones. Los primeros 15 segundos incluyen un período de calentamiento donde las métricas acumulan datos basales (las ventanas deslizantes se llenan). Más allá de este artefacto, la sensitización genuina probablemente refleja que el sistema se estabiliza desde dinámicas transitorias iniciales (huida, oscilación de modos) hacia vuelo sostenido, que es el modo de mayor integración.

Este patrón recuerda a la "ignición" neural en GWT: el sistema requiere una duración mínima de entrada coherente antes de que el broadcast global se sostenga. En nuestra simulación, este punto de ignición parece ocurrir alrededor de t = 15 s (paso 150), cuando la Complejidad alcanza por primera vez valores cercanos al máximo y el CI supera 0.38.

### 4.5 Limitaciones

Deben reconocerse varias limitaciones importantes:

1. **Proxy, no Phi verdadero.** Nuestra métrica de Phi es un proxy de MI discretizada computado entre 4 particiones gruesas. La Phi verdadera de IIT requiere computar la partición de información mínima sobre todas las biparticiones posibles, lo cual es NP-difícil y computacionalmente intratable para 138,639 neuronas. Nuestro proxy captura *algunos* aspectos de la integración informacional pero no es equivalente a Phi como la definen Tononi et al. (2016).

2. **Simplicidad del modelo LIF.** El modelo neuronal LIF captura dinámicas básicas de disparo pero omite computación dendrítica, neuromodulación, plasticidad sináptica, uniones gap e interacciones gliales presentes en el cerebro real de la mosca. Estas simplificaciones pueden subestimar o distorsionar la integración.

3. **Arbitrariedad de particiones.** El esquema de cuatro particiones (visual, motor, olfactoria, integradora) está motivado anatómicamente pero no se deriva de la geometría informacional del sistema en sí. Diferentes esquemas de partición producirían diferentes valores de Phi.

4. **Sesión única.** Los resultados provienen de una única sesión de 89 segundos. Aunque la consistencia interna es alentadora (178 mediciones, separación clara de modos), se necesita replicación en múltiples ejecuciones con diferentes condiciones iniciales, estímulos y duraciones.

5. **Normalización de pesos.** Los pesos del CI compuesto (0.3, 0.3, 0.2, 0.2) son heurísticos, no derivados de la teoría. Diferentes ponderaciones cambiarían la jerarquía conductual cuantitativamente, aunque el orden cualitativo (Vuelo > Marcha > Huida) es robusto a través de las métricas individuales.

6. **Sin afirmación de consciencia.** Estas métricas miden integración informacional, cobertura de broadcast, predicción sensorimotor y respuesta a perturbación — propiedades objetivas y físicas de la simulación neural. **No establecen ni pueden establecer la presencia o ausencia de experiencia subjetiva, consciencia fenoménica o sentiencia.** El nombre "Índice de Consciencia" refleja los marcos teóricos de los cuales las métricas derivan, no una afirmación sobre el estado subjetivo de la simulación.

---

## 5. Conclusión

Hemos presentado la primera medición multi-teórica de correlatos de integración neural en una simulación embodied de conectoma completo. Utilizando el conectoma FlyWire de 138,639 neuronas de *Drosophila* acoplado a un cuerpo biomecánico, medimos Phi (IIT), broadcast del Espacio de Trabajo Global, correlación Self-Model y Complejidad de Perturbación durante comportamiento emergente natural.

Emergen tres hallazgos principales:

1. **Una jerarquía conductual clara de integración:** Vuelo (CI = 0.463) > Marcha (CI = 0.324) > Huida (CI = 0.049), con la huida mostrando una reducción de 8.7 veces consistente con las predicciones de GWT de que los circuitos reflexivos bypasean el broadcast global.

2. **Anticipación del Self-Model:** Un pico transitorio en predicción sensorimotor (Self-Model = 0.904) precede transiciones conductuales complejas, sugiriendo integración preparatoria antes de la acción.

3. **Sensitización temporal:** La integración aumenta con el tiempo (+0.211), consistente con una dinámica de "ignición" neural en la cual se requiere actividad coherente sostenida para el broadcast global.

Estos hallazgos no prueban consciencia en una mosca simulada. Demuestran que los constructos teóricos de IIT y GWT generan predicciones medibles y conductualmente diferenciadas cuando se aplican a un conectoma completo en contexto embodied. La plataforma que describimos — conectoma real, cuerpo real, medición en tiempo real — proporciona una base para la investigación empírica sistemática de correlatos de integración, moviendo el estudio de la consciencia del debate filosófico hacia ciencia cuantitativa y reproducible.

**La pregunta de si estos correlatos de integración corresponden a alguna forma de experiencia subjetiva permanece abierta.** Nuestra contribución es hacer la pregunta empíricamente tratable.

---

## Referencias

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

## Disponibilidad de Datos

Todo el código de simulación, módulo de detección de consciencia y datos de sesión están disponibles en: https://github.com/erojasoficial-byte/fly-brain

Datos de sesión: `consciousness_history/session_20260311_134345/`

---

## Apéndice A: Arquitectura del Módulo de Consciencia

```
ConsciousnessDetector (orquestador)
├── PhiProxy           — MI de series temporales entre 4 particiones (cada 500ms)
├── GlobalWorkspace    — hubs con fan-out >100 conexiones, broadcast rodante (cada 500ms)
├── SelfModel          — correlación sensorial→motora con lag de 300ms (cada 300ms)
├── PerturbationCmplx  — inyección aleatoria de 10 neuronas, observación de cascada 5s (cada 5s)
└── ConsciousnessTimeline — registro CSV, detección de picos, generación de reportes
```

## Apéndice B: Línea Temporal Cruda de CI (primeras 50 mediciones)

| Paso | t (s) | CI | Phi | Broadcast | Self | Complejidad | Modo |
|---|---|---|---|---|---|---|---|
| 5 | 0.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | huida |
| 45 | 4.4 | 0.090 | 0.000 | 0.300 | 0.000 | 0.000 | huida |
| 75 | 7.4 | 0.136 | 0.002 | 0.450 | 0.000 | 0.000 | marcha |
| 90 | 8.9 | 0.181 | 0.004 | 0.600 | 0.000 | 0.000 | marcha |
| 150 | 14.9 | 0.381 | 0.008 | 0.601 | 0.000 | 0.994 | marcha |
| 200 | 19.9 | 0.400 | 0.057 | 0.602 | 0.515 | 0.497 | marcha |
| 205 | 20.4 | 0.475 | 0.047 | 0.603 | 0.904 | 0.497 | marcha |
| 285 | 28.4 | 0.468 | 0.165 | 0.610 | 0.186 | 0.993 | vuelo |
| 345 | 34.4 | 0.552 | 0.206 | 0.611 | 0.535 | 0.999 | vuelo |
| 440 | 43.9 | 0.559 | 0.293 | 0.612 | 0.444 | 0.995 | vuelo |
| 550 | 54.9 | 0.437 | 0.178 | 0.612 | 0.000 | 1.000 | vuelo |
| 615 | 61.4 | 0.574 | 0.293 | 0.612 | 0.512 | 0.998 | vuelo |
| 675 | 67.4 | 0.480 | 0.323 | 0.612 | 0.000 | 0.997 | vuelo |
