# Escape Guiado Visualmente en una Simulación Embodied del Cerebro Completo de Drosophila

## Resumen

Presentamos la primera demostración de comportamiento de escape guiado visualmente que emerge de una simulación embodied del conectoma cerebral completo de *Drosophila melanogaster*. Nuestro sistema conecta un modelo de 138,639 neuronas de tipo integra-y-dispara con fugas (LIF), ejecutado en GPU mediante PyTorch, a una simulación biomecánicamente realista del cuerpo (NeuroMechFly v2) con visión de ojo compuesto (1,442 omatidios). Cuando una esfera oscura se aproxima a la mosca virtual, señales de contraste visual se inyectan en neuronas T2 de la lobula identificadas a partir de las anotaciones de FlyWire. La señal se propaga a través de la matriz real de pesos sinápticos del conectoma para activar las neuronas descendentes Giant Fiber, desencadenando locomoción de escape — sin reglas de comportamiento programadas manualmente. La mosca camina, ve la amenaza que se aproxima y huye. Hasta donde sabemos, este es el primer sistema de lazo cerrado donde el escape desencadenado visualmente emerge de la dinámica del conectoma cerebral completo en un agente embodied.

## 1. Introducción

El conectoma sináptico completo del cerebro adulto de *Drosophila melanogaster* ha sido mapeado (Dorkenwald et al., 2024; Schlegel et al., 2024), comprendiendo aproximadamente 139,000 neuronas y 50 millones de conexiones sinápticas. Por separado, se han desarrollado simulaciones biomecánicamente realistas del cuerpo de la mosca (NeuroMechFly; Lobato-Rios et al., 2022), que permiten locomoción basada en física impulsada por comandos neuronales. Sin embargo, conectar una simulación neuronal del cerebro completo a un cuerpo biomecánico — creando una mosca virtual embodied que percibe, procesa y actúa a través de sus circuitos neuronales reales — ha permanecido como un desafío abierto.

La respuesta de escape al estímulo de aproximación (looming) es uno de los circuitos visuomotores mejor caracterizados en *Drosophila*. Un objeto oscuro que se aproxima activa neuronas columnares de la lobula (LC4, LPLC2), las cuales excitan las neuronas descendentes Giant Fiber (GF), desencadenando un salto rápido de escape e iniciación de vuelo (von Reyn et al., 2014; Ache et al., 2019). Este circuito abarca toda la jerarquía de procesamiento visual: fotorreceptores (R1-R8) → lamina (L1/L2) → medulla (Mi1, Tm1/Tm2) → lobula (T2, T4/T5) → neuronas de proyección visual (LC4) → neuronas descendentes (GF) → circuitos motores.

Aquí cerramos el lazo: una mosca virtual con ojos compuestos ve una esfera oscura que se aproxima en una simulación de física 3D, la señal visual entra al conectoma cerebral completo a nivel de la lobula, se propaga a través de pesos sinápticos reales para activar la Giant Fiber, y los comandos motores resultantes impulsan la locomoción biomecánica de escape.

## 2. Métodos

### 2.1 Simulación Neuronal

El modelo cerebral comprende 138,639 neuronas de tipo integra-y-dispara con fugas (LIF) con sinapsis de función alfa, simuladas en GPU usando PyTorch. La matriz de conectividad se deriva del conectoma cerebral completo de FlyWire (Dorkenwald et al., 2024), con pesos sinápticos de la columna `Excitatory × Connectivity` del conjunto de datos publicado. Parámetros clave: constante de tiempo de membrana τ_mem = 20 ms, constante de tiempo sináptica τ_syn = 5 ms, umbral de disparo V_th = −45 mV, potencial de reposo V_rest = −52 mV, período refractario t_ref = 2.2 ms, paso temporal de simulación dt = 0.1 ms.

La entrada externa se entrega mediante generadores de spikes de Poisson con factor de escala de amplitud S = 250. Las tasas de disparo se establecen por neurona; en cada paso temporal, una prueba de Bernoulli con probabilidad p = tasa × dt/1000 determina si se genera un spike de Poisson de amplitud S.

### 2.2 Simulación del Cuerpo

El cuerpo de la mosca se simula usando NeuroMechFly v2 (flygym; Lobato-Rios et al., 2022) con el motor de física MuJoCo. El modelo corporal incluye seis patas articuladas (7 grados de libertad cada una, 42 en total), almohadillas tarsales adhesivas y sensores de contacto. La locomoción es gobernada por un HybridTurningController que acepta señales de impulso izquierda/derecha y genera patrones coordinados de marcha trípode. Paso temporal de simulación del cuerpo: 0.1 ms.

### 2.3 Visión de Ojo Compuesto

Cada ojo compuesto comprende 721 omatidios dispuestos en un arreglo hexagonal, simulados por el modelo de retina de flygym. La entrada visual se obtiene renderizando la escena 3D desde cámaras montadas en los ojos usando el renderizador nativo de MuJoCo (para evitar conflictos de contexto OpenGL con el visor interactivo en Windows). Las imágenes crudas de cámara pasan por corrección de ojo de pez y muestreo hexagonal de píxeles, produciendo un arreglo de observación (2, 721, 2) con valores en [0, 1] que representan intensidades de canales de fotorreceptores pale y yellow.

### 2.4 Pipeline de Procesamiento Visual

#### 2.4.1 El Problema de Desajuste de Escala

Un hallazgo crítico de este trabajo es que el modelo LIF del conectoma no puede propagar señales a través de múltiples capas sinápticas mediante transmisión de spikes de red. El generador de spikes de Poisson produce spikes con amplitud 250, escalados por wScale = 0.275 para una entrada efectiva de 68.75 por spike. En contraste, los spikes de red tienen amplitud 1, produciendo solo peso × 0.275 ≈ 0.6–2.0 por spike para conexiones típicas de la vía visual. Este desajuste de escala de 50× significa que incluso un disparo fuerte en una capa (por ejemplo, medulla a 40 Hz) produce una entrada postsináptica despreciable comparada con la estimulación directa de Poisson.

Verificamos esto cuantitativamente: el peso sináptico promedio de Mi1 → T2 es 2.34 (7,589 conexiones), mientras que Tm1 → T2 promedia 1.24 (1,629 conexiones). Al ejecutar el modelo cerebral con neuronas L1/L2 de la lamina estimuladas directamente a tasas biológicamente apropiadas (L1: 137–200 Hz, L2: 20–83 Hz), se produjeron cero spikes en neuronas T2, LC4 y GF después de 500 ms de simulación. La señal muere en la frontera medulla–lobula.

#### 2.4.2 Inyección de T2 Basada en Contraste

Basándonos en el análisis de pesos, inyectamos tasas de Poisson a nivel de T2 en la lobula — la última etapa de procesamiento antes de las neuronas de proyección visual LC4. Las neuronas T2 (3,240 identificadas de las anotaciones de FlyWire, tipos T2 y T2a) responden al movimiento oscuro de campo pequeño en *Drosophila* biológica. Desde T2 en adelante, la señal se propaga a través de la **matriz real de pesos del conectoma** para alcanzar LC4 y la Giant Fiber.

La tasa de disparo de T2 para cada neurona se calcula a partir del contraste:

```
brillo = media(canal_pale, canal_yellow)  por omatidio
brillo_medio = media(brillo)  a través de todos los omatidios
contraste_crudo = (brillo_medio - brillo) / brillo_medio
contraste = max(0, contraste_crudo - 0.3)  # umbral filtra textura del suelo
tasa_T2 = 120 Hz × min(contraste, 1.0)
```

El umbral de contraste de 0.3 es necesario para filtrar la variación natural de luminancia del suelo de ajedrez de la arena (contraste ≈ 0.12), mientras se preserva la señal fuerte de contraste de la esfera oscura (contraste ≈ 1.0). Las neuronas T2 mapeadas a omatidios con brillo igual o superior a la media reciben tasa cero, produciendo cero actividad de red en ausencia de un estímulo oscuro.

#### 2.4.3 Identificación de Neuronas

Todas las neuronas de la vía visual fueron identificadas del conjunto de datos de anotaciones de neuronas de FlyWire (Schlegel et al., 2024) mediante coincidencia exacta de cell_type y separadas en poblaciones izquierda/derecha por la columna `side`. Conteos de neuronas en el conectoma:

| Capa | Tipo | Cantidad | Rol |
|------|------|----------|-----|
| Retina | R1-R8 | 10,580 | Fotorreceptores (identificados, no inyectados) |
| Lamina | L1 | 1,590 | Vía OFF (identificada, no inyectada) |
| Lamina | L2 | 1,696 | Vía ON (identificada, no inyectada) |
| Medulla | Mi1 | 1,582 | Intrínsecamente activa (identificada, no inyectada) |
| Medulla | Tm1 | 1,554 | Transmedular ON (identificada, no inyectada) |
| Medulla | Tm2 | 1,552 | Transmedular ON (identificada, no inyectada) |
| Lobula | T2/T2a | 3,240 | Movimiento OFF → **inyectadas con tasas de contraste** |
| Placa lobular | LC4 | 104 | Detector de looming (impulsada por conectoma) |
| Descendente | GF | 2 | Giant Fiber de escape (impulsada por conectoma) |

### 2.5 Puente Cerebro-Cuerpo

Las tasas de disparo de las neuronas descendentes (DN) se decodifican usando un estimador de ventana deslizante (ventana de 50 ms). La tasa normalizada de la Giant Fiber (0–1) determina el comportamiento de escape: cuando la tasa de GF excede el umbral de 0.3, el sistema entra en modo de escape con impulso máximo en dirección reversa (alejándose del estímulo que se aproxima). La locomoción de marcha es impulsada por neuronas descendentes P9 que reciben entrada constante de Poisson a 100 Hz.

**Histéresis de modo.** Para prevenir la oscilación rápida entre modos de comportamiento (que produce locomoción espasmódica), se aplica una duración mínima de modo de 300 ms. Las transiciones de modo solo se permiten cuando el modo actual ha estado activo durante al menos esta duración. Esto elimina el parpadeo caminata↔escape causado por fluctuaciones transitorias en las tasas de neuronas descendentes. El umbral de escape táctil se establece en 35 N para evitar disparar escape por las fuerzas de contacto normales al caminar (que pueden alcanzar 28 N durante pasos vigorosos).

### 2.6 Arena de Looming

Una esfera oscura (radio 6 mm, material oscuro con reflejos especulares) se aproxima a la mosca desde 120 mm a lo largo de un ángulo de aproximación configurable a 15 mm/s. La esfera está implementada como un cuerpo mocap de MuJoCo. La arena presenta un entorno visual naturalista: skybox degradado (azul a horizonte pálido), luz solar direccional cálida con proyección de sombras, y una superficie de suelo verde-marrón tipo ajedrez natural. Las zonas de sabor se renderizan como parches emisivos en el suelo (verde para azúcar, rojo para amargo) y las fuentes de olor como esferas brillantes con halos translúcidos. Etiquetas flotantes ("AZUCAR", "VENENO", "COMIDA", "PELIGRO") identifican cada zona.

**Protocolo de loom único.** La esfera se aproxima una sola vez. Cuando pasa detrás de la mosca, se estaciona en z = −100 mm (invisible), permitiendo que la respuesta de la Giant Fiber decaiga naturalmente a través del conectoma sin terminación artificial del estímulo. Este protocolo biológicamente fiel asegura que el aterrizaje es determinado por la propia dinámica neuronal del cerebro, no por un temporizador programado.

### 2.7 Visualización de Actividad Neuronal en Tiempo Real

Un monitor cerebral dedicado se ejecuta como un proceso separado (mediante `multiprocessing` de Python) para evitar conflictos de contexto OpenGL con MuJoCo. Recibe datos de actividad neuronal a través de una cola sin bloqueo y renderiza un mapa cerebral dorsal a 30 fps usando pygame. La visualización mapea 16 regiones cerebrales — organizadas en grupos funcionales (visual, looming, escape, motor, retroceso, acicalamiento, alimentación) — a posiciones 2D anatómicamente plausibles que representan los lóbulos ópticos y el cerebro central.

**Renderizado con glow gaussiano.** Cada región se renderiza como un gradiente radial continuo en lugar de círculos rellenos discretos. Tres capas gaussianas se componen por región: un halo exterior (σ = r × 1.8), un brillo interior (σ = r × 0.8), y un núcleo caliente (σ = r × 0.3) que se desplaza hacia el blanco a alta intensidad. Las superficies se pre-renderizan a 16 niveles discretos de intensidad por región (~272 superficies en total) usando operaciones de arrays numpy, y luego se transfieren por frame con mezcla aditiva (`BLEND_ADD`) para bloom natural. Este enfoque elimina la computación numpy por frame mientras preserva la calidad de gradiente suave.

**Suavizado temporal y animación de pulso.** Los valores crudos de intensidad pasan por suavizado exponencial (τ = 120 ms) para prevenir transiciones visuales abruptas. Las regiones activas exhiben adicionalmente una modulación sinusoidal de respiración a ~2.5 Hz con amplitud de ±8%. Cada región recibe una fase inicial aleatoria, produciendo pulsación asincrónica que transmite una sensación de dinámica neuronal distribuida en lugar de parpadeo sincronizado.

**Partículas de conexión.** El flujo de señal entre regiones conectadas se visualiza mediante partículas luminosas que viajan de origen a destino a lo largo de cada conexión. La tasa de generación de partículas es proporcional a la intensidad de la región fuente (0–4 partículas/s), con un máximo de 6 partículas por conexión. Cada partícula aparece gradualmente durante el primer 15% y se desvanece durante el último 15% de su trayectoria. Las líneas de conexión se renderizan como guiones animados (guión de 8 px, espacio de 5 px) cuya velocidad de desplazamiento escala con el nivel de actividad.

**Capas pre-renderizadas.** Tres capas estáticas se componen una vez durante la inicialización: (1) un patrón de cuadrícula hexagonal que proporciona un sustrato sci-fi sutil, (2) una silueta cerebral con gradiente radial (centro más brillante, bordes transparentes) renderizada a partir de tres elipses superpuestas mediante numpy, y (3) una capa de líneas de escaneo estilo CRT (1 px de línea oscura cada 3 píxeles) aplicada con mezcla sustractiva. Estas contribuyen a la estética holográfica sin costo computacional por frame.

**HUD y barra lateral.** Una pantalla de información superpuesta (HUD) muestra el tiempo de simulación, modo de comportamiento (caminando/escape/acicalamiento con código de color), tipo de estímulo, valores de impulso bilateral y asimetría de amenaza durante el escape. Una barra lateral muestra la actividad de grupos de neuronas descendentes (DN) como barras horizontales con relleno de gradiente y lecturas numéricas en tiempo real. Los elementos de texto usan renderizado con halo de glow multi-offset para legibilidad contra el fondo oscuro.

El pipeline de renderizado completo por frame comprende: (1) suavizar intensidades, (2) computar modulación de pulso, (3) transferir cuadrícula hexagonal, (4) transferir silueta cerebral, (5) dibujar conexiones con guiones animados, (6) actualizar y dibujar partículas, (7) dibujar glows de regiones mediante búsqueda en caché, (8) aplicar capa de líneas de escaneo, (9) dibujar HUD y barra lateral, (10) flip de display. Las operaciones de dibujo totales son aproximadamente 140 por frame a 30 fps, con overhead de CPU despreciable ya que todas las operaciones numpy ocurren solo durante la inicialización.

### 2.8 Integración Somatosensorial y Auditiva

Además de la visión, el sistema procesa dos modalidades sensoriales adicionales a través de las neuronas del Órgano de Johnston (JO), los aferentes mecanosensoriales y auditivos primarios en *Drosophila*.

**Tacto (mecanosensorial).** Las fuerzas de contacto de MuJoCo se leen de 36 sensores (6 patas × 6 segmentos: tibia + 5 segmentos tarsales) en cada paso temporal de simulación. Las fuerzas se agrupan por pata y se computa la magnitud máxima por pata. Las tres patas izquierdas (LF, LM, LH) se mapean a neuronas JO-E/C del hemisferio izquierdo, y las tres patas derechas (RF, RM, RH) a neuronas JO-E/C del hemisferio derecho. Se identificaron 428 neuronas JO de tacto (222 izquierdas, 206 derechas) de las anotaciones de FlyWire por coincidencia de prefijos cell_type: JO-E, JO-EDC, JO-EDM, JO-EDP, JO-EV, JO-EVL, JO-EVM, JO-EVP, JO-C, JO-CA, JO-CL y JO-CM. La magnitud de fuerza se mapea a tasa de disparo mediante:

```
exceso = max(fuerza - 0.3, 0)          # umbral filtra contacto normal del suelo
tasa = min(exceso / 9.7, 1.0) × 250 Hz
```

Fuerzas por debajo de 0.3 N (contacto normal al caminar) producen cero activación JO. Fuerzas moderadas (1.5–5.0 N) impulsan comportamiento de acicalamiento a través de la vía del conectoma JO-E → aDN1. Fuerzas que exceden 5.0 N (por ejemplo, colisión con la esfera que se aproxima) desencadenan escape táctil a nivel del puente cerebro-cuerpo, proporcionando una vía de respaldo cuando el conectoma no propaga la señal hasta la Giant Fiber.

**Sonido (auditivo).** Los subtipos JO-A y JO-B procesan vibración de campo cercano en *Drosophila* biológica: las neuronas JO-A están sintonizadas en frecuencia (respondiendo preferencialmente al canto de cortejo de ~200 Hz), mientras que las neuronas JO-B codifican la amplitud de vibración de forma amplia. Identificamos 390 neuronas JO auditivas (213 izquierdas, 177 derechas) de las anotaciones de FlyWire. Fuentes de vibración virtuales se colocan en la arena, cada una definida por posición, frecuencia y amplitud. La señal de vibración en cada antena se computa como:

```
atenuación = amplitud / (1 + (distancia / 40mm)²)
ganancia_frecuencia = exp(-0.5 × ((f - 200) / 80)²)
efectiva = atenuación × (0.6 × ganancia_frecuencia + 0.4)
tasa = efectiva × 200 Hz
```

La activación bilateral depende del ángulo de la fuente relativo al rumbo de la mosca, con neuronas ipsilaterales recibiendo tasas más altas (división ponderada por coseno con modulación de ±40%). Esta asimetría direccional permite la fonotaxis: la mosca se orienta hacia fuentes de frecuencia de cortejo a través de una combinación de vías JO → DN de giro mediadas por el conectoma y un sesgo de orientación a nivel del puente.

**Coexistencia multimodal.** Las tasas somatosensoriales se inyectan en el cerebro usando máximo elemento-a-elemento con las tasas existentes, asegurando que los estímulos manuales por teclado, las entradas visuales y las entradas mecanosensoriales/auditivas coexistan sin interferencia mutua. Las tres modalidades sensoriales (visión, tacto, sonido) pueden estar activas simultáneamente, con la selección de modo de comportamiento siguiendo la jerarquía de prioridad: escape > acicalamiento > caminata.

### 2.9 Sistema Gustativo

*Drosophila* detecta sustancias químicas a través de neuronas receptoras gustativas (GRN) ubicadas en los tarsos, probóscide y márgenes alares. Implementamos la gustación tarsal: cuando los efectores finales de las patas contactan zonas de sabor definidas en el suelo de la arena, las poblaciones GRN correspondientes se activan.

**Zonas de sabor.** Regiones circulares en el suelo de la arena se definen con posición central, radio e identidad de sabor (azúcar o amargo). Se renderizan como cilindros coloreados translúcidos en MuJoCo (verde para azúcar, rojo para amargo) con `conaffinity=0, contype=0` para evitar interferir con la física.

**Poblaciones GRN.** De las anotaciones del conectoma FlyWire, se identificaron 21 GRN de azúcar y 41 GRN de amargo. En cada paso de simulación, las seis posiciones de los efectores finales tarsales se verifican contra todas las zonas de sabor. Solo los pies cerca del nivel del suelo (z < 0.5 mm) se consideran en contacto. La tasa de disparo escala con el número de patas tocando una zona:

```
tasa = min(patas_en_zona / 2, 1.0) × tasa_máxima
```

donde tasa_máxima es 200 Hz para GRN de azúcar y 250 Hz para GRN de amargo. Dos o más patas en una zona producen activación completa.

**Efectos comportamentales.** La activación de GRN de azúcar se propaga a través del conectoma hasta las neuronas MN9 (motor de probóscide), desencadenando modo de alimentación — la mosca detiene la caminata y extiende la probóscide. La activación de GRN de amargo desencadena escape a nivel del puente, anulando otros comportamientos con la misma prioridad que el escape por Giant Fiber. La jerarquía de prioridad comportamental se actualiza a: escape (GF o amargo o táctil) > acicalamiento > alimentación > caminata.

**Coexistencia multimodal.** Las tasas gustativas se inyectan usando el mismo mecanismo de máximo elemento-a-elemento que las tasas somatosensoriales. Las cuatro modalidades sensoriales (visión, tacto, sonido, gusto) pueden estar activas simultáneamente mediante las banderas `--visual --somatosensory --gustatory`.

### 2.10 Sistema Olfativo

*Drosophila* navega paisajes químicos usando ~2,300 neuronas receptoras olfativas (ORNs) en las antenas, cada una sintonizada a volátiles específicos. Implementamos olfacción bilateral con dos poblaciones receptoras identificadas de las anotaciones de FlyWire:

- **ORN_DM1** (equivalente a Or42b, 68 neuronas): responde a volátiles relacionados con alimento (vinagre, ésteres frutales). Impulsa quimiotaxis atractiva.
- **ORN_DA2** (equivalente a Or56a, 39 neuronas): responde a geosmina (señal de peligro microbiano). Impulsa escape aversivo.

**Fuentes de olor y gradiente de concentración.** Fuentes de olor virtuales se colocan en la arena, cada una definida por posición, tipo (atractivo/repulsivo), amplitud y dispersión. La concentración sigue decaimiento inverso-cuadrado:

```
c(r) = amplitud / (1 + (r / dispersión)²)
```

Esto produce un gradiente suave que la mosca puede navegar. Las fuentes se renderizan como pequeñas esferas coloreadas en MuJoCo (verde = atractivo, púrpura = repulsivo).

**Detección bilateral por antenas.** Las posiciones de las antenas izquierda y derecha se computan desde la posición central y el rumbo de la mosca, con desplazamiento perpendicular a la dirección de avance de ±1.0 mm (ligeramente exagerado vs anatomía real para producir gradientes funcionales a escala de simulación). La concentración en cada antena se computa independientemente, y la asimetría bilateral impulsa la quimiotaxis:

```
sesgo_atracción = (c_derecha - c_izquierda) / (c_derecha + c_izquierda)
```

Este sesgo se amplifica por un factor de ganancia (4.0×) y se suma al impulso de giro de la mosca durante la caminata, causando que la mosca se oriente gradualmente hacia fuentes atractivas.

**Efectos comportamentales.** La activación de ORN atractivos (DM1) añade un sesgo de giro hacia la fuente de alimento durante modo de caminata — la mosca realiza quimiotaxis curvándose hacia concentraciones más altas. La activación de ORN repulsivos (DA2) por encima del umbral del 30% desencadena modo de escape con evitación direccional: la mosca gira lejos del lado con mayor concentración aversiva. La jerarquía de prioridad comportamental es: escape (GF o amargo o táctil o olfativo repulsivo) > acicalamiento > alimentación > caminata.

**Coexistencia multimodal.** Las tasas olfativas se inyectan usando máximo elemento-a-elemento. Las cinco modalidades sensoriales (visión, tacto, sonido, gusto, olfato) pueden estar activas simultáneamente mediante `--visual --somatosensory --gustatory --olfactory`.

### 2.11 Canto Alar (Vocalización)

Los machos de *Drosophila* producen canciones de cortejo extendiendo y vibrando un ala. El modelo NeuroMechFly tiene cuerpos alares estáticos (sin articulaciones de ala en el MJCF), por lo que implementamos la producción de canto virtualmente: los patrones de actividad de neuronas descendentes seleccionan el modo de canto, que emite una señal de vibración desde la posición de la mosca.

**Tipos de canto.** Tres modos basados en comunicación acústica real de *Drosophila*:
- **Canto de pulso** (200 Hz, amplitud 0.7): señal de cortejo de intervalo inter-pulso, la canción principal de apareamiento a corta distancia.
- **Canto sinusoidal** (160 Hz, amplitud 0.5): señal continua de baja frecuencia de cortejo de aproximación.
- **Zumbido de alarma** (400 Hz, amplitud 0.9): señal de estrés de alta frecuencia durante escape.

**Desencadenamiento.** El modo de canto se determina por las tasas de disparo DN en cada paso de simulación:
- Tasa de MN9 (alimentación/aproximación) por encima de 0.03 desencadena cortejo, alternando entre pulso (60% del ciclo) y sinusoidal (40%) en un período de 1 segundo.
- Tasa de GF (escape) por encima de 0.15 desencadena zumbido de alarma, anulando el cortejo.

**Auto-audición y señalización social.** El canto alar se emite como `VibrationSource` en la posición de la mosca, atenuado al 20% de amplitud para auto-audición. Esta fuente es procesada por las neuronas JO del sistema somatosensorial, creando un bucle sensoriomotor cerrado: actividad DN → producción de canto → detección JO → actividad cerebral. En un escenario multi-mosca, otras moscas detectarían la vibración a amplitud completa, habilitando interacción social acústica.

**Salida de audio.** El proceso del monitor cerebral genera tonos de audio en tiempo real via `pygame.mixer`, con ondas sinusoidales pre-renderizadas a 160, 200 y 400 Hz.

### 2.12 Sistema de Vuelo Virtual

NeuroMechFly v2 no tiene modo de vuelo — las alas son cuerpos de malla estáticos sin articulaciones ni actuadores. Implementamos vuelo virtual pragmáticamente usando el mecanismo `xfrc_applied` de MuJoCo para aplicar fuerzas externas al cuerpo del Tórax, simulando sustentación aerodinámica y empuje.

**Máquina de estados de vuelo.** Cuatro estados gobiernan el vuelo: EN_TIERRA → DESPEGUE → VOLANDO → ATERRIZANDO → EN_TIERRA. La tasa normalizada de la Giant Fiber (GF) excediendo 0.06 dispara el despegue; la GF cayendo por debajo de 0.06 inicia el aterrizaje. Un período de enfriamiento de 3 segundos post-aterrizaje previene el re-despegue inmediato por actividad residual de GF.

**Fuerzas traslacionales.** Durante el despegue, sustentación = 1.4 × mg (peso corporal) proporciona ascenso rápido; durante el vuelo, un controlador P mantiene la altitud objetivo (5 mm) con ganancia 0.12. El empuje horizontal es proporcional a la actividad GF/P9 a lo largo del rumbo de escape fijado. El arrastre aerodinámico (lineal 0.008 × mg por mm/s, vertical 0.015 × mg por mm/s) proporciona desaceleración natural. Durante el aterrizaje, la sustentación decrece con la altitud: `sustentación = 0.5 × mg × max(0, 1 - altitud/15mm)`, produciendo un descenso controlado.

**Control de orientación via override de quaternion.** El control de orientación basado en torques falla en cuerpos articulados con más de 100 articulaciones: los controladores PD de las articulaciones, las masas de las patas y la dinámica de contacto crean torques competitivos que abruman cualquier torque correctivo aplicado externamente, resultando en giro incontrolable. En su lugar, sobrescribimos directamente el quaternion de la articulación libre en `data.qpos` después de cada paso de física, fijando el cuerpo a una orientación vertical con el ángulo yaw de escape determinado al despegar. La velocidad angular (`data.qvel[3:6]`) se pone a cero simultáneamente. Esto garantiza fijación rígida del rumbo independientemente de la dinámica del cuerpo articulado.

**Vuelo de escape balístico.** Los vuelos de escape reales de *Drosophila* son balísticos — el rumbo se compromete al despegar y se mantiene durante todo el vuelo. El rumbo de escape se computa desde `fly_orientation` (el eje X del cuerpo en el marco mundo, obtenido via el sensor `framexaxis` de flygym) en el momento en que la GF excede el umbral.

**Congelamiento de patas durante el vuelo.** Las articulaciones de las patas se fijan a la pose neutral de pie (`groom_ctrl.neutral`) con adhesión desactivada durante el vuelo, previniendo torques asimétricos por oscilación de las patas.

### 2.13 Extensión de Probóscide

El modelo NeuroMechFly incluye cuerpos pasivos de Rostrum y Haustellum (los segmentos proximal y distal de la probóscide) con mallas STL de alta resolución, pero sin articulaciones ni actuadores. Añadimos una articulación de bisagra dinámicamente al cuerpo del Rostrum via la API MJCF de dm_control antes de la compilación del modelo:

```
articulación: tipo=bisagra, eje=[0,1,0] (pitch), rango=[-0.1, 1.2] rad
rigidez=50 (retorno por resorte), amortiguamiento=5
```

Cuando el puente cerebro-cuerpo entra en modo de alimentación (neuronas motoras MN9 activas via la vía GRN de azúcar → conectoma → MN9), el `qpos` de la articulación se establece en 1.0, extendiendo la probóscide hacia abajo. Cuando la alimentación cesa, la rigidez del resorte (50) la retrae automáticamente. Esto crea un bucle sensoriomotor completo: contacto tarsal con azúcar → spikes GRN → propagación por conectoma → activación MN9 → extensión de probóscide, sin reglas de comportamiento programadas más allá de la articulación biomecánica.

## 3. Resultados

### 3.1 Escape Desencadenado Visualmente

Cuando la esfera oscura se aproxima, la mosca exhibe una respuesta de escape gradual que escala con el tamaño angular de la amenaza:

| Tiempo (s) | Distancia de la bola (mm) | Tasa GF (normalizada) | Comportamiento | Posición x de la mosca (mm) |
|------------|--------------------------|----------------------|----------------|---------------------------|
| 0.2 | 75 | 0.000 | Caminando | +0.3 |
| 0.4 | 70 | 0.188 | Caminando | +0.7 |
| 0.6 | 65 | 0.300 | Caminando → Escape | +1.2 |
| 0.8 | 60 | 0.400 | Escape | +0.6 |
| 1.0 | 55 | 0.450 | Escape | −0.9 |
| 2.0 | 30 | 0.450 | Escape | −5.3 |
| 3.2 | 0 | 0.550 | Escape | −14.5 |

La mosca transiciona de marcha hacia adelante a escape en t ≈ 0.6 s cuando la bola está a 65 mm (diámetro angular ≈ 10.6°). Luego invierte dirección, moviéndose de x = +1.2 mm a x = −14.5 mm — un desplazamiento de 15.7 mm alejándose de la amenaza que se aproxima.

### 3.2 Verificación Causal

Para verificar que el escape es causalmente impulsado por la visión, probamos el modelo cerebral con condiciones de estímulo variables:

| Condición | Neuronas T2 activas | Tasa GF (Hz) |
|-----------|--------------------|--------------|
| Sin bola (suelo uniforme) | 0 | 0.0 |
| Bola pequeña (5% omatidios oscuros) | 162 | 50.0 |
| Bola mediana (20% oscuros) | 648 | 78.3 |
| Bola grande (50% oscuros) | 1,620 | 90.0 |

Con cero estimulación de T2, la GF produce cero spikes — confirmando que el escape es enteramente impulsado por la entrada visual, no por actividad espontánea.

### 3.3 Verificación de la Vía del Conectoma

Conectividad T2 → LC4: 425 conexiones sinápticas no nulas (peso medio 1.08), alcanzando 99 de 104 neuronas LC4. La activación T2 → GF procede a través de múltiples vías del conectoma, no exclusivamente a través de LC4. Cuando T2 se estimula a 100 Hz uniformemente, LC4 dispara a 18.5 Hz y GF a 80 Hz. La rica interconectividad del conectoma proporciona vías redundantes desde la lobula hasta las neuronas descendentes.

## 4. Discusión

### 4.1 Significancia

Este trabajo demuestra que el comportamiento visuomotor significativo puede emerger de la dinámica del conectoma cerebral completo en un agente embodied. La respuesta de escape no está programada — surge de la propagación de señales de contraste visual a través de 3,240 neuronas T2 de la lobula, a través de la matriz real de pesos sinápticos, hasta 2 neuronas descendentes Giant Fiber, que luego impulsan la locomoción biomecánica. El conectoma determina *cuáles* patrones de activación de T2 producen escape y *con qué intensidad* responde la Giant Fiber.

### 4.2 Limitaciones

**Capas de potencial gradual**: El sistema visual biológico desde la retina hasta la medulla usa potenciales graduales (analógicos), no spikes. El modelo LIF no puede capturar esto, necesitando inyección directa de tasas a nivel de T2 en lugar de en los fotorreceptores.

**Desajuste de escala**: La diferencia de amplitud de 50× entre la estimulación de Poisson y la transmisión sináptica de red es una limitación fundamental de la parametrización actual del modelo. Reducir la escala de Poisson o amplificar los pesos de la vía visual podría permitir una propagación de señal más profunda en trabajo futuro.

**Mapeo espacial**: El mapeo de omatidio a neurona usa distribución por ID ordenado en lugar de coordenadas retinotópicas verdaderas, que aún no están disponibles en las anotaciones del conectoma.

**Comportamiento visual único**: El circuito de escape al looming fue el comportamiento visual principal probado. Las entradas mecanosensoriales (tacto) y auditivas (vibración) a través del Órgano de Johnston ahora complementan la vía visual, proporcionando integración sensorial multimodal a través del mismo conectoma.

### 4.3 Observabilidad en Tiempo Real

El monitor cerebral proporciona retroalimentación visual en tiempo real de la dinámica neuronal durante la simulación. El mapa cerebral dorsal renderiza la actividad de regiones funcionales como campos de glow gaussiano con suavizado temporal, partículas animadas en las conexiones que muestran la dirección y magnitud del flujo de señal, y modulación de pulso que transmite la naturaleza distribuida del procesamiento neuronal. Esta visualización fue instrumental durante el desarrollo para diagnosticar problemas de propagación de señal (por ejemplo, verificar que la activación de T2 alcanzaba la GF a través del conectoma) y continúa sirviendo como herramienta de validación cualitativa: la cascada visual desde retina → T2 → LC4 → GF → neuronas motoras es inmediatamente aparente cuando la esfera de looming entra en el campo visual de la mosca. Al ejecutarse como un proceso separado, el monitor no introduce overhead de rendimiento al bucle principal de simulación.

### 4.4 Direcciones Futuras

- Reducir el factor de escala de Poisson para permitir propagación multicapa (retina → lamina → medulla → lobula)
- Agregar mapeo retinotópico usando datos de conectividad a nivel de columna
- Implementar detección de movimiento (T4/T5 → respuesta optomotora) para estabilización de curso
- Integración multisensorial: combinar escape visual con atracción/repulsión olfativa
- Aprendizaje de lazo cerrado: adaptar pesos sinápticos basándose en resultados comportamentales

## 5. Resumen de Métodos

| Componente | Implementación | Escala |
|------------|---------------|--------|
| Modelo cerebral | LIF + sinapsis alfa, PyTorch GPU | 138,639 neuronas |
| Conectoma | FlyWire (Dorkenwald et al., 2024) | ~50M sinapsis |
| Modelo corporal | NeuroMechFly v2 / MuJoCo | 42 GDL, 6 patas |
| Visión | Ojo compuesto, 721 omatidios/ojo | 1,442 total |
| Inyección visual | T2 lobula, basada en contraste | 3,240 neuronas |
| Vía del conectoma | T2 → LC4 → GF (pesos reales) | Verificada |
| Monitor cerebral | Glow gaussiano, partículas, pygame (proceso separado) | 20 regiones, 30 fps |
| Tacto (JO-E/C) | Fuerzas de contacto → tasas JO bilaterales | 428 neuronas |
| Sonido (JO-A/B) | Fuentes de vibración → tasas JO bilaterales | 390 neuronas |
| Gusto (GRN azúcar) | Contacto tarsal → tasas GRN de azúcar | 21 neuronas |
| Gusto (GRN amargo) | Contacto tarsal → tasas GRN de amargo | 41 neuronas |
| Olfato (ORN_DM1/Or42b) | Gradiente de olor → tasas ORN atractivas bilaterales | 68 neuronas |
| Olfato (ORN_DA2/Or56a) | Gradiente de olor → tasas ORN repulsivas bilaterales | 39 neuronas |
| Canto alar | Actividad DN → vibración virtual → auto-audición JO | 3 tipos de canto |
| Vuelo virtual | GF → fuerzas xfrc_applied + override de quaternion qpos | 4 estados |
| Probóscide | MN9 → articulación bisagra dinámica en cuerpo Rostrum | 1 GDL |
| Velocidad de simulación | ~800 pasos cerebrales/s (RTX 4060) | 0.1 ms/paso |

## Referencias

- Ache, J.M., et al. (2019). Neural basis for looming size and velocity encoding in the Drosophila giant fiber escape pathway. *Current Biology*.
- Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*.
- Lobato-Rios, V., et al. (2022). NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. *Nature Methods*.
- Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome cell type quantification. *Nature*.
- von Reyn, C.R., et al. (2014). A spike-timing mechanism for action selection. *Nature Neuroscience*.

---
