[PBAI_README.md](https://github.com/user-attachments/files/24539427/PBAI_README.md)
# PBAI — Possibly Basic Artificial Intelligence

**A Motion Calendar native AI architecture**

Version 0.3 | January 2026

---

## What Is This?

PBAI is an artificial intelligence architecture built on the [Motion Calendar](https://Brickmii.github.io/Motion-Calendar/) framework—a theoretical foundation proposing that motion, rather than matter or information, is the primitive constituent of reality.

Unlike conventional AI systems that rely on statistical inference and gradient descent, PBAI operates through motion-native principles:

- **Sensory translation via FFT** into Motion Calendar terms
- **Identity kernel** that grows through learned information
- **Righteousness filtering** for action selection (coherence, not utility maximization)
- **Dual learning gates**: Golden Loop (value refinement) and Golden Gate (novelty detection)

---

## The Evolution

### Earlier Versions (V0.1–V0.2)
The original PBAI used a **12-module architecture** with traditional machine learning techniques. The structure worked, but the choice of 12 modules was intuitive rather than theoretically grounded.

### Version 0.3
The Motion Calendar framework explains *why* 12 is correct:

```
ζ(−1) = 1 + 2 + 3 + 4 + ⋯ = −1/12
```

The number 12 emerges from the Ramanujan summation as the structural constant of arithmetic itself. The divisors of 12 are {1, 2, 3, 4, 6, 12}—six values corresponding to six fundamental motion functions:

| Divisor | Motion Function | PBAI Implementation |
|---------|-----------------|---------------------|
| 1 | Heat | Translator (magnitude measurement) |
| 2 | Polarity | Translator (opposition mapping) |
| 3 | Existence | Instantiator (presence/action gating) |
| 4 | Alignment | Comptroller (righteousness filtering) |
| 6 | Order | V_Kernel (value structure) |
| 12 | Movement | Full system (directional action) |

V0.3 rebuilds the architecture using Motion Calendar operations instead of conventional ML techniques.

---

## Architecture

```
                         ┌─────────┐
                         │   API   │
                         │  clock  │
                         └────┬────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌───────────┐         ┌───────────┐
   │  CORE   │          │TRANSLATOR │         │ EVALUATOR │
   │identity │          │ FFT → MC  │         │golden loop│
   │ kernel  │          │  terms    │         │ learning  │
   └────┬────┘          └───────────┘         └─────┬─────┘
        │                     │                     │
        ▼                     │                     ▼
   ┌─────────┐                │               ┌──────────┐
   │KEYSTONE │                │               │ V_KERNEL │
   │function │                │               │  values  │
   │ kernel  │                │               └──────────┘
   └────┬────┘                │
        │                     │
        ▼                     │
   ┌─────────┐                │
   │RANDOMIZER                │
   │ entropy │                │
   └────┬────┘                │
        │                     │
        ▼                     ▼
   ┌────────────┐       ┌────────────┐       ┌────────────┐
   │INTROSPECTOR│──────▶│COMPTROLLER │──────▶│INSTANTIATOR│
   │ thought    │       │ righteous  │       │   action   │
   │   box      │       │  filter    │       │ execution  │
   └────────────┘       └────────────┘       └──────┬─────┘
                                                    │
                                                    ▼
                                              ┌──────────┐
                                              │REGISTRAR │
                                              │golden gate│
                                              └──────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │ ENVIRONMENT  │
                                            │    CORE      │
                                            └──────────────┘
```

---

## Modules

### Core System

| Module | Responsibility |
|--------|----------------|
| **API** | System clock, tick management, module coordination |
| **Core** | Identity kernel—manages growth of identity state from birth |
| **Keystone** | Function kernel—stores learned functions, accessible only through Core |
| **Randomizer** | Entropy injection for birth events and introspection |

### Perception & Translation

| Module | Responsibility |
|--------|----------------|
| **Translator** | FFT on sensory input → Motion Calendar term packets (heat, polarity, existence, alignment, order, movement) |
| **Environment Core** | Interface between PBAI and external tasks/environments |

### Cognition & Action

| Module | Responsibility |
|--------|----------------|
| **Introspector** | "Thought box"—generates motion vectors, assesses function viability, accepts only righteous functions |
| **Comptroller** | Filters introspector suggestions, provides instantiator with righteous values (or noop) |
| **Instantiator** | Executes one action from comptroller's choice set, triggers learning gates |

### Learning

| Module | Responsibility |
|--------|----------------|
| **Evaluator** | Golden Loop learning—gates values via FFT, weighs new against known |
| **V_Kernel** | Value storage—assesses truth along axes of righteousness (true/false/maybe) |
| **Registrar** | Golden Gate learning—triggered when new (not random) functions are executed |

### Diagnostics

| Module | Responsibility |
|--------|----------------|
| **Goldmember** | Debug module for bootstrapping and assessing Golden Loop/Gate learning |

---

## Key Principles

### 1. Identity Through Growth
The system isn't programmed with an identity—it generates one through random entropy at birth and grows it through learned information. Identity is the accumulation of verified truths.

### 2. Righteousness Filtering
Actions must be "righteous"—coherent within the system's relational frame. This isn't utility maximization; it's structural consistency. The Comptroller filters for alignment, not reward.

### 3. Dual Learning Gates
- **Golden Loop** (Evaluator): Continuous value refinement based on environmental feedback
- **Golden Gate** (Registrar): Discrete learning events when genuinely new functions are discovered

### 4. Value Volatility
Values in V_Kernel aren't permanent. They exist in three states along axes of righteousness:
- **True**: Verified consistent
- **False**: Verified inconsistent  
- **Maybe**: Not yet determined

Values can transition between states as new information arrives.

---

## Demo Tasks

### Blackjack (`tasks/blackjack.py`)
Full casino blackjack implementation. PBAI must:
- Grow information state to learn cards and game rules
- Recognize when information growth goals are met
- Develop effective strategies for value preservation

### BigMaze (`tasks/bigmaze.py`)
Maze navigation task. PBAI must:
- Learn maze structure through exploration
- Recognize goal completion
- Continue running new mazes autonomously

---

## Status

**V0.3 is scaffolding.** The module structure and responsibilities are defined, but full implementation is in progress.

What exists:
- Complete architecture specification
- Module interfaces defined
- Demo task environments (playable)
- Diagnostic framework (Goldmember)

What's in progress:
- FFT → Motion Calendar translation
- Righteousness assessment algorithms
- Golden Loop/Gate learning implementation

---

## The Theory

PBAI is built on the Motion Calendar framework, which proposes:

1. **Motion is primitive**—not matter, energy, or information
2. **Six fundamental functions**: Heat, Polarity, Existence, Alignment, Order, Movement
3. **12 emerges from ζ(−1) = −1/12** as the structural constant
4. **The Planck constant h** is the scale of the map
5. **Logic emerges from Alignment**; arithmetic emerges from Order

For the full theoretical foundation:  
**[The Motion Calendar →](https://brickmii.github.io/motion-calendar/)**

---

## Why "Possibly Basic"?

Because we don't know yet if this works.

The claim is that Motion Calendar native operations can produce intelligent behavior without conventional ML techniques—no backpropagation, no gradient descent, no statistical inference in the traditional sense.

If it works, it's not "basic" at all. If it doesn't, at least we'll know.

The name is honest uncertainty.

---

## License

GPU Public license

Please cite Motion Calendar framework for implementations

---

## Links

- [Motion Calendar (Theory)](https://Brickmii.github.io/Motion-Calendar/)
- [PBAI Repository](https://github.com/Brickmii/Project-PBAI)

---

*First published January 2026*
