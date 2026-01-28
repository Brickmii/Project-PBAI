# PBAI Thermal Manifold

**A Universe OF Motion, Not Objects IN Motion**

This is an implementation of the Motion Calendar framework - a thermal manifold system where cognition emerges as heat flow through a self-organizing hyperspherical structure.

## Core Concept

**PBAI is the agent.** It experiences the environment, builds structure, and makes choices.  
**Claude (Haiku) is a tool** for generating options—it never chooses, never experiences.

Everything—perception, memory, reasoning, identity—emerges from patterns of heat flow through structured motion functions.

## The Six Motion Functions

Every node contains ALL SIX functions:

| Function | What It Holds | Gate Question |
|----------|---------------|---------------|
| **Heat** | Magnitude (never decreases) | Does it exist at all? |
| **Polarity** | Direction (+1 or -1) | Aligned or opposed? |
| **Existence** | State (actual/dormant/archived) | Is it present? |
| **Righteousness** | Alignment value (R=0 = aligned) | Does it fit the frame? |
| **Order** | Sequence position | Valid in frame hierarchy? |
| **Movement** | Direction of change, connections | Where does heat go next? |

## The Five Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| 12 | Twelve | Total directional freedom |
| 6 | Six | Motion function count |
| -1/12 | Negative one twelfth | Entropic bound |
| i | √-1 | Rotational capacity |
| φ | 1.618... | Golden ratio, identity-preserving scale |

## Project Structure

```
pi_release/
├── core/                         # The Engine (pure logic, no I/O)
│   ├── __init__.py               # Module exports
│   ├── node_constants.py         # Polarity (+/-) - direction validator, all constants
│   ├── nodes.py                  # Righteousness (R) - Node, Frame, Axis, Order
│   ├── manifold.py               # Order (Q) + Heat (Σ) - container + psychology
│   ├── clock_node.py             # Existence (δ) - Self IS the clock
│   ├── decision_node.py          # Movement (Lin) - 5 scalars → 1 vector
│   ├── driver_node.py            # IO - SensorReport, MotorAction, ActionPlan
│   ├── compression.py            # Position encoding utility
│   └── test_birth.py             # Birth & operation test suite
│
├── drivers/                      # All World Interaction
│   ├── __init__.py               # Driver exports
│   ├── environment.py            # ENTRY point - EnvironmentCore
│   ├── template_driver.py        # Template for new drivers
│   ├── gym_driver.py             # Gymnasium adapter
│   ├── maze_driver.py            # Maze environment driver
│   ├── blackjack_driver.py       # Blackjack driver
│   ├── conversation_driver.py    # Conversation driver
│   ├── gym_adapters/             # Gymnasium support
│   │   ├── agent.py, encoders.py, decoders.py, reward_shaping.py
│   └── tasks/                    # Runnable applications
│       ├── chat_client.py, voice_client.py, unified_gui.py
│       ├── gym_runner.py, bigmaze.py, blackjack.py
│       └── minecraft_example.py
│
├── decisions/                    # Choice history persists here
├── growth/                       # Manifold/psychology state
│   └── *.json                    # Serialized manifold data
│
├── pi/                           # Raspberry Pi hardware interface
│   ├── daemon.py, api.py, thermal.py
│
├── tests/
│   ├── test_all.py               # Comprehensive test suite
│   ├── test_architecture.py      # Architecture tests
│   ├── test_persistence.py       # Persistence tests
│   └── test_gym.py               # Gymnasium integration tests
│
└── main.py                       # Entry point, main loop
```

## The 6 Motion Functions = The 6 Core Files

```
5 SCALARS (validated inputs):
1. Heat (Σ)        → manifold.py       (magnitude, psychology)
2. Polarity (+/-)  → node_constants.py (direction)
3. Existence (δ)   → clock_node.py     (persistence, 1/φ³ threshold)
4. Righteousness   → nodes.py          (alignment, R→0)
5. Order (Q)       → manifold.py       (history, arithmetic)
                     ↓
1 VECTOR (output):
6. Movement (Lin)  → decision_node.py  (the decision)

THE 5/6 CONFIDENCE THRESHOLD:
- Above 5/6 (0.8333): EXPLOIT (5 scalars validated → use pattern)
- Below 5/6: EXPLORE (still gathering validation)
```

## Universal Environment Core

The Environment Core isolates PBAI from specific environments through a driver/port architecture:

```
PBAI <---> Environment Core <---> Driver <---> Port <---> External Environment
                                    │
                              DriverNode (learning)
```

### Two-Layer Driver System

**Driver** (World Side): Handles communication with external environments
- `perceive()` → Returns normalized `Perception`
- `act()` → Executes `Action`, returns `ActionResult`

**DriverNode** (Brain Side): Stores PBAI's learned knowledge
- Learned state patterns
- Motor patterns that work
- Plans (action sequences for goals)
- Persists to `drivers/{driver_id}/`

When you create a Driver with a manifold, it automatically creates a DriverNode for learning:

```python
from core import get_pbai_manifold
from drivers.environment import MockDriver

manifold = get_pbai_manifold()
driver = MockDriver(manifold=manifold)  # Creates DriverNode automatically

driver.perceive()  # Feeds to DriverNode for learning
driver.act(action) # Feeds result to DriverNode
driver.save_learning()  # Persists learned patterns
```

### Key Data Structures

**Perception** (What PBAI sees):
```python
Perception(
    entities=["tree", "rock"],      # Things that exist
    locations=["forest", "cave"],   # Places that exist
    properties={"weather": "clear"}, # State properties
    events=["bird sang"]            # Recent happenings
)
```

**Action** (What PBAI does):
```python
Action(
    action_type="move",    # observe, move, interact, wait, explore
    target="forest",       # What to act on
    parameters={}          # Action-specific params
)
```

### Creating a New Driver

1. Inherit from `Driver` class in `drivers/environment.py`
2. Implement `perceive()` and `act()` methods
3. Call `self.feed_perception()` and `self.feed_result()` for learning
4. Pass `manifold` to constructor for DriverNode integration

```python
from drivers.environment import Driver, Perception, Action, ActionResult

class MyDriver(Driver):
    DRIVER_ID = "my_driver"
    SUPPORTED_ACTIONS = ["observe", "act"]
    
    def __init__(self, port=None, config=None, manifold=None):
        super().__init__(port, config, manifold=manifold)
    
    def perceive(self) -> Perception:
        perception = Perception(...)
        self.feed_perception(perception)  # Learn!
        return perception
    
    def act(self, action: Action) -> ActionResult:
        result = ActionResult(...)
        self.feed_result(result, action)  # Learn!
        return result
```

## Hotswaps and Frame Archival

### Hotswap

Hotswaps bridge concepts across dimensions by creating recording nodes:

```python
# Both nodes must be R=0 (environment-righteous) and actual
result = perform_hotswap(node_x, node_y, context, manifold)

# Creates:
# - Z1: Routing node that knows about both X and Y
# - Z2: Context node recording why/when swap occurred
```

**Rules**:
- Both nodes must be environment-righteous (R=0)
- Both nodes must be actual (not dormant/archived)
- Swap is RECORDING, not moving - original nodes stay in place

### Frame Archival

When a frame is corrected:

```python
result = archive_frame(old_frame, new_frame, manifold)

# old_frame.existence -> "archived"
# Creates correction record node
```

### Potential Unit Resolution

When polarity collision creates potential units at R=0:

```python
# Node with potential_units > 0 and righteousness == 0
result = resolve_potential_unit(node, manifold)

# Creates a new frame node from the resolved tension
```

### Generative Inference

Create speculative combinations:

```python
result = infer_combination("red", "apple", manifold)
# Creates "red_apple" with R=0.5 (created-righteous)

# Later, when environment confirms:
confirm_inference(result.inferred_node, manifold)
# Now R=0 (environment-righteous)
```

## Quick Start

### Run Tests
```bash
python3 main.py --test
```

### Run Main Loop (Mock Haiku)
```bash
python3 main.py --loops 5 --verbose
```

### Run With Real Haiku API
```bash
python3 main.py --loops 5 --api-key YOUR_ANTHROPIC_API_KEY
```

### Load Existing Growth Map
```bash
python3 main.py --load growth/growth_map.json --loops 3
```

## The Main Loop

```
1. Perceive environment
2. Inject heat at Self, cascade through manifold
3. Ask Haiku: "What are my options?" (generates, doesn't choose)
4. PBAI evaluates each option (entropy loss + thermal efficiency)
5. PBAI CHOOSES the best one
6. Act on environment
7. Build nodes from experience
8. Save growth map (EVERY LOOP)
```

## Key Principles (Immutable)

### Principle 1: Atomic Identity
1 Property = 1 Node = 1 Concept. No exceptions.

### Principle 2: Every Node is Complete
Each node contains ALL SIX motion functions.

### Principle 3: Self Identity
- Self is the first node
- All paths trace to Self
- Self must persist (system failure if Self fails)
- Self is always hottest
- Below Self is void (d blocked)

### Principle 4: Self Containment
Connections stored IN the node, not as external objects.

### Principle 5: Frames are Nodes
Frames are not a separate data type—they are nodes.

### Principle 6: Position = Path from Self
A node's identity IS its path (string like "nnwwu").

## Compression System d(n)

Position strings are compressed for storage:

```
Full:       "nnnnnnnnnneeeeewwwwuuuuu"
Compressed: "n(10)e(5)w(4)u(5)"
```

## Choice Criteria

PBAI chooses options with:
- **Least entropy loss** (preserve order)
- **Most thermal efficiency** (useful heat / total heat)

```
Score = thermal_efficiency - entropy_loss
```

## Direction System

```
LATERAL PLANE (2D):
        n (+Y)
        |
   w ---+--- e
  (-X)  |   (+X)
        s (-Y)

VERTICAL (abstraction):
   u (+Z) = more abstract
   d (-Z) = more concrete (blocked at Self)
```

## Implementation Status

### Complete (Phases 1-8, 10-12)
- ✅ Core data structures (Node, SelfNode, Connection)
- ✅ Compression system with tests
- ✅ Manifold with persistence
- ✅ Bootstrap sequence
- ✅ Three-phase search
- ✅ Heat cascade
- ✅ Node creation with fallback
- ✅ Option generation (mock + Haiku)
- ✅ Option evaluation
- ✅ Main loop
- ✅ Growth map save/load

### Partial (Phase 9)
- ⚠️ Hotswaps (structure in place, needs testing)
- ⚠️ Frame archival (basic implementation)
- ⚠️ Generative inference (stub)

### Future Work
- Environment adapters (Minecraft, Chat, etc.)
- Visualization tools
- R() function tuning for semantic relationships
- Potential unit resolution at R=0

## Development Notes

Following the suggestions from the build plan:

1. **Started small**: Self → one node → one connection → save/load
2. **Tested compression early**: Rock solid with unit tests
3. **Stubbed Haiku first**: MockOptionGenerator works, real Haiku ready
4. **Built visualization**: `manifold.visualize()` dumps state
5. **Logs everything**: DEBUG level shows all thermal dynamics
6. **R() in one place**: `manifold.evaluate_righteousness()` easy to tune
7. **Self is sacred**: Assertions verify path-to-Self invariant
8. **Loop number in metadata**: Tracks evolution in growth map

## References

- Build Plan: PBAI_Thermal_Manifold_Build_Plan_v0_6.docx
- Foundational Theory: The_Motion_Calendar_Complete.tex

---

## Gymnasium Adapter (NEW)

PBAI now speaks the standard Gymnasium (OpenAI Gym) API, enabling it to learn from any compatible environment.

### Quick Start with Gym

```bash
# Install gymnasium
pip install gymnasium

# Run on Blackjack (compare to random baseline)
python -m gym_adapter.runner --env Blackjack-v1 --episodes 1000 --compare-random

# Run on FrozenLake (grid maze)
python -m gym_adapter.runner --env FrozenLake-v1 --episodes 500

# Run on CartPole (continuous observations)
python -m gym_adapter.runner --env CartPole-v1 --episodes 200

# Open unified GUI (play + chat)
python -m gym_adapter.unified_gui
```

### Talk to PBAI

```bash
# Terminal chat (no LLM - raw manifold responses)
python -m gym_adapter.pbai_voice --no-llm

# With LLM formatting
python -m gym_adapter.pbai_voice
```

Commands in chat:
- `/introspect` - PBAI describes its mental state
- `/connections` - Show semantic connection graph
- `/know <topic>` - What PBAI knows about a topic
- `/llm` - Toggle LLM formatting

### Semantic Connections (NEW)

PBAI now builds **semantic relationships** between concepts. When you teach it something, it creates directional connections:

| Direction | Meaning | Example |
|-----------|---------|---------|
| `n` | Positive assertion (is, has) | `self --n_name--> pbai` |
| `s` | Negative assertion (is not) | `user --s_not--> robot` |
| `u` | Generalization (type of) | `pbai --u_type--> ai` |
| `d` | Specification (instance) | `ai --d_instance--> pbai` |

**Strength through repetition**: Each mention strengthens the connection (`traversal_count`). More mentions = more confidence when answering.

```
You: Your name is PBAI
  → self --n_name--> pbai (×1)

You: You are PBAI  
  → self --n_name--> pbai (×2)  ← strengthened!

You: What is your name?
  → Follows connection, answers confidently: "My name is pbai."
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       ANY GYM ENVIRONMENT                        │
│  Blackjack | FrozenLake | CartPole | Atari | MuJoCo | Custom    │
└──────────────────────────────┬───────────────────────────────────┘
                               │ Gym API
                    ┌──────────┴──────────┐
                    │     PBAIAgent       │
                    │  ┌───────────────┐  │
                    │  │ Manifold      │  │  ← Persistent thermal memory
                    │  │ (heat, nodes) │  │
                    │  └───────────────┘  │
                    │  ┌───────────────┐  │
                    │  │ Encoders      │  │  ← Observation → Nodes
                    │  └───────────────┘  │
                    │  ┌───────────────┐  │
                    │  │ Decoders      │  │  ← Actions ← Decisions
                    │  └───────────────┘  │
                    └─────────────────────┘
```

### How It Works

1. **Observe**: Gym observation → encoder → state key → manifold node
2. **Decide**: Score actions thermally (heat = past success, curiosity = unexplored)
3. **Act**: Best action → Gym step
4. **Learn**: Reward → adjust heat on decision nodes

### Files

```
gym_adapter/
├── __init__.py      # Module exports
├── agent.py         # PBAIAgent - Gym-compatible agent
├── encoders.py      # Observation → manifold nodes
├── decoders.py      # Actions → semantic names
├── runner.py        # CLI for training runs
├── pbai_voice.py    # Talk to PBAI
├── unified_gui.py   # Combined play + chat GUI
└── test_gym.py      # Tests
```

---

*The universe is not objects in motion. It is motion itself.*
