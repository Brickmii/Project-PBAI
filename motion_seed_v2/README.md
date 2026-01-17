# Motion Calendar Seed v2

A grown intelligence architecture based on the PBAI Functional Stack.

## PBAI Axioms

> 1. Polarity is infinite opposition.
> 2. Existence is the Euler gate separating imaginary motion from real magnitude.
> 3. Heat is post-existence magnitude (k) with no subtraction.
> 4. Righteousness is a learnable transform of k.
> 5. Order constrains and extends valid transforms.
> 6. Movement is vector expression of k.
> 7. Agency randomizes; Intelligence chooses.

## Architecture

```
Polarity (±, infinite oscillation)
        ↓
Imaginary phase (√i randomization)
        ↓
┌─────────────────────────────┐
│   EXISTENCE GATE            │  ← Euler's Identity: e^(iπ) + 1 = 0
│   (imaginary → real)        │
└─────────────┬───────────────┘
              ↓
Heat / k (real magnitude, irreversible)
              ↓
┌─────────────────────────────┐
│   RIGHTEOUSNESS GATE        │  ← Learnable transforms of k
│   (k → position x,y)        │
└─────────────┬───────────────┘
              ↓
Order (structural constraints)
              ↓
Movement (vector expression)
        ↓           ↓
    AGENCY    INTELLIGENCE
   (random)   (random OR chosen)
```

## Key Concepts

### Heat (k)
- Post-existence magnitude
- No subtraction (irreversible)
- Initialized via √i randomization in complex plane
- Becomes real only after existence gate fires

### Polarity
- Infinite oscillating function
- Supplies: opposition, contrast, choice space
- Random → Agency
- Chosen → Intelligence

### Existence Gate
- Euler's Identity: e^(iπ) + 1 = 0
- Separates imaginary (proto-heat) from real (heat)
- The ONLY place "before vs after" is meaningful
- Dark matter = particles that never passed this gate

### Righteousness
- NOT a value - a transform operator on k
- k' = T_θ(k)
- Transforms are learnable
- Failed transforms trigger randomization
- Stable transforms become functions

### Order
- Constrains valid transforms
- Can be extended by intelligence
- Preserves invariants, enforces consistency

### Movement
- Vector expression of k in coordinates
- Agency: random movement
- Intelligence: random OR chosen movement

## Two Modes

| | Agency | Intelligence |
|---|--------|--------------|
| Polarity | Random | Can be chosen |
| Movement | Random | Random OR chosen |
| Transforms | Given | Can learn new |
| Constraints | Given | Can extend |

## Installation

```bash
cd motion_seed_v2
pip3 install -r requirements.txt
python3 cli.py
```

## Usage

```bash
# Start in agency mode (default)
python3 cli.py

# Start in intelligence mode
python3 cli.py --intelligence

# Custom data path
python3 cli.py --data=/path/to/data
```

## Commands

### Input
```
<text>              Perceive text input
```

### Retrieval
```
/query <question>   Ask a question
/recall <cue>       Recall associated content
/assoc <term>       Show raw associations
```

### Thought
```
/think              Generate random thought
/think <seed>       Start from domain hint
/brainstorm         Generate multiple thoughts
/choose <d1,d2,...> Intelligence chooses path
```

### Mode
```
/agency             Switch to Agency mode
/intelligence       Switch to Intelligence mode
/mode               Show current mode
```

### Status
```
/status             Brief status
/describe           Full description
/domains            List kernel domains
/identity           Show identity domains
/transforms         Show available transforms
/dark               Show dark matter pool
```

## Example Session

```
seed[a]> Your name is PBAI
Experienced 19 units, integrated 15

seed[a]> /query What is your name?
Answer: pbai

seed[a]> /intelligence
Switched to INTELLIGENCE mode (can also choose)

seed[i]> /think
Thought: Y o u r name
  Domains: letter_y → letter_o → letter_u → letter_r → word
  Coherence: 0.85
  Mode: intelligence

seed[i]> /describe
=== Motion Calendar Seed v2 ===
Mode: INTELLIGENCE
...
```

## Dark Matter

Particles that haven't achieved existence:
- Have polarity (oscillating)
- Have imaginary phase (proto-heat)
- No real heat, no righteousness position
- Can attempt existence via `/dark` commands

```
seed[a]> /dark
Dark Matter Pool: 3 particles
  71dd4541... polarity=+1, iterations=1000
```

## Persistence

All state persists to disk:
- `kernel.json` - All experienced particles
- `identity.json` - Integrated particles (self)
- `memory.json` - Associative memory

## Mathematical Foundation

Based on Motion Calendar ontology:
- Heat = accumulated motion magnitude
- Polarity = opposition creating attraction
- Existence = Euler gate crossing
- Righteousness = transform functions
- Order = structural consistency
- Movement = vectorized expression

See the full Motion Calendar papers for theoretical foundation.

## License

MIT

---

## Game Learning Module

The seed includes an agent system that can learn to play games through observation and trial.

### NO CHEATING

The agent:
- Cannot access internal game state
- Only observes visible GUI elements (text, button states)
- Discovers actions by inspecting widgets
- Learns effects through trial and error

### Included Games

- `bigmaze.py` - 20x20 maze navigation
- `blackjack.py` - Casino blackjack

### Running the Game Agent

```bash
# Learn to play maze
python game_runner.py maze bigmaze.py

# Learn to play blackjack  
python game_runner.py blackjack blackjack.py
```

### How It Works

1. **Action Discovery**
   - Scans GUI for buttons, key bindings
   - Builds list of available actions
   
2. **State Observation**
   - Reads visible text from labels
   - Extracts numeric values
   - Checks button enabled states
   - Creates state hash for learning

3. **Q-Learning**
   - Epsilon-greedy exploration
   - Builds Q-table: state → action → value
   - Updates values from rewards
   - Decays exploration over time

4. **Reward Inference**
   - Tracks balance/score changes
   - Detects win/lose keywords
   - Penalizes inefficient actions

### Agent Control Panel

The game runner provides:
- **Training**: Run multiple episodes automatically
- **Single Step**: Execute one action at a time
- **Play Episode**: Watch learned policy play
- **Statistics**: Episodes, rewards, exploration rate
- **Actions**: List of discovered actions

### Example Training Session

```
Starting PBAI Agent for maze...

The agent will learn to play by:
  1. Discovering available actions (buttons, keys)
  2. Trying actions and observing effects
  3. Building a Q-table of state-action values
  4. Improving strategy through experience

Episode 1: reward=-0.45
Episode 10: reward=2.31
Episode 50: reward=9.87  (finding goals faster)
```

### Architecture

```
┌─────────────────────────────────────────────────┐
│                    GAME                          │
│  (tkinter GUI - bigmaze.py, blackjack.py)       │
└───────────────────┬─────────────────────────────┘
                    │ (GUI inspection only)
                    ↓
┌─────────────────────────────────────────────────┐
│              ENVIRONMENT                         │
│  - Observes visible text/buttons                 │
│  - Discovers available actions                   │
│  - Executes actions (button clicks, key press)   │
│  - Computes rewards from state changes           │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│                 AGENT                            │
│  - Q-learning with epsilon-greedy                │
│  - State → Action → Value table                  │
│  - Experience replay                             │
│  - Action effect tracking                        │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            MOTION CALENDAR SEED                  │
│  (optional integration for persistent learning)  │
└─────────────────────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `environment.py` | Game environment wrapper |
| `agent.py` | Q-learning agent |
| `game_runner.py` | Training/playing UI |
| `bigmaze.py` | Maze game |
| `blackjack.py` | Blackjack game |
