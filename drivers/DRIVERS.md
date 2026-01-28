# PBAI Drivers - Architecture Reference

## Directory Structure

```
drivers/
├── __init__.py           # Package exports
├── environment.py        # EnvironmentCore - THE routing hub
├── gym_driver.py         # Unified gym driver (uses GameHandlers)
├── blackjack_driver.py   # Your custom blackjack with card counting
├── maze_driver.py        # Standalone bigmaze game driver
├── conversation_driver.py # Chat/voice driver
├── template_driver.py    # Template for new drivers
├── run.py                # Generic driver runner utility
│
├── gym_adapters/         # Gym I/O translation
│   ├── encoders.py       # State → semantic keys
│   └── decoders.py       # Action names ↔ gym ints
│
└── tasks/                # Runnable applications
    ├── unified_gui.py    # GUI for all gym envs ← USE THIS
    ├── gym_runner.py     # CLI runner for gym envs
    ├── blackjack.py      # Blackjack GUI (uses blackjack_driver.py)
    ├── bigmaze.py        # Maze GUI (uses maze_driver.py)
    ├── chat_client.py    # Chat interface
    └── voice_client.py   # Voice interface
```

## What To Use

### For Gym Environments (FrozenLake, Blackjack-v1, CartPole, etc.)
```bash
python -m drivers.tasks.unified_gui
```
Uses: `gym_driver.py` → `GameHandlers` → `gym_adapters/encoders.py` + `decoders.py`

### For Custom Blackjack (with card counting)
```bash
python -m drivers.tasks.blackjack
```
Uses: `blackjack_driver.py` (standalone, doesn't use gym)

### For Maze Game
```bash
python -m drivers.tasks.bigmaze
```
Uses: `maze_driver.py` (standalone, doesn't use gym)

## gym_driver.py GameHandlers

The unified gym driver routes to specialized handlers:

| Environment | Handler | State Key Format |
|------------|---------|-----------------|
| FrozenLake, CliffWalking, Taxi | `GridGameHandler` | `{env}_r{row}c{col}` |
| Blackjack-v1 | `BlackjackGameHandler` | `bj_{h/s}{sum}v{dealer}` |
| CartPole, MountainCar, Acrobot | `GenericGameHandler` | binned features |

## Data Flow

```
User runs unified_gui.py
         │
         ▼
   EnvironmentCore (environment.py)
         │
         ├── perceive() → GymDriver.perceive()
         │                    │
         │                    ▼
         │               GameHandler.get_state_key()
         │                    │
         │                    ▼
         │               Encoder (encoders.py)
         │
         ├── decide() → DecisionNode → Action
         │
         ├── act() → GymDriver.act()
         │               │
         │               ▼
         │           Decoder (decoders.py) → gym.step()
         │               │
         │               ▼
         │           GameHandler.record_outcome()
         │
         └── feedback() → DecisionNode.complete_decision()
```

## Creating a New Driver

1. Copy `template_driver.py`
2. Implement `perceive()`, `act()`, `supports_action()`
3. Register with EnvironmentCore
4. Create task file in `tasks/` to run it
