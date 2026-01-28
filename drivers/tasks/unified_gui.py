"""
PBAI Unified Interface

A GUI that lets you:
1. Watch PBAI play Gymnasium environments
2. Talk to PBAI about what it's doing
3. See the manifold state in real-time

Usage:
    python -m drivers.tasks.unified_gui
    python -m drivers.tasks.unified_gui --env CartPole-v1
"""

import sys
import os
import threading
import queue
from time import time, sleep
from typing import Optional, Dict, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except ImportError:
    print("tkinter not available")
    sys.exit(1)

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed, Gym features disabled")

from core import Manifold, K, get_growth_path, get_pbai_manifold
from drivers.environment import EnvironmentCore
from drivers.gym_driver import GymDriver, create_gym_driver
from drivers.tasks.voice_client import PBAIVoice, PBAIMind

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class UnifiedGUI:
    """
    Unified interface for PBAI:
    - Left panel: Chat with PBAI
    - Center panel: Gym environment control
    - Right panel: Manifold state
    
    Uses the ONE PBAI manifold - all drivers share the same mind.
    Architecture: Uses EnvironmentCore for proper ENTRY/EXIT flow.
    """
    
    ENVS = [
        "Blackjack-v1",
        "FrozenLake-v1", 
        "FrozenLake8x8-v1",
        "CliffWalking-v1",
        "Taxi-v3",
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
    ]
    
    def __init__(self, root: tk.Tk, default_env: str = "Blackjack-v1"):
        self.root = root
        self.root.title("PBAI Unified Interface ")
        self.root.geometry("1400x800")
        self.root.configure(bg='#1a1a2e')
        
        # Get the ONE PBAI manifold
        self.growth_path = get_growth_path("growth_map.json")
        self.manifold = get_pbai_manifold(self.growth_path)
        
        # Clean up any invalid nodes from previous runs
        cleaned = self.manifold.cleanup_invalid_nodes()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} invalid nodes from previous session")
        
        logger.info(f"PBAI mind ready: {len(self.manifold.nodes)} nodes")
        
        # Architecture: EnvironmentCore is the ENTRY point
        self.env_core = EnvironmentCore(self.manifold)
        
        # PBAI components - ONE manifold, multiple driver interfaces
        self.driver: Optional[GymDriver] = None
        self.voice = PBAIVoice(manifold=self.manifold, use_llm=False)  # Start without LLM
        
        # Environment name (driver holds the actual env)
        self.env_name = default_env
        
        # State
        self.running = False
        self.episode = 0
        self.step = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        
        # Threading
        self.command_queue = queue.Queue()
        self.update_queue = queue.Queue()
        
        self._setup_ui()
        self._start_update_loop()
        
        # Save on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def save_growth(self):
        """Save to unified growth map."""
        try:
            self.manifold.save_growth_map(self.growth_path)
            logger.info(f"Saved {len(self.manifold.nodes)} nodes to growth map")
        except Exception as e:
            logger.warning(f"Failed to save growth map: {e}")
    
    def _on_close(self):
        """Handle window close - save and exit."""
        self.running = False
        self.save_growth()
        # Deactivate driver through EnvironmentCore
        if self.driver:
            self.env_core.deactivate_driver()
        self.root.destroy()
    
    def _setup_ui(self):
        """Build the interface."""
        # Main container
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === LEFT: Chat Panel ===
        chat_frame = tk.Frame(main, bg='#16213e', width=450)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        chat_frame.pack_propagate(False)
        
        tk.Label(chat_frame, text="💬 Talk to PBAI", font=('Arial', 14, 'bold'),
                 bg='#16213e', fg='#00ff88').pack(pady=10)
        
        # Chat history
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, font=('Consolas', 10), bg='#0f0f1a', fg='#e0e0e0',
            wrap=tk.WORD, height=35
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # Tag configs for colors
        self.chat_display.tag_config('user', foreground='#64b5f6')
        self.chat_display.tag_config('pbai', foreground='#81c784')
        self.chat_display.tag_config('system', foreground='#ffb74d')
        
        # Input
        input_frame = tk.Frame(chat_frame, bg='#16213e')
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.chat_input = tk.Entry(input_frame, font=('Arial', 11), bg='#2a2a4a', fg='white',
                                   insertbackground='white')
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_input.bind('<Return>', self._on_chat_submit)
        
        tk.Button(input_frame, text="Send", command=self._on_chat_submit,
                  bg='#4CAF50', fg='white').pack(side=tk.RIGHT, padx=(5, 0))
        
        # === CENTER: Gym Control ===
        gym_frame = tk.Frame(main, bg='#16213e', width=450)
        gym_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        gym_frame.pack_propagate(False)
        
        tk.Label(gym_frame, text="🎮 Gymnasium Control", font=('Arial', 14, 'bold'),
                 bg='#16213e', fg='#00ff88').pack(pady=10)
        
        # Environment selector
        env_row = tk.Frame(gym_frame, bg='#16213e')
        env_row.pack(fill=tk.X, padx=10)
        
        tk.Label(env_row, text="Environment:", bg='#16213e', fg='white').pack(side=tk.LEFT)
        
        self.env_var = tk.StringVar(value=self.env_name)
        env_combo = ttk.Combobox(env_row, textvariable=self.env_var, values=self.ENVS, width=20)
        env_combo.pack(side=tk.LEFT, padx=10)
        env_combo.bind('<<ComboboxSelected>>', self._on_env_change)
        
        # Control buttons
        btn_row = tk.Frame(gym_frame, bg='#16213e')
        btn_row.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = tk.Button(btn_row, text="▶ Start", command=self._start_training,
                                   bg='#4CAF50', fg='white', width=10)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(btn_row, text="⏹ Stop", command=self._stop_training,
                                  bg='#f44336', fg='white', width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.step_btn = tk.Button(btn_row, text="⏭ Step", command=self._single_step,
                                  bg='#2196F3', fg='white', width=10)
        self.step_btn.pack(side=tk.LEFT, padx=2)
        
        # Speed control
        speed_row = tk.Frame(gym_frame, bg='#16213e')
        speed_row.pack(fill=tk.X, padx=10)
        
        tk.Label(speed_row, text="Speed:", bg='#16213e', fg='white').pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=100)
        tk.Scale(speed_row, from_=10, to=1000, orient=tk.HORIZONTAL,
                 variable=self.speed_var, bg='#16213e', fg='white',
                 highlightthickness=0).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Stats display
        stats_frame = tk.LabelFrame(gym_frame, text="Statistics", bg='#16213e', fg='white')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_labels = {}
        for i, stat in enumerate(['Episode', 'Step', 'Episode Reward', 'Total Reward', 'Exploration']):
            row = tk.Frame(stats_frame, bg='#16213e')
            row.pack(fill=tk.X)
            tk.Label(row, text=f"{stat}:", bg='#16213e', fg='#aaa', width=15, anchor='w').pack(side=tk.LEFT)
            lbl = tk.Label(row, text="0", bg='#16213e', fg='#00ff88', width=15, anchor='w')
            lbl.pack(side=tk.LEFT)
            self.stats_labels[stat] = lbl
        
        # Episode log
        tk.Label(gym_frame, text="Episode Log:", bg='#16213e', fg='white').pack(anchor='w', padx=10)
        self.episode_log = scrolledtext.ScrolledText(
            gym_frame, font=('Consolas', 9), bg='#0f0f1a', fg='#e0e0e0',
            wrap=tk.WORD, height=15
        )
        self.episode_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # === RIGHT: Manifold State ===
        manifold_frame = tk.Frame(main, bg='#16213e', width=400)
        manifold_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        manifold_frame.pack_propagate(False)
        
        tk.Label(manifold_frame, text="🧠 Manifold State", font=('Arial', 14, 'bold'),
                 bg='#16213e', fg='#00ff88').pack(pady=10)
        
        # Mind state
        mind_frame = tk.LabelFrame(manifold_frame, text="PBAI Mind", bg='#16213e', fg='white')
        mind_frame.pack(fill=tk.X, padx=10)
        
        self.mind_labels = {}
        for stat in ['Mood', 'Confidence', 'Focus', 'Nodes']:
            row = tk.Frame(mind_frame, bg='#16213e')
            row.pack(fill=tk.X)
            tk.Label(row, text=f"{stat}:", bg='#16213e', fg='#aaa', width=12, anchor='w').pack(side=tk.LEFT)
            lbl = tk.Label(row, text="-", bg='#16213e', fg='#00ff88', anchor='w')
            lbl.pack(side=tk.LEFT, fill=tk.X)
            self.mind_labels[stat] = lbl
        
        # Hot nodes
        tk.Label(manifold_frame, text="Hot Nodes:", bg='#16213e', fg='white').pack(anchor='w', padx=10, pady=(10, 0))
        self.nodes_display = scrolledtext.ScrolledText(
            manifold_frame, font=('Consolas', 9), bg='#0f0f1a', fg='#00ff88',
            wrap=tk.NONE, height=25
        )
        self.nodes_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add welcome message
        self._add_chat("system", "Welcome! I'm PBAI. You can talk to me or watch me learn.")
        self._add_chat("system", "Commands: /introspect, /know <topic>, /help")
    
    def _add_chat(self, sender: str, text: str):
        """Add message to chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        prefix = {"user": "You", "pbai": "PBAI", "system": "SYS"}[sender]
        tag = sender
        
        self.chat_display.insert(tk.END, f"{prefix}: ", tag)
        self.chat_display.insert(tk.END, f"{text}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _on_chat_submit(self, event=None):
        """Handle chat input."""
        text = self.chat_input.get().strip()
        if not text:
            return
        
        self.chat_input.delete(0, tk.END)
        self._add_chat("user", text)
        
        # Handle commands
        if text.startswith("/"):
            self._handle_command(text[1:])
        else:
            # Talk to PBAI
            response = self.voice.process_input(text)
            self._add_chat("pbai", response)
    
    def _handle_command(self, cmd: str):
        """Handle slash commands."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if command == "introspect":
            result = self.voice.introspect()
            self._add_chat("system", result)
        
        elif command == "know":
            if arg:
                result = self.voice.share_knowledge(arg)
                self._add_chat("system", result)
            else:
                self._add_chat("system", "Usage: /know <topic>")
        
        elif command == "help":
            self._add_chat("system", """Commands:
/introspect - Show PBAI's current mental state
/know <topic> - Show what PBAI knows about topic
/stats - Show agent statistics
/help - Show this help""")
        
        elif command == "stats":
            if self.driver:
                info = self.driver.get_info()
                lines = [f"{k}: {v}" for k, v in info.items() if not isinstance(v, list)]
                self._add_chat("system", "\n".join(lines))
            else:
                self._add_chat("system", "No driver active. Start training first.")
        
        else:
            self._add_chat("system", f"Unknown command: {command}")
    
    def _on_env_change(self, event=None):
        """Handle environment change."""
        new_env = self.env_var.get()
        if new_env != self.env_name:
            self.env_name = new_env
            if self.running:
                self._stop_training()
            self._add_chat("system", f"Switched to {new_env}")
    
    def _init_env(self):
        """Initialize gymnasium environment ."""
        if not GYM_AVAILABLE:
            self._add_chat("system", "Error: gymnasium not installed")
            return False
        
        try:
            # Deactivate old driver if exists
            if self.driver:
                self.env_core.deactivate_driver()
            
            # Create new GymDriver
            self.driver = create_gym_driver(self.env_name, manifold=self.manifold)
            
            # Register and activate through EnvironmentCore
            self.env_core.register_driver(self.driver)
            if not self.env_core.activate_driver(self.driver.DRIVER_ID):
                self._add_chat("system", f"Failed to activate {self.env_name}")
                return False
            
            self._add_chat("system", f"Loaded {self.env_name} (unified architecture)")
            self._log_episode(f"Initialized {self.env_name}")
            self._log_episode(f"Actions: {self.driver.get_available_actions()}")
            return True
            
        except Exception as e:
            self._add_chat("system", f"Error loading env: {e}")
            logger.exception("Failed to init env")
            return False
    
    def _start_training(self):
        """Start training loop."""
        if not self._init_env():
            return
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.DISABLED)
        
        # Start training thread
        thread = threading.Thread(target=self._training_loop, daemon=True)
        thread.start()
    
    def _stop_training(self):
        """Stop training."""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.step_btn.config(state=tk.NORMAL)
    
    def _single_step(self):
        """Run a single step ."""
        if not self.driver:
            if not self._init_env():
                return
            self.driver.reset()
            self.episode_reward = 0.0
        
        # Use EnvironmentCore.step() for full cycle
        action, result, heat_changes = self.env_core.step()
        
        reward = result.changes.get("reward", 0)
        done = result.changes.get("done", False)
        
        self.step += 1
        self.episode_reward += reward
        self.total_reward += reward
        
        action_name = action.target or action.action_type
        self._log_episode(f"Step {self.step}: {action_name} -> reward={reward:.2f}")
        
        if done:
            self.episode += 1
            self._log_episode(f"Episode {self.episode} complete: reward={self.episode_reward:.2f}")
            self._log_episode(f"  Baseline: {self.driver._episode_baseline:.2f}")
            self.driver.reset()
            self.episode_reward = 0.0
            # Save every 10 episodes
            if self.episode % 10 == 0:
                self.save_growth()
        
        self._update_displays()
    
    def _training_loop(self):
        """Background training loop ."""
        self.driver.reset()
        self.episode_reward = 0.0
        steps_this_episode = 0
        
        while self.running:
            # Use EnvironmentCore.step() for full cycle
            action, result, heat_changes = self.env_core.step()
            
            reward = result.changes.get("reward", 0)
            done = result.changes.get("done", False)
            
            # Debug: Log all steps for troubleshooting
            if "Blackjack" in self.env_name or "FrozenLake" in self.env_name:
                if reward != 0 or done:
                    logger.info(f"Step {steps_this_episode}: action={action.target}, reward={reward}, done={done}")
            
            self.step += 1
            steps_this_episode += 1
            self.episode_reward += reward
            self.total_reward += reward
            
            # Prevent runaway episodes
            if steps_this_episode > 50:  # Blackjack should never exceed ~10
                logger.error(f"Episode exceeded 50 steps ({steps_this_episode}) - forcing reset. Last action: {action.target}, result: {result.outcome}")
                done = True
            
            if done:
                self.episode += 1
                self.update_queue.put(('episode', self.episode, self.episode_reward))
                self.driver.reset()
                self.episode_reward = 0.0
                steps_this_episode = 0
                # Save every 10 episodes
                if self.episode % 10 == 0:
                    self.save_growth()
            
            # Throttle based on speed
            delay = 1.0 / self.speed_var.get()
            sleep(delay)
    
    def _log_episode(self, text: str):
        """Add to episode log."""
        self.episode_log.insert(tk.END, f"{text}\n")
        self.episode_log.see(tk.END)
    
    def _update_displays(self):
        """Update all displays."""
        # Stats
        self.stats_labels['Episode'].config(text=str(self.episode))
        self.stats_labels['Step'].config(text=str(self.step))
        self.stats_labels['Episode Reward'].config(text=f"{self.episode_reward:.2f}")
        self.stats_labels['Total Reward'].config(text=f"{self.total_reward:.2f}")
        
        # Get exploration rate from manifold
        if self.driver:
            exp_rate = self.manifold.get_exploration_rate()
            self.stats_labels['Exploration'].config(text=f"{exp_rate:.1%}")
        
        # Mind state
        mind = self.voice.read_mind()
        self.mind_labels['Mood'].config(text=mind.mood)
        self.mind_labels['Confidence'].config(text=f"{mind.confidence:.0%}")
        self.mind_labels['Focus'].config(text=mind.focus[0][0] if mind.focus else "-")
        self.mind_labels['Nodes'].config(text=str(len(self.manifold.nodes)))
        
        # Hot nodes
        self.nodes_display.delete(1.0, tk.END)
        nodes = sorted(self.manifold.nodes.values(), key=lambda n: n.heat, reverse=True)
        for n in nodes[:30]:
            if not n.concept.startswith('bootstrap'):
                bar = "█" * min(10, int(n.heat))
                self.nodes_display.insert(tk.END, f"{n.concept[:25]:25} {n.heat:6.2f} {bar}\n")
    
    def _start_update_loop(self):
        """Start periodic UI updates."""
        def update():
            # Process queued updates
            try:
                while True:
                    msg = self.update_queue.get_nowait()
                    if msg[0] == 'episode':
                        _, ep, reward = msg
                        self._log_episode(f"Episode {ep}: reward={reward:.2f}")
            except queue.Empty:
                pass
            
            # Periodic display update
            if self.running or self.episode > 0:
                self._update_displays()
            
            self.root.after(100, update)
        
        update()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Blackjack-v1", help="Default environment")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = UnifiedGUI(root, default_env=args.env)
    root.mainloop()


if __name__ == "__main__":
    main()
