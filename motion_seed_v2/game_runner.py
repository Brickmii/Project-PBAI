#!/usr/bin/env python3
"""
Motion Calendar Seed - Game Runner
Runs games with agent learning and playing.

Usage:
    python game_runner.py maze      # Run maze with agent
    python game_runner.py blackjack # Run blackjack with agent
    python game_runner.py maze --train 100  # Train for 100 episodes
"""

import sys
import os
import time
import tkinter as tk
from tkinter import ttk
import threading
import importlib.util

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import create_environment
from agent import create_agent


class GameRunner:
    """
    Runs a game with an agent that learns to play it.
    
    Provides:
    - Game window
    - Agent control panel
    - Training/playing modes
    - Statistics display
    """
    
    def __init__(self, game_path: str, game_type: str):
        self.game_path = game_path
        self.game_type = game_type
        
        # Load game module
        self.game_module = self._load_game_module(game_path)
        
        # Create main window for agent control
        self.control_root = tk.Tk()
        self.control_root.title(f"PBAI Agent - {game_type.title()}")
        self.control_root.geometry("400x600")
        self.control_root.configure(bg="#1a1a2e")
        
        # Create game window
        self.game_root = tk.Toplevel(self.control_root)
        self.game_root.title(f"{game_type.title()} - Game Window")
        
        # Initialize game
        self.game_instance = self._create_game_instance()
        
        # Create environment and agent
        self.env = create_environment(game_type, self.game_root, self.game_instance)
        self.agent = create_agent(game_type, self.env)
        
        # Control state
        self.is_training = False
        self.is_playing = False
        self.training_thread = None
        
        # Setup control panel UI
        self._setup_control_panel()
        
        # Discover actions
        self.agent.discover_actions()
        self._update_actions_display()
    
    def _load_game_module(self, path: str):
        """Load game module from file."""
        spec = importlib.util.spec_from_file_location("game_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def _create_game_instance(self):
        """Create instance of the game."""
        if self.game_type.lower() == "maze":
            return self.game_module.PlayableMazeApp(self.game_root)
        elif self.game_type.lower() == "blackjack":
            return self.game_module.BlackjackGUI(self.game_root)
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")
    
    def _setup_control_panel(self):
        """Setup the agent control panel UI."""
        # Title
        title = tk.Label(
            self.control_root,
            text=f"PBAI Agent Control",
            font=("Arial", 16, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        title.pack(pady=10)
        
        # Status frame
        status_frame = tk.LabelFrame(
            self.control_root,
            text="Status",
            font=("Arial", 10, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=("Arial", 10),
            bg="#1a1a2e",
            fg="#0f3460"
        )
        self.status_label.pack(pady=5)
        
        # Stats frame
        stats_frame = tk.LabelFrame(
            self.control_root,
            text="Statistics",
            font=("Arial", 10, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(
            stats_frame,
            height=8,
            width=40,
            font=("Courier", 9),
            bg="#16213e",
            fg="#e94560"
        )
        self.stats_text.pack(pady=5, padx=5)
        
        # Actions frame
        actions_frame = tk.LabelFrame(
            self.control_root,
            text="Discovered Actions",
            font=("Arial", 10, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.actions_text = tk.Text(
            actions_frame,
            height=6,
            width=40,
            font=("Courier", 9),
            bg="#16213e",
            fg="#0f3460"
        )
        self.actions_text.pack(pady=5, padx=5)
        
        # Training controls
        train_frame = tk.LabelFrame(
            self.control_root,
            text="Training",
            font=("Arial", 10, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        train_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Episodes input
        ep_frame = tk.Frame(train_frame, bg="#1a1a2e")
        ep_frame.pack(pady=5)
        
        tk.Label(
            ep_frame,
            text="Episodes:",
            font=("Arial", 10),
            bg="#1a1a2e",
            fg="#e94560"
        ).pack(side=tk.LEFT, padx=5)
        
        self.episodes_entry = tk.Entry(ep_frame, width=10)
        self.episodes_entry.insert(0, "50")
        self.episodes_entry.pack(side=tk.LEFT, padx=5)
        
        # Training buttons
        btn_frame = tk.Frame(train_frame, bg="#1a1a2e")
        btn_frame.pack(pady=5)
        
        self.train_btn = tk.Button(
            btn_frame,
            text="Start Training",
            font=("Arial", 10, "bold"),
            bg="#4ade80",
            fg="black",
            command=self._toggle_training
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.step_btn = tk.Button(
            btn_frame,
            text="Single Step",
            font=("Arial", 10),
            bg="#60a5fa",
            fg="black",
            command=self._single_step
        )
        self.step_btn.pack(side=tk.LEFT, padx=5)
        
        # Playing controls
        play_frame = tk.LabelFrame(
            self.control_root,
            text="Playing",
            font=("Arial", 10, "bold"),
            bg="#1a1a2e",
            fg="#e94560"
        )
        play_frame.pack(fill=tk.X, padx=10, pady=5)
        
        play_btn_frame = tk.Frame(play_frame, bg="#1a1a2e")
        play_btn_frame.pack(pady=5)
        
        self.play_btn = tk.Button(
            play_btn_frame,
            text="Play Episode",
            font=("Arial", 10, "bold"),
            bg="#f59e0b",
            fg="black",
            command=self._play_episode
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(
            play_btn_frame,
            text="Reset Game",
            font=("Arial", 10),
            bg="#ef4444",
            fg="white",
            command=self._reset_game
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        speed_frame = tk.Frame(play_frame, bg="#1a1a2e")
        speed_frame.pack(pady=5)
        
        tk.Label(
            speed_frame,
            text="Play Speed:",
            font=("Arial", 10),
            bg="#1a1a2e",
            fg="#e94560"
        ).pack(side=tk.LEFT, padx=5)
        
        self.speed_scale = tk.Scale(
            speed_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg="#1a1a2e",
            fg="#e94560",
            highlightthickness=0
        )
        self.speed_scale.set(0.2)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Epsilon display
        self.epsilon_label = tk.Label(
            self.control_root,
            text=f"Exploration Rate (ε): {self.agent.epsilon:.3f}",
            font=("Arial", 10),
            bg="#1a1a2e",
            fg="#e94560"
        )
        self.epsilon_label.pack(pady=5)
        
        # Update stats periodically
        self._update_stats()
    
    def _update_stats(self):
        """Update statistics display."""
        stats = self.agent.get_stats()
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Episodes: {stats['total_episodes']}\n")
        self.stats_text.insert(tk.END, f"Steps: {stats['total_steps']}\n")
        self.stats_text.insert(tk.END, f"Total Reward: {stats['total_reward']:.2f}\n")
        self.stats_text.insert(tk.END, f"Avg Reward (last 100): {stats['average_reward']:.3f}\n")
        self.stats_text.insert(tk.END, f"States Visited: {stats['states_visited']}\n")
        self.stats_text.insert(tk.END, f"Experiences: {stats['experiences_stored']}\n")
        self.stats_text.insert(tk.END, f"Epsilon: {stats['epsilon']:.3f}\n")
        
        self.epsilon_label.config(text=f"Exploration Rate (ε): {stats['epsilon']:.3f}")
        
        # Schedule next update
        self.control_root.after(500, self._update_stats)
    
    def _update_actions_display(self):
        """Update discovered actions display."""
        self.actions_text.delete(1.0, tk.END)
        
        for name, action in self.agent.known_actions.items():
            enabled = "✓" if action.enabled else "○"
            self.actions_text.insert(tk.END, f"{enabled} {name} ({action.action_type})\n")
    
    def _toggle_training(self):
        """Toggle training on/off."""
        if self.is_training:
            self.is_training = False
            self.train_btn.config(text="Start Training", bg="#4ade80")
            self.status_label.config(text="Training stopped")
        else:
            self.is_training = True
            self.train_btn.config(text="Stop Training", bg="#ef4444")
            self.status_label.config(text="Training...")
            
            # Start training in thread
            try:
                num_episodes = int(self.episodes_entry.get())
            except:
                num_episodes = 50
            
            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(num_episodes,)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def _training_loop(self, num_episodes: int):
        """Run training episodes."""
        for episode in range(num_episodes):
            if not self.is_training:
                break
            
            reward = self.agent.run_episode(max_steps=500)
            
            self.control_root.after(0, lambda e=episode, r=reward: 
                self.status_label.config(text=f"Episode {e+1}: reward={r:.2f}"))
            
            # Small delay to allow GUI updates
            time.sleep(0.01)
        
        self.is_training = False
        self.control_root.after(0, lambda:
            self.train_btn.config(text="Start Training", bg="#4ade80"))
        self.control_root.after(0, lambda:
            self.status_label.config(text="Training complete"))
    
    def _single_step(self):
        """Execute a single agent step."""
        result = self.agent.step()
        
        if result:
            self.status_label.config(
                text=f"Action: {result.action.name}, Reward: {result.reward:.3f}"
            )
            self._update_actions_display()
    
    def _play_episode(self):
        """Play one episode with learned policy."""
        self.status_label.config(text="Playing episode...")
        self.play_btn.config(state=tk.DISABLED)
        
        def play():
            delay = self.speed_scale.get()
            reward = self.agent.play_episode(max_steps=500, delay=delay)
            
            self.control_root.after(0, lambda:
                self.status_label.config(text=f"Episode finished: reward={reward:.2f}"))
            self.control_root.after(0, lambda:
                self.play_btn.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=play)
        thread.daemon = True
        thread.start()
    
    def _reset_game(self):
        """Reset the game."""
        self.env.reset()
        self.status_label.config(text="Game reset")
        self._update_actions_display()
    
    def run(self):
        """Run the game runner."""
        self.control_root.mainloop()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python game_runner.py <game_type> [game_path]")
        print("  game_type: 'maze' or 'blackjack'")
        print("  game_path: path to game file (optional)")
        print()
        print("Examples:")
        print("  python game_runner.py maze bigmaze.py")
        print("  python game_runner.py blackjack blackjack.py")
        return
    
    game_type = sys.argv[1].lower()
    
    # Determine game path
    if len(sys.argv) >= 3:
        game_path = sys.argv[2]
    else:
        # Default paths
        if game_type == "maze":
            game_path = "bigmaze.py"
        elif game_type == "blackjack":
            game_path = "blackjack.py"
        else:
            print(f"Unknown game type: {game_type}")
            return
    
    if not os.path.exists(game_path):
        print(f"Game file not found: {game_path}")
        return
    
    print(f"Starting PBAI Agent for {game_type}...")
    print(f"Game file: {game_path}")
    print()
    print("The agent will learn to play by:")
    print("  1. Discovering available actions (buttons, keys)")
    print("  2. Trying actions and observing effects")
    print("  3. Building a Q-table of state-action values")
    print("  4. Improving strategy through experience")
    print()
    
    runner = GameRunner(game_path, game_type)
    runner.run()


if __name__ == "__main__":
    main()
