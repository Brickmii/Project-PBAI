# ---- Add below your existing code in main.py ----

import tkinter as tk
from environments.bigmaze import PlayableMazeApp
from pbai.maze_integration import MazeWithPBAI


class TkMazeEnvAdapter:
    """
    Adapts bigmaze.PlayableMazeApp to the API expected by MazeWithPBAI.
    """

    def __init__(self, app: PlayableMazeApp):
        self.app = app

    @property
    def player_pos(self):
        return self.app.player_pos

    def reset(self):
        # New maze + reset player
        self.app.new_game()

    def at_goal(self) -> bool:
        return self.app.game_won

    def get_valid_moves(self):
        r, c = self.app.player_pos
        size = self.app.maze_size
        grid = self.app.maze.grid

        candidates = {
            "up": (r - 1, c),
            "down": (r + 1, c),
            "left": (r, c - 1),
            "right": (r, c + 1),
        }

        valid = []
        for action, (nr, nc) in candidates.items():
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                valid.append(action)
        return valid

    def move(self, action: str) -> bool:
        r, c = self.app.player_pos
        if action == "up":
            nr, nc = r - 1, c
        elif action == "down":
            nr, nc = r + 1, c
        elif action == "left":
            nr, nc = r, c - 1
        elif action == "right":
            nr, nc = r, c + 1
        else:
            return False

        size = self.app.maze_size
        if 0 <= nr < size and 0 <= nc < size and self.app.maze.grid[nr][nc] == 0:
            self.app.player_pos = (nr, nc)
            self.app.draw_maze()

            # win check (same logic as bigmaze.on_key_press)
            if self.app.player_pos == self.app.maze.goal1:
                self.app.game_won = True
                self.app.status_label.config(text="🤖 PBAI reached Goal 1!", fg="#f87171")
            elif self.app.player_pos == self.app.maze.goal2:
                self.app.game_won = True
                self.app.status_label.config(text="🤖 PBAI reached Goal 2!", fg="#60a5fa")

            return True

        return False

def run_pbai_in_tk(app: PlayableMazeApp, max_steps: int = 2000, step_ms: int = 25):
    env = TkMazeEnvAdapter(app)
    wrapper = MazeWithPBAI(env)

    steps = 0
    total_pressure = 0
    decisions_log = []

    def tick():
        nonlocal steps, total_pressure, decisions_log

        if steps >= max_steps or env.at_goal():
            print("\n=== PBAI Run Complete ===")
            print({
                "success": env.at_goal(),
                "steps": steps,
                "total_pressure": total_pressure,
                "final_position": env.player_pos,
                "decisions_summary": {
                    "pay": decisions_log.count("pay"),
                    "produce": decisions_log.count("produce"),
                    "ignore": decisions_log.count("ignore"),
                }
            })
            return

        action = wrapper.get_pbai_suggested_action()
        result = wrapper.step(action)

        steps += 1
        total_pressure += result["pbai_pressure"]
        decisions_log.extend(result["pbai_decisions"])

        if steps % 25 == 0:
            app.status_label.config(
                text=f"🤖 Step {steps} | Pressure={result['pbai_pressure']} | Last={action}",
                fg="#6ee7b7"
            )

        app.root.after(step_ms, tick)

    tick()
if __name__ == "__main__":
    root = tk.Tk()
    app = PlayableMazeApp(root)

    # Save root on the app so run_pbai_in_tk can schedule ticks
    app.root = root

    # Let PBAI drive the maze
    run_pbai_in_tk(app, max_steps=2000, step_ms=25)

    root.mainloop()
