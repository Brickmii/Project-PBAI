import tkinter as tk
import random


class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        self.start = (1, 1)
        self.goal1 = None
        self.goal2 = None

    def generate(self):
        # Generate maze using recursive backtracking
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        stack = [self.start]
        self.grid[self.start[0]][self.start[1]] = 0
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

        while stack:
            current = stack[-1]
            row, col = current
            neighbors = []
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 1 <= nr < self.height - 1 and 1 <= nc < self.width - 1 and self.grid[nr][nc] == 1:
                    neighbors.append((nr, nc))

            if neighbors:
                next_cell = random.choice(neighbors)
                nr, nc = next_cell
                wall_row = row + (nr - row) // 2
                wall_col = col + (nc - col) // 2
                self.grid[wall_row][wall_col] = 0
                self.grid[nr][nc] = 0
                stack.append(next_cell)
            else:
                stack.pop()

        # Now place two random goals
        self.place_random_goals()

        # Ensure start is open
        self.grid[self.start[0]][self.start[1]] = 0

    def place_random_goals(self):
        """Place two goals at random valid positions"""
        # Get all open cells (not walls)
        open_cells = []
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if self.grid[row][col] == 0:
                    # Only consider cells far enough from start
                    dist_to_start = abs(row - self.start[0]) + abs(col - self.start[1])
                    if dist_to_start >= 8:  # Minimum distance from start
                        open_cells.append((row, col))

        if len(open_cells) < 2:
            # Fallback: use default positions if not enough space
            self.goal1 = (self.height - 4, self.width - 4)
            self.goal2 = (self.height - 6, self.width - 7)
            self.grid[self.goal1[0]][self.goal1[1]] = 0
            self.grid[self.goal2[0]][self.goal2[1]] = 0
            return

        # Pick first goal randomly
        self.goal1 = random.choice(open_cells)
        open_cells.remove(self.goal1)

        # Pick second goal far from first
        valid_goals = []
        for cell in open_cells:
            dist = abs(cell[0] - self.goal1[0]) + abs(cell[1] - self.goal1[1])
            if dist >= 6:  # Minimum distance between goals
                valid_goals.append(cell)

        if valid_goals:
            self.goal2 = random.choice(valid_goals)
        else:
            # If none are far enough, pick any
            self.goal2 = random.choice(open_cells)

        # Ensure goals are open (they should be, but just in case)
        self.grid[self.goal1[0]][self.goal1[1]] = 0
        self.grid[self.goal2[0]][self.goal2[1]] = 0


class PlayableMazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("20x20 Maze - Find Either Random Goal!")
        self.root.geometry("800x780")
        self.root.configure(bg="#0f1923")

        self.maze_size = 20
        self.cell_size = 25
        self.maze = MazeGenerator(self.maze_size, self.maze_size)
        self.player_pos = self.maze.start
        self.game_won = False

        self.create_widgets()
        self.new_game()

        # Bind keys
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.focus_set()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="Find Either Random Goal to Win!",
            font=("Arial", 18, "bold"),
            bg="#0f1923",
            fg="#e0e0e0"
        )
        title.pack(pady=10)

        # Canvas
        canvas_frame = tk.Frame(self.root, bg="#1d3557", bd=3, relief="sunken")
        canvas_frame.pack(padx=20, pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.maze_size * self.cell_size,
            height=self.maze_size * self.cell_size,
            bg="#0a1929"
        )
        self.canvas.pack()

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#0f1923")
        btn_frame.pack(pady=15)

        self.new_game_btn = tk.Button(
            btn_frame,
            text="New Maze",
            font=("Arial", 12, "bold"),
            bg="#4ade80",
            fg="black",
            width=12,
            command=self.new_game
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=12)

        self.restart_btn = tk.Button(
            btn_frame,
            text="Restart",
            font=("Arial", 12, "bold"),
            bg="#fb923c",
            fg="black",
            width=12,
            command=self.restart
        )
        self.restart_btn.pack(side=tk.LEFT, padx=12)

        # Legend
        legend = tk.Frame(self.root, bg="#0f1923")
        legend.pack(pady=8)

        tk.Label(legend, text="🟨 You", font=("Arial", 10), bg="#0f1923", fg="#fde047").pack(side=tk.LEFT, padx=8)
        tk.Label(legend, text="🟥 Goal 1", font=("Arial", 10), bg="#0f1923", fg="#f87171").pack(side=tk.LEFT, padx=8)
        tk.Label(legend, text="🟦 Goal 2", font=("Arial", 10), bg="#0f1923", fg="#60a5fa").pack(side=tk.LEFT, padx=8)

        # Status
        self.status_label = tk.Label(
            self.root,
            text="Use WASD or Arrow Keys to move",
            font=("Arial", 12),
            bg="#0f1923",
            fg="#6ee7b7"
        )
        self.status_label.pack(pady=5)

    def new_game(self):
        self.maze.generate()
        self.player_pos = self.maze.start
        self.game_won = False
        self.draw_maze()
        self.status_label.config(text="Navigate to either goal!", fg="#6ee7b7")

    def restart(self):
        self.player_pos = self.maze.start
        self.game_won = False
        self.draw_maze()
        self.status_label.config(text="Navigate to either goal!", fg="#6ee7b7")

    def draw_maze(self):
        self.canvas.delete("all")

        for row in range(self.maze_size):
            for col in range(self.maze_size):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                if self.maze.grid[row][col] == 1:
                    # Wall
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill="#374151",
                        outline="#111827"
                    )
                else:
                    # Path
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill="#0a1929",
                        outline="#1e293b"
                    )

        # Draw Goal 1 (red)
        self.draw_goal(self.maze.goal1, "#f87171", "G1")
        # Draw Goal 2 (blue)
        self.draw_goal(self.maze.goal2, "#60a5fa", "G2")
        # Draw player
        self.draw_player()

    def draw_goal(self, pos, color, label):
        if pos is None:
            return
        row, col = pos
        x1 = col * self.cell_size + 4
        y1 = row * self.cell_size + 4
        x2 = x1 + self.cell_size - 8
        y2 = y1 + self.cell_size - 8
        self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="white", width=2)
        self.canvas.create_text(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            text=label,
            fill="white",
            font=("Arial", 8, "bold")
        )

    def draw_player(self):
        row, col = self.player_pos
        x1 = col * self.cell_size + 5
        y1 = row * self.cell_size + 5
        x2 = x1 + self.cell_size - 10
        y2 = y1 + self.cell_size - 10
        self.canvas.create_oval(x1, y1, x2, y2, fill="#fde047", outline="#facc15", width=2)

    def on_key_press(self, event):
        if self.game_won:
            return

        key = event.keysym.lower()
        dr, dc = 0, 0

        if key in ['up', 'w']:
            dr = -1
        elif key in ['down', 's']:
            dr = 1
        elif key in ['left', 'a']:
            dc = -1
        elif key in ['right', 'd']:
            dc = 1
        else:
            return

        new_row = self.player_pos[0] + dr
        new_col = self.player_pos[1] + dc

        # Validate move
        if (0 <= new_row < self.maze_size and
                0 <= new_col < self.maze_size and
                self.maze.grid[new_row][new_col] == 0):

            self.player_pos = (new_row, new_col)
            self.draw_maze()

            # Check win
            if self.player_pos == self.maze.goal1:
                self.game_won = True
                self.status_label.config(text="🎉 You reached Goal 1! Victory!", fg="#f87171")
            elif self.player_pos == self.maze.goal2:
                self.game_won = True
                self.status_label.config(text="🎉 You reached Goal 2! Victory!", fg="#60a5fa")


if __name__ == "__main__":
    root = tk.Tk()
    app = PlayableMazeApp(root)
    root.mainloop()