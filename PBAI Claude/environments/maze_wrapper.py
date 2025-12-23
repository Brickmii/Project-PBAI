from pbai_root.pbai.translator import translate

ACTION_TO_MOVEMENT = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}
state = translate(self.agent_id, movement)

class MazeWrapper:
    def __init__(self, maze_env, agent_id="maze_agent_1"):
        self.maze = maze_env
        self.agent_id = agent_id

    def step(self, action: str):
        success = self.maze.move(action)

        movement = ACTION_TO_MOVEMENT.get(action, "_")
        if not success:
            movement = "_"

        at_goal = self.maze.at_goal()

        # Update PBAI internal state ONLY
        state = translate(
            identity=self.agent_id,
            external_motion=movement,
        )

        return {
            "state": state,
            "movement": movement,
            "at_goal": at_goal,
        }
