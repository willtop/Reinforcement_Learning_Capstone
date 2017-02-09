class GameParams:
    """
    tap_position, restart_tap_position: normalized screen points [x, y]
    terminal_state_detector: dictionary that maps normalized screen points [x, y] to colors [r, g, b].
    """
    def __init__(self, tap_position, restart_tap_position, terminal_state_map):
        self.tap_position = tap_position
        self.restart_tap_position = restart_tap_position
        self.terminal_state_map = terminal_state_map

stack_params = GameParams(
    [0.5, 0.5],
    [0.5, 0.5],
    {
        [0.5, 0.8]: [255, 255, 255],
        [0.3, 0.8]: [255, 255, 255],
    })
