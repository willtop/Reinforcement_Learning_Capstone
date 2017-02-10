class GameParams:
    """
    tap_position, restart_tap_position, terminal_pixel_position: normalized screen points [x, y]
    terminal_pixel_color: color [r, g, b].
    """
    def __init__(self, tap_position, restart_tap_position, terminal_pixel_position, terminal_pixel_color):
        self.tap_position = tap_position
        self.restart_tap_position = restart_tap_position
        self.terminal_pixel_position = terminal_pixel_position
        self.terminal_pixel_color = terminal_pixel_color

stack_params = GameParams(
    [0.5, 0.5],
    [0.5, 0.5],
    [0.85, 0.84],
    255)
