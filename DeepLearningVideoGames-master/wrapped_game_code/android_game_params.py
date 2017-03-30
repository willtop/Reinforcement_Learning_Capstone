class GameParams:
    """
    tap_position, restart_tap_position, terminal_pixel_position: list of normalized screen points [x, y]
    terminal_pixel_color: list of color [r, g, b].
    """
    def __init__(self, name, tap_position, restart_tap_position, terminal_pixel_position, terminal_pixel_color,
                 is_terminal_state_check_negative = False):
        self.name = name
        self.tap_position = tap_position
        self.restart_tap_position = restart_tap_position
        self.terminal_pixel_position = terminal_pixel_position
        self.terminal_pixel_color = terminal_pixel_color
        self.is_terminal_state_check_negative = is_terminal_state_check_negative

stack_params = GameParams(
    'stack',
    [0.5, 0.5],
    [0.5, 0.5],
    [[0.85, 0.84]],
    [[255, 255, 255]])

flappy_cat_params = GameParams(
    'flappy_cat',
    [0.99, 0.99],
    [0.3, 0.7],
    [[0.3, 0.69], [0.25, 0.67]],
    [[0, 255, 0], [64, 0, 64]])

boom_dots_params = GameParams(
    'boom_dots',
    [0.5, 0.5],
    [0.7, 0.5],
    [[0.2, 0.2]],
    [[0, 255, 0]])

zigzag_params = GameParams(
    'zigzag',
    [0.7, 0.55],
    [0.7, 0.55],
    [[0.7, 0.55]],
    [[225, 225, 225]])

zigzag_boom_params = GameParams(
    'zigzag_boom',
    [0.5, 0.1],
    [0.6, 0.7],
    [[0, 0]],
    [[248, 202, 5]],
    True)
