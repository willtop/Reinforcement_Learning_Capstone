class GameParams:
    """
    tap_position, restart_tap_position, terminal_pixel_position: normalized screen points [x, y]
    terminal_pixel_color: color [r, g, b].
    """
    def __init__(self, tap_position, restart_tap_position, terminal_pixel_position, 
                terminal_pixel_color, scoring_pixel_position, scoring_pixel_color, score_box):
        self.tap_position = tap_position
        self.restart_tap_position = restart_tap_position
        self.terminal_pixel_position = terminal_pixel_position
        self.terminal_pixel_color = terminal_pixel_color
        self.scoring_pixel_position = scoring_pixel_position
        self.scoring_pixel_color = scoring_pixel_color
        self.score_box = score_box

    
boom_dots_params = GameParams(
    [0.75, 0.65], # tap_position; make it different from restart tap position to avoid mistakenly restart
    [0.7, 0.75], # restart_tap_position; confirmed
    [[0.3, 0.75], [0.7, 0.75]], # terminal_pixel_position; confirmed
    [136, 136], # terminal_pixel_color; confirmed
    [[0.5, 0.2]], # scoring_pixel_position;
    [87], # scoring_pixel_color;
    [130, 80, 200, 167] # confirmed
    )
