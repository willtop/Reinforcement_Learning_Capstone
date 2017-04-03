class GameParams:
    """
    tap_position, restart_tap_position, terminal_pixel_position: normalized screen points [x, y]
    terminal_pixel_color: color [r, g, b].
    """
    def __init__(self, tap_position, restart_tap_position, terminal_pixel_position, terminal_pixel_color, score_box):
        self.tap_position = tap_position
        self.restart_tap_position = restart_tap_position
        self.terminal_pixel_position = terminal_pixel_position
        self.terminal_pixel_color = terminal_pixel_color
        self.score_box = score_box

stack_params = GameParams(
    [0.5, 0.5],
    [0.5, 0.5],
    [[0.85, 0.84]],
    [255],
    [140, 120, 205, 170])
    
Jason_stack_params = GameParams(
    [0.5, 0.55],
    [0.5, 0.55],
    # [[0.15, 0.88],[0.85, 0.87],[0.94, 0.1]],
    # [255,255,255],
    [[0.15, 0.88],[0.85, 0.87]],
    [255,255],
    [205, 110, 270, 185])
    
Jason_stack_params_phone = GameParams(
    [0.5, 0.55],
    [0.5, 0.55],
    # [[0.15, 0.88],[0.85, 0.87],[0.94, 0.1]],
    # [255,255,255],
    [[0.14, 0.84],[0.87, 0.85]],
    [255,255],
    [190, 123, 290, 198])
        
Jason_colour_switch_params = GameParams(
    [0.5, 0.55],
    [0.5, 0.55],
    [[0.15, 0.88],[0.85, 0.87]],
    [255,255],
    [205, 110, 270, 185])
    
# Jason_stack_params = GameParams(
    # [0.5, 0.5],
    # [0.5, 0.5],
    # [[0.145, 0.88],[0.85, 0.86]],
    # [255,255],
    # [205, 115, 270, 190])

