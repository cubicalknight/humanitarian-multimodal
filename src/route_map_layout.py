"""Keep notebook route maps at the feasible-route comparison's window size."""

MAP_FIGURE_SIZE = (6.5, 4.8)
# Measured from the original feasible-route map at shipment 22-0237, K=5.
MAP_WINDOW_WIDTH_INCHES = 5.238372813201111
MAP_WINDOW_HEIGHT_INCHES = 3.12
MAP_WINDOW_BOUNDS = (
    (1 - MAP_WINDOW_WIDTH_INCHES / MAP_FIGURE_SIZE[0]) / 2,
    0.24,
    MAP_WINDOW_WIDTH_INCHES / MAP_FIGURE_SIZE[0],
    MAP_WINDOW_HEIGHT_INCHES / MAP_FIGURE_SIZE[1],
)


def apply_map_layout(axis):
    """Expand the projected extent to fill the fixed window without distortion.

    Call after setting the route extent and before placing labels or computing
    display-space offsets. Expanding limits preserves all existing route bounds
    while preventing Cartopy's equal-aspect adjustment from shrinking the axes.
    """
    axis.figure.set_size_inches(*MAP_FIGURE_SIZE)
    axis.set_position(MAP_WINDOW_BOUNDS)
    axis.set_aspect('equal', adjustable='box')
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_span, y_span = x_max - x_min, y_max - y_min
    aspect = MAP_WINDOW_WIDTH_INCHES / MAP_WINDOW_HEIGHT_INCHES
    if x_span / y_span < aspect:
        x_center = (x_min + x_max) / 2
        half_width = y_span * aspect / 2
        axis.set_xlim(x_center - half_width, x_center + half_width)
    else:
        y_center = (y_min + y_max) / 2
        half_height = x_span / aspect / 2
        axis.set_ylim(y_center - half_height, y_center + half_height)
