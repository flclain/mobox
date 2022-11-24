from matplotlib import colors


def get_color(idx):
    h = (37 * idx) % 360 / 360.
    color = (1, 1, 1) if idx < 0 else colors.hsv_to_rgb([h, 0.5, 1])
    return color


def get_style_map():
    """Map element type to line style & color.

    Reference: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/map.proto
    """
    style_map = {
        # Freeway.
        ("lane", 1): ["--", "grey"],
        # Surface street.
        ("lane", 2): ["--", "grey"],
        # Bike lane.
        ("lane", 3): ["--", "grey"],

        # Physical road boundary that doesn't have traffic on the other side (e.g.,a curb or the k-rail on the right side of a freeway).
        ("road_edge", 1): ["-", "green"],
        # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
        ("road_edge", 2): ["--", "green"],

        # Broken single white.
        ("road_line", 1): ["-", "grey"],
        # Solid single white.
        ("road_line", 2): ["-", "grey"],
        # Solid double white.
        ("road_line", 3): ["-", "grey"],
        # Broken single yellow.
        ("road_line", 4): ["--", "yellow"],
        # Broken double yellow.
        ("road_line", 5): ["--", "yellow"],
        # Solid single yellow.
        ("road_line", 6): ["-", "yellow"],
        # Solid double yellow.
        ("road_line", 7): ["-", "yellow"],
        # Passing double yellow.
        ("road_line", 8): ["--", "yellow"],
    }
    return style_map
