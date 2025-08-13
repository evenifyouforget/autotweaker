from collections import namedtuple

Waypoint = namedtuple('Waypoint', ['x', 'y', 'rotation'])

def generate_waypoints(screenshot: np.ndarray,
                       limit_time_seconds: float | None = None,
                       limit_iterations: int | None = None) -> list[Waypoint]:
    # TODO
    # note screenshot will be shape (Y, X) with integer "colors"
    pass