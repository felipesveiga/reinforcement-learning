from typing import (
    TypeAlias,
    Dict,
    Callable
)

RouteStatsAlias: TypeAlias = Dict[str, int]
RoutesStatsAlias: TypeAlias = Dict[Callable, Dict]
CheckpointConfigAlias: TypeAlias = Dict[str, str|int] | None

