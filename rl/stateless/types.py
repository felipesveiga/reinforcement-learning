from typing import (
    TypeAlias,
    Dict,
    Callable,
    Any,
)

RouteAlias: TypeAlias = Callable[[Any], int]
RouteStatsAlias: TypeAlias = Dict[str, int]
RoutesStatsAlias: TypeAlias = Dict[Callable, Dict]
CheckpointConfigAlias: TypeAlias = Dict[str, str|int] | None
BetaParametersAlias: Dict[Callable, Dict[str, float]]