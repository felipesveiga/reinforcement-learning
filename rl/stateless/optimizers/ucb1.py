from rl.stateless.optimizers.base import StatelessOptimizer
from rl.stateless.types import RouteAlias, RoutesStatsAlias
from numpy import log, argmax

class UCB1(StatelessOptimizer):
    def __init__(self, alpha:float=2):
        self.alpha = alpha
        super().__init__(warmup=True)

    def _choose_route(self, routes_stats:RoutesStatsAlias, n:int)->RouteAlias:
        ...