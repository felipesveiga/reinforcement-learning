from numpy import argmax
from random import random, choice
from rl.stateless.optimizers.base import Optimizer
from rl.stateless.types import RouteAlias, RoutesStatsAlias

class EpsilonGreedy(Optimizer):
    def __init__(self, eps:float, no_warmup:bool):
        self.eps = eps
        super().__init__(no_warmup)
    
    def _choose_route(self, routes_stats:RoutesStatsAlias, n:int)->RouteAlias:
        routes = list(routes_stats.keys())
        chosen_idx = argmax([routes_stats[route]['successes']/n for route in routes])
        if random()>self.eps:
            return routes[chosen_idx]
        return choice(list(set(routes).difference(routes[chosen_idx])))