import numpy as np
from rl.stateless.optimizers.base import Optimizer
from rl.stateless.types import RouteAlias, RoutesStatsAlias
from typing import List

def _choose_suboptimal(routes:List[RouteAlias], probs:List[float], arg_max:int):
    suboptimal_routes = [routes[i] for i in range(len(routes)) if i != arg_max]
    suboptimal_probs = [probs[i] for i in range(len(probs)) if i != arg_max]
    return np.random.choice(suboptimal_routes, p=np.array(suboptimal_probs)/sum(suboptimal_probs))

class EpsilonGreedy(Optimizer):
    def __init__(self, eps:float, no_warmup:bool):
        self.eps = eps
        super().__init__(no_warmup)
    
    def _choose_route(self, routes_stats:RoutesStatsAlias, n:int)->RouteAlias:
        routes = list(routes_stats.keys())
        probs = [routes_stats[route]['successes']/routes_stats[route]['n'] for route in routes]
        arg_max = np.argmax(probs)
        if np.random(size=1)>self.eps:
            return routes[arg_max]
        return _choose_suboptimal(routes, probs, arg_max)