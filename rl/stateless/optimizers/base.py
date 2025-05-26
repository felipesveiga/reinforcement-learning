from typing import (
        Callable,
        Any,
)
from rl.stateless.types import RouteAlias, RouteStatsAlias, RoutesStatsAlias

class Optimizer:
    '''
        Base class for any `rl.stateless` optimizer.
        
        Parameter
        ---------
        `warmup`: bool
            A boolean indicating if a warm-up round is necessary.
            Use it in case the algorithm you run crashes when it executes a route for the first time.
    '''
    def __init__(self, warmup:bool):
        self.warmup = warmup
    
    def _choose_route(self, routes_stats:RoutesStatsAlias, **kwargs)->Callable:
        ...

    def _update_stats(self, route:RouteAlias, route_stats:RouteStatsAlias)->RouteStatsAlias:
       route_stats['n']+=1
       route_stats['successes'] += route()  
       return route_stats

    def _warmup(self, routes_stats:RoutesStatsAlias)->RoutesStatsAlias:
        if self.warmup:
            pass
        else:
            for route in routes_stats:
                if routes_stats[route]['n'] == 0:
                    routes_stats[route] = self._update_stats(route, routes_stats[route])
        return routes_stats
            
    def execute(self, routes_stats:RoutesStatsAlias, **kwargs)->RoutesStatsAlias:
        route = self._choose_route(routes_stats, **kwargs)
        routes_stats[route] = self._update_stats(route, routes_stats[route])
        return routes_stats