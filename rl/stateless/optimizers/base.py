from typing import (
        Callable,
        Any,
)
from rl.stateless.types import RouteAlias, RouteStatsAlias, RoutesStatsAlias

class Optimizer:
    def __init__(self, no_warmup:bool):
        self.no_warmup = None
    
    def _choose_route(self, routes_stats:RoutesStatsAlias, n:int)->Callable:
        ...

    def _update_stats(self, route:RouteAlias, route_stats:RouteStatsAlias)->RouteStatsAlias:
       route_stats['n']+=1
       route_stats['successes'] += route()  
       return route_stats

    def warmup(self, routes_stats:RoutesStatsAlias)->RoutesStatsAlias:
        if self.no_warmup:
            pass
        else:
            for route in routes_stats:
                if routes_stats[route]['n'] == 0:
                    routes_stats[route] = self._update_stats(route, routes_stats[route])
        return routes_stats
            
    def execute(self, routes_stats:RoutesStatsAlias)->RoutesStatsAlias:
        route = self._choose_route()
        routes_stats[route] = self._update_stats(route, routes_stats[route])
        return routes_stats