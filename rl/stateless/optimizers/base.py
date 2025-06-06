from typing import (
        Callable,
        Any,
)
from rl.stateless.types import RouteAlias, RouteStatsAlias, RoutesStatsAlias

class StatelessOptimizer:
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
        f'''
            Base Method for choosing the route to be taken in a given iteration. 

            Parameters
            ----------
            `routes_stats`: {RoutesStatsAlias}
                A dictionary with all the execution data from the routes.
            `**kwargs`: 
                Additional arguments to be informed depending on the optimizer.

            Returns
            -------
            The chosen route to be taken.
        '''
        ...

    def _update_stats(self, route:RouteAlias, route_stats:RouteStatsAlias)->RouteStatsAlias:
       f'''
           Executes a provided route, updating its performance stats.   

           Parameters
           ----------
           `route`: {RouteAlias} 
                The route function to be executed.
            `route_stats`: {RouteStatsAlias}
                The dictionary with the route's stats.
       
            Returns
            -------
            The updated stats dictionary.
       '''
       route_stats['n']+=1
       route_stats['successes'] += route()  
       return route_stats

    def _warmup(self, routes_stats:RoutesStatsAlias)->RoutesStatsAlias:
        f'''
            Invokes a warm-up step, if necessary for the algorithm.

            Parameter
            ---------
            `routes_stats`: {RoutesStatsAlias} 
                The dictionary that points routes to their respective stats.
            
            Returns
            -------
            The dictionary updated by the warm-up stage if it was necessary. Otherwise 
            the unmodified data is returned.
        '''
        if self.warmup:
            for route in routes_stats:
                if routes_stats[route]['n'] == 0:
                    routes_stats[route] = self._update_stats(route, routes_stats[route])
        return routes_stats
            
    def execute(self, routes_stats:RoutesStatsAlias, **kwargs)->RoutesStatsAlias:
        '''
            Chooses and executes a route for a given iteration.

            Parameter
            ---------
            `routes_stats`: {RoutesStatsAlias} 
                The dictionary that points routes to their respective stats.
            `**kwargs`: 
                Additional arguments to be informed depending on the optimizer.

            Returns
            -------
            The updated route to stats dictionary.
        '''
        route = self._choose_route(routes_stats, **kwargs)
        routes_stats[route] = self._update_stats(route, routes_stats[route])
        return routes_stats