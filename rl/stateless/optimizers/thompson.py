from typing import List
from numpy import argmax
from scipy.stats import beta
from rl.stateless.optimizers.base import StatelessOptimizer 
from rl.stateless.types import (
    RouteAlias, 
    RoutesStatsAlias,
    RouteStatsAlias,
)

class Thompson(StatelessOptimizer):
    '''
        Implementation of the Thompson Sampling/Bayesian Bandits optimizer for
        Multi-Armed Bandits like cases.
    '''
    def __init__(self):
        super().__init__(warmup=True)

    def _warmup(self, routes_stats:RoutesStatsAlias)->RoutesStatsAlias:
        f'''
            Customized warm-up desgined for Thompson Sampling, in which we 
            initialize the beta distributions' parameters 

            Parameter
            ---------
            `routes_stats`: {RoutesStatsAlias} 
                The dictionary that points routes to their respective stats.
            
            Returns
            -------
            The initialized `routes_stats` dictionary.
        '''
        if self.warmup:
            for route in routes_stats:
                if routes_stats[route]['n'] == 0:
                    routes_stats[route].update({'a':1,'b':1})
        return routes_stats

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
       reward = route()  
       route_stats['n']+=1
       route_stats['successes'] += reward 
       route_stats['a'] += reward
       route_stats['b'] += route_stats['n'] - reward
       return route_stats

    def _choose_route(self, routes_stats:RoutesStatsAlias,**kwargs):
        f'''
            Chooses the current iteration's route based on the Thompson Sampling 
            algorithm.
            
           Parameter
           ---------
            `route_stats`: {RouteStatsAlias}
                The dictionary with the route's stats.
       
            Returns
            -------
            The route to be executed.
        '''
        routes = [route for route in routes_stats]
        means = [beta.mean(routes_stats[route]['a'], routes_stats[route]['b']) for route in routes]
        return routes[argmax(means)]