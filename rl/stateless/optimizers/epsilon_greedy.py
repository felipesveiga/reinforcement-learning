import numpy as np
from rl.stateless.optimizers.base import StatelessOptimizer 
from rl.stateless.types import RouteAlias, RoutesStatsAlias
from typing import List

E = 10e-8

def _choose_suboptimal(routes:List[RouteAlias], probs:List[float], arg_max:int)->RouteAlias:
    f'''
        Chooses the route when we are in an iteration in which the optimal one 
        is disregarded.

        Parameters
        ----------
        `routes`: List[{RouteAlias}]
            A list with all the possible alternatives the agent can take.
        `props`: List[float]
            A list with all the success rates for the routes.
        `arg_max`: int
            The index of the optimal route.
    
       Returns
       -------
       The chosen function route. 
    '''
    suboptimal_routes = [routes[i] for i in range(len(routes)) if i != arg_max]
    suboptimal_probs = [probs[i]+E for i in range(len(probs)) if i != arg_max]
    return np.random.choice(suboptimal_routes, p=np.array(suboptimal_probs)/sum(suboptimal_probs))

class EpsilonGreedy(StatelessOptimizer):
    '''
        Epsilon-Greedy optimizer.

        Parameters
        ---------
        `eps`: float
            The probability of choosing a suboptimal route.
    '''
    def __init__(self, eps:float):
        if eps<=0 or eps>=1:
            raise ValueError('Invalid `eps` value. It must be in the range ]0,1[') 
        self.eps = eps
        super().__init__(warmup=True)
    
    def _choose_route(self, routes_stats:RoutesStatsAlias, **kwargs)->RouteAlias:
        f'''
            Chooses the route to be taken in a given iteration based on the Epsilon-Greedy algorithm. 

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
        routes = list(routes_stats.keys())
        probs = [routes_stats[route]['successes']/routes_stats[route]['n'] for route in routes]
        arg_max = np.argmax(probs)
        if np.random.random(size=1)>self.eps:
            return routes[arg_max]
        return _choose_suboptimal(routes, probs, arg_max)