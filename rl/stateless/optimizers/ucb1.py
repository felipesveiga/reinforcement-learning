from rl.stateless.optimizers.base import StatelessOptimizer
from rl.stateless.types import RouteAlias, RoutesStatsAlias
from numpy import sqrt, log, argmax

ALPHA = 2

class UCB1(StatelessOptimizer):
    f'''
        UBC1 Optimizer. It chooses the routes based on the Hoeffding Inequality. 

        Parameter
        ---------
        `alpha`: float, defaults to {ALPHA}
            The alpha parameter. Defines the intensity of exploration of the algorithm

    '''
    def __init__(self, alpha:float=ALPHA):
        self.alpha = alpha
        super().__init__(warmup=True)

    def _choose_route(self, routes_stats:RoutesStatsAlias,**kwargs)->RouteAlias:
        f'''
            Chooses the route to be taken in a given iteration based on the UCB1 algorithm. 

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
        n, routes = kwargs['n'], [route for route in routes_stats]
        averages = [routes_stats[route]['successes']/routes_stats[route]['n'] for route in routes]
        ucbs = [averages[i]+sqrt(self.alpha*log(n)/routes_stats[route]['n']) for i,route in enumerate(routes)]
        return routes[argmax(ucbs)]