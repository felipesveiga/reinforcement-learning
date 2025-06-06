from rl.stateless.optimizers.base import StatelessOptimizer
from rl.stateless.types import RouteAlias, RoutesStatsAlias
from numpy import sqrt, log, argmax

class UCB1(StatelessOptimizer):
    def __init__(self, alpha:float=2):
        self.alpha = alpha
        super().__init__(warmup=True)

    def _choose_route(self, routes_stats:RoutesStatsAlias,**kwargs)->RouteAlias:
        n, routes = kwargs['n'], [route for route in routes_stats]
        averages = [routes_stats[route]['successes']/routes_stats[route]['n'] for route in routes]
        ucbs = [averages[i]+sqrt(self.alpha*log(n)/routes_stats[route]['n']) for i,route in enumerate(routes)]
        return routes[argmax(ucbs)]

    def execute(self, routes_stats, **kwargs):
        return super().execute(routes_stats, **kwargs)