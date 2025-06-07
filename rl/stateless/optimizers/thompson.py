import numpy as np
from rl.stateless.optimizers.base import StatelessOptimizer 
from rl.stateless.types import (
    RouteAlias, 
    RoutesStatsAlias,
    BetaParametersAlias
)
from typing import List

class Thompson(StatelessOptimizer):
    
    def __init_betas(self, routes_stats:RoutesStatsAlias):
        self.betas_parameters_ = {route:{'a':1,'b':1} for route in routes_stats}

    def __init__(self, warmup:bool):
        self.betas_parameters_ = None
        super().__init__(warmup=warmup)

    def _choose_route(self, routes_stats, **kwargs):
        if not self.betas_parameters_:
            self.__init_betas()