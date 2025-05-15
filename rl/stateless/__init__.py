# TODO:
# Classe método Greedy (com opção para rodarmos o Optimistic Initial Value)
# Classe Epsilon Greedy

from typing import (
                    Callable, 
                    List,
                    Any,
                    )
from rl.stateless.optimizers import Optimizer

class StatelessAgent:
    '''
        Represents an agent designed for stateless environments.

        That means that it will always have the same `n` amount of options to choose and their 
        probabilistic distributions will only change based on its past decisions and not on the situation
        it is.
    '''
    def __init__(self, routes:List[Callable[[Any], float]], optimizer:Optimizer):
        self.routes = routes
        self.optimizer = Optimizer
        self.routes_probs_ = {route:0 for route in routes}

    def simulate(self, max_iter:int, iter_checkpoints:int|None):
        self.routes_probs_  = self.optimizer.warmup(self)
        for i in range(max_iter):
            ...