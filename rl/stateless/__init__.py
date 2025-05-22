# TODO:
# Classe método Greedy (com opção para rodarmos o Optimistic Initial Value)
# Classe Epsilon Greedy

from tqdm import tqdm
from typing import (
                    Callable, 
                    List,
                    Dict,
                    Any,
                    )
from rl.stateless.optimizers.base import Optimizer
from rl.stateless.utils import validate_checkpoints_config, checkpoint_policy
from rl.stateless.types import RouteAlias, CheckpointConfigAlias

class StatelessAgent:
    '''
        Represents an agent designed for stateless environments.

        That means that it will always have the same `n` amount of options to choose and their 
        probabilistic distributions will only change based on its past decisions and not on the situation
        it is.
    '''
    def __init__(self, routes:List[RouteAlias], optimizer:Optimizer):
        self.routes = routes
        self.optimizer = optimizer
        self.n_ = 0
        self.routes_stats_ = {route:{'n':0, 'successes':0} for route in routes}

    def _evaluate(self, checkpoint_config:CheckpointConfigAlias):
        self.n_ += 1
        self.routes_stats_ =  self.optimizer.execute(self.routes_stats_) 
        checkpoint_policy(checkpoint_config, self.route_stats_, self.n_)

    def evaluate(self, checkpoint_config:CheckpointConfigAlias=None):
        validate_checkpoints_config(checkpoint_config)
        self.routes_stats_ = self.optimizer.warmup(self.routes_stats_)
        self._evaluate(checkpoint_config)
        return self