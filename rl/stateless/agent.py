from typing import List
from rl.stateless.optimizers.base import StatelessOptimizer
from rl.stateless.utils import validate_checkpoints_config, checkpoint_policy
from rl.stateless.types import (
    RouteAlias, 
    CheckpointConfigAlias, 
    RoutesStatsAlias
)

class StatelessAgent:
    f'''
        Represents an agent designed for stateless environments.

        That means that it will always have the same `n` amount of options to choose and their 
        probabilistic distributions will only change based on its past decisions and not on the situation
        it is.

        Parameters
        ----------
        `routes`: List[{RouteAlias}]
            A list with all the possible alternatives the agent can take.
        `optimizer`: `StatelessOptimizer`
            The optimizer algorithm used in the route selection.
        `checkpoint_config`: {CheckpointConfigAlias}
            A dictionary with the configurations for checkpoint saving. It must provide the 
            iterations interval in which the metadata is stored and the target directory.
            {{
                "n": "The amount of iterations to be awaited before saving",
                "output_path":"The target directory"
            }}

        Attributes
        ----------
        `n_`: int
            The amount of executions done so far.
        `routes_stats_`: {RoutesStatsAlias}
            The dictionary pointing the routes methods to their respective usage stats.

        Methods
        -------
        `evaluate`: Performs an evaluation.
    '''
    def __init__(
                 self, 
                 routes:List[RouteAlias], 
                 optimizer:StatelessOptimizer, 
                 checkpoint_config:CheckpointConfigAlias=None
                 )->None:
        validate_checkpoints_config(checkpoint_config)
        self.routes = routes
        self.optimizer = optimizer
        self.checkpoint_config = checkpoint_config
        self.n_ = 0
        self.routes_stats_ = {route:{'n':0, 'successes':0} for route in routes}

    def _evaluate(self):
        '''
            Performs the evaluation, verifying whether checkpointing is necessary.
        '''
        self.n_ += 1
        self.routes_stats_ =  self.optimizer.execute(self.routes_stats_, n=self.n_) 
        checkpoint_policy(self.checkpoint_config, self.routes_stats_, self.n_)

    def evaluate(self):
        '''
            Performs an evaluation. 
        '''
        self.routes_stats_ = self.optimizer._warmup(self.routes_stats_)
        self._evaluate()
        return self