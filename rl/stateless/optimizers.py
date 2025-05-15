from abc import ABC
from typing import Dict, Callable

class Optimizer(ABC):
    def __init__(self, no_warmup):
        self.no_warmup = None

    def warmup(self, routes_probs:Dict[Callable, float]):
        if self.no_warmup:
            return routes_probs
        else:
            for item in routes_probs:
                ...
    def pick():
        ...