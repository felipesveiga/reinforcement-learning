from typing import Callable, List, Dict

class EpsilonGreedy:
    def __init__(self, eps:float, max_iter:int, scheduler:Callable[[float], float], callbacks:List[Callable]):
        self.eps = eps
        self.max_iter = max_iter
        self.scheduler = scheduler
        self.callbacks = callbacks

    def simulate(routes:Dict[str, Callable]):
        ...