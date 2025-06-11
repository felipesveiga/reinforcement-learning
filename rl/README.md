# `rl` package
This is a home-made Reinforcement Learning library developed solely for educational purposes.

Here, you can find most of the well-known RL algorithms separated in the following modules:
* `stateless`: Contains the implementation of objects used in Multi-Armed Bandits like problems, in which there is no concept of state transitioning.

## The `stateless` Module
Its main class is `StatelessAgent`. It represents the agent that will carry out the execution of the options (or routes) available.
You need to initialize it with an optimizer from the `stateless.optimizers` submodule. Currently, it only has the following optimizers:
* Epsilon-Greedy
* UCB1
* Thompson Sampling / Bayesian Bandits

Here is a sample code for guidance:

```python
import numpy as np
from rl.stateless import StatelessAgent
from rl.stateless.optimizers import Thompson

def best():
    if np.random.random()<.9:
        return 1
    return 0

def worst():
    if np.random.random()<.35:
        return 1
    return 0

agent = StatelessAgent(routes=[best, worst], optimizer=Thompson(), checkpoint_config={'n':10, 'output_path':'data/'})

for _ in range(1000):
    agent.evaluate()
```