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

### Checkpointing
The user is able to save the routes' stats for every $n$ iterations. To do so, they must pass a dictionary as the value for the `StatelessAgent`'s `checkpoint_config` argument with the following schema.

```json
 {
    "n":"The number of rounds to await the next register of stats",
    "output_path":"The directory in which the checkpoints are stored"
 }
```

Here is a sample code for guidance:

```python
import numpy as np
from rl.stateless import StatelessAgent
from rl.stateless.optimizers import Thompson

def best():
    '''
        The most favorable route. 
    '''
    if np.random.random()<.9:
        return 1
    return 0

def worst():
    '''
        The least favorable route.
    '''
    if np.random.random()<.35:
        return 1
    return 0

# Instantiating our agent.
agent = StatelessAgent(routes=[best, worst], optimizer=Thompson(), checkpoint_config={'n':10, 'output_path':'data/'})

# Simulating the decision-making process for 1000 rounds.
for _ in range(1000):
    agent.evaluate()
```