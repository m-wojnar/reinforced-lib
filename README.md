# Reinforced-lib: Reinforcement learning library

## Overview

Reinforced-lib is a Python library designed to support research and prototyping using Reinforced Learning (RL) algorithms. The library can serve as a simple solution with ready to use RL workflows, as well as an expandable framework with programmable behaviour. Thanks to a fuctional implementation of librarys core, we can provide full acces to JAX’s jit functionality, which boosts the agents performance significantly.

## Example code

```python
import gym

import reinforced_lib as rfl
from reinforced_lib.agents import ThompsonSampling
from reinforced_lib.exts import IEEE_802_11_ax

rlib = rfl.RLib(
    agent_type=ThompsonSampling,
    ext_type=IEEE_802_11_ax
)

env = gym.make('WifiSimulator-v1')

state = env.reset()
done = False

while not done:
    action = rlib.sample(**state)
    state, reward, done, info = env.step(action)
```

## Integrated IEEE 802.11ax support

Library design is distinctly influenced by the desire to support research in Wi-Fi. It can be a tool for researchers to optimize the Wi-Fi protocols with built-in RL algorithms and provided IEEE 802.11ax environment extension.

We also provide simple [ns-3](https://www.nsnam.org/) simulation and RL-based rate adaptation manager for the IEEE 802.11ax standard in [examples](https://github.com/m-wojnar/reinforced-lib/tree/main/examples/ns-3).

## Citing Reinforced-lib

To cite this repository:

```
@software{reinforcedlib2022,
  author = {Maksymilian Wojnar and Wojciech Ciężobka},
  title = {{R}einforced-lib: {R}einforcement learning library},
  url = {http://github.com/m-wojnar/reinforced-lib},
  year = {2022},
}
```
