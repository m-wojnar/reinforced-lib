# Reinforced-lib: Reinforcement learning library

![build and test](https://github.com/m-wojnar/reinforced-lib/actions/workflows/python-package.yml/badge.svg)
![docs](https://readthedocs.org/projects/reinforced-lib/badge/?version=latest)

## Overview

Reinforced-lib is a Python library designed to support research and prototyping using Reinforced Learning (RL) 
algorithms. The library can serve as a simple solution with ready to use RL workflows, as well as an expandable 
framework with programmable behaviour. Thanks to a functional implementation of the library's core, we can provide 
full access to JAX’s jit functionality, which boosts the agents performance significantly.

## Installation

You can install the latest released version of Reinforced-lib from PyPI via:

```bash
pip install reinforced-lib
```

You can also download source code and install the development dependencies if you want to build the documentation locally:

```bash
git clone git@github.com:m-wojnar/reinforced-lib.git
cd reinforced-lib
pip3 install ".[dev]"
```

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

Library design is distinctly influenced by the desire to support research in Wi-Fi. It can be a tool for researchers 
to optimize the Wi-Fi protocols with built-in RL algorithms and provided IEEE 802.11ax environment extension.

We also provide simple [ns-3](https://www.nsnam.org/) simulation and RL-based rate adaptation manager for the 
IEEE 802.11ax standard in [examples](https://github.com/m-wojnar/reinforced-lib/tree/main/examples/ns-3).

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
