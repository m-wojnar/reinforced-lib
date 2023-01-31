.. _examples_page:

########
Examples
########


.. _ns3_connection:

********************
Connection with ns-3
********************

We will demonstrate the cooperation of Reinforced-lib with an external WiFi simulation software based on an example of
an ML-controlled rate adaptation manager. To simulate the WiFi environment, we will use the popular, research oriented
network simulator - ns-3. To learn more about the simulator, we encourage to visit the
`official website <https://www.nsnam.org/>`_ or read the
`ns-3 tutorial <https://www.nsnam.org/docs/release/3.36/tutorial/html/index.html>`_.


Environment setup
=================

To perform experiments with Python-based Reinforced-lib and C++-based ns-3, you need to setup an environment which
consists of the following:

  * favourite C++ compiler (we assume that you already have one in your dev stack),
  * ns-3 (connection tested on the ns-3.36.1 version),
  * ns3-ai (https://github.com/hust-diangroup/ns3-ai/).


Installing ns-3
---------------

There are a few ways to install ns-3, all described in the `ns-3 wiki <https://www.nsnam.org/wiki/Installation>`_,
but we recommend to install ns-3 by cloning the git dev repository:

.. code-block:: bash

    git clone https://gitlab.com/nsnam/ns-3-dev.git

Then change directory to the newly created ``ns-3-dev/`` and build ns-3:

.. code-block:: bash

    ./ns3 configure --enable-examples --build-profile=optimized
    ./ns3 build

Once you have built ns-3 (with examples enabled), it should be easy to run the sample programs with the following command:

.. code-block:: bash

    ./ns3 run wifi-simple-adhoc

If the installation process succeeded, you should have two new ``.pcap`` files in the ``ns-3-dev/`` directory, namely
``wifi-simple-adhoc-0-0.pcap`` and ``wifi-simple-adhoc-1-0.pcap``


Installing ns3-ai
-----------------

The ns3-ai module interconnects ns-3 and Reinforced-lib (or any other python-writen software) by transferring data through
the shared memory pool. The memory is accessed by both sides thus making the connection. You can read more about the ns3-ai on the
`ns3-ai official repository <https://github.com/hust-diangroup/ns3-ai>`_. To install the module, clone the ns3-ai repository to the
``ns-3-dev/contrib/`` directory and install the ``py_interface`` module with pip:

.. code-block:: bash

    cd $YOUR_NS3_PATH/contrib/
    git clone https://github.com/hust-diangroup/ns3-ai.git
    pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"

To have ns3-ai working, you need to rebuild ns-3, but before doing so, we transfer some necessary files to
enable experiments with the provided rate adaptation manager.

.. warning::

    ns3-ai (as of 18.09.2022) is not compatible with the ns-3.36 or later. We have forked and modified the official
    ns3-ai repository to make it compatible with the 3.36 version. In order to install our compatible version run the
    following commands instead:

    .. code-block:: bash

        cd $YOUR_NS3_PATH/contrib/
        git clone https://github.com/m-wojnar/ns3-ai.git
        cd "$YOUR_NS3_PATH/contrib/ns3-ai/"
        git checkout ml4wifi
        pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"


Transferring RA example files
-----------------------------

In ``$REINFORCED_LIB/examples/ns-3/`` there are two directories. The ``rlib-sim`` contains an
example scenario, which will be described in the :ref:`next section <rlib-sim>`. The ``rlib-wifi-manager`` directory
contains an ns-3 contrib module with the specification of a custom rate adaptation manager that communicates with python
with the use of ns3-ai. You need to transfer both of these directories in the appropriate locations by running the
following commands:

.. code-block:: bash

    cp -r $REINFORCED_LIB/examples/ns-3/rlib-sim $YOUR_NS3_PATH/scratch/
    cp -r $REINFORCED_LIB/examples/ns-3/rlib-wifi-manager $YOUR_NS3_PATH/contrib/

.. note::

    To learn more about adding contrib modules to ns-3, visit
    the `ns-3 manual <https://www.nsnam.org/docs/manual/html/new-modules.html>`_.


.. _rlib-sim:

Simulation scenario
===================


ns-3 (C++) part
---------------

We supply an example scenario ``rlib-sim/sim.cc`` to test the rate adaptation manager in the 802.11ax environment.
The scenario is highly customizable but the key points
are that there is one access point (AP) and a variable number (``--nWifi``) of stations (STA); there is an uplink, saturated
communication (from stations to AP) and the AP is in line of sight with all the stations; all the stations are at the point of
:math:`(0, 0)~m` and the AP can either be at :math:`(0, 0)~m` as well or in some distance (``--initialPosition``)
from the stations. The AP can also be moving with a constant velocity (``--velocity``) to simulate dynamic scenarios.
Other assumptions from the simulation are the log-distance propagation `loss model <https://www.nsnam.org/docs/models/html/propagation.html>`_,
A-MPDU frame aggregation, 5 Ghz frequency band, and single spatial stream.
  
  Changeable simulation parameters:
  
  * duration of the simulation (excluding the warmup stage) (s) ``--simulationTime``, default is 20 s,
  * duration of the warmup stage (s) - time for the simulator to enable all the mechanisms before the traffic generation begins ``--warmupTime``, default is 2 s,
  * time interval between successive measurements (s) ``--logEvery``, default is 1 s,
  * simulation seed ``--RngRun``,
  * aggregated traffic generator data rate (Mb/s) ``--dataRate``, default is 125 Mb/s,
  * channel width (MHz) ``--channelWidth``, default is 20 MHz,
  * shortest guard interval to use (ns) ``--minGI``, default is 3200 ns,
  * rate adaptation manager ``--wifiManager``, default is ``"ns3::RLibWifiManager"``, meaning that the manager is on the Reinforced-lib side,
  * relative path where the simulation output file will be saved in the CSV format ``--csvPath``, default to ``""``, meaning not saving at all,
  * name of the Wi-Fi manager in the CSV file ``wifiManagerName``, default to ``"RLib"``,
  * relative path where the PCAP file from the AP will be saved ``--pcapPath``, default is ``""``, meaning no pcap at all.


Reinforced-lib (python) end
---------------------------

The provided rate adaptation manager is implemented in the file ``$REINFORCED_LIB/examples/ns-3/main.py``. Here we specify the
communication with the ns-3 simulator by defining the environment's observation space and the action space, we create the ``RLib``
agent, we provide the agent-environment interaction loop which reacts to the incoming (aggregated) frames by responding with an appropriate MCS,
and cleans up the environment when the simulation is done. Below we include and explain the essential code snippets.

.. code-block:: python
    :linenos:
    :lineno-start: 6

    from py_interface import *

    from reinforced_lib import RLib
    from reinforced_lib.agents.mab import ThompsonSampling
    from reinforced_lib.exts.wifi import IEEE_802_11_ax_RA

In line 6 we include the ns3-ai structures which enables us to use the shared memory communication.
Next we import the ``RLib`` class which is the main interface of the library that merges the agent and the environment.
We chose the :ref:`Thompson sampling <Thompson Sampling>` agent to demonstrate the manager performance. The environment
will be of course :ref:`802.11ax <IEEE 802.11ax>`, so we import an appropriate extension.

.. code-block:: python
    :linenos:
    :lineno-start: 13

    class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('power', c_double),
        ('time', c_double),
        ('cw', c_uint32),
        ('n_failed', c_uint32),
        ('n_successful', c_uint32),
        ('n_wifi', c_uint32),
        ('station_id', c_uint32),
        ('mcs', c_uint8),
        ('type', c_uint8)
    ]


    class Act(Structure):
        _pack_ = 1
        _fields_ = [
            ('station_id', c_uint32),
            ('mcs', c_uint8)
        ]

Next we define the ns3-ai structures that describe the environment space and action space accordingly. The structures must
strictly reflect the ones defined in the 
`header file <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3/rlib-wifi-manager/model/rlib-wifi-manager.h>`_
``rlib-wifi-manager/model/rlib-wifi-manager.h`` because it is the interface of the shared memory data bridge between
python and C++. You can learn more about the data exchange model
`here <https://github.com/hust-diangroup/ns3-ai/tree/master/examples/a_plus_b>`_.


.. code-block:: python
    :linenos:
    :lineno-start: 73

    rl = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax_RA
    )

    exp = Experiment(mempool_key, mem_size, "rlib-sim", ns3_path)
    var = Ns3AIRL(memblock_key, Env, Act)

In line 73, we create an instance of RLib by supplying the Thompson sampling agent and the 802.11ax environment extension.
We define the ns3-ai experiment in line 78 by setting the memory key, the memory size, the name of the ns-3 scenario, and the path
to the ns3 root directory. In line 79, we create a handler to the shared memory interface by providing an arbitrary key and
the previously defined environment and action structures.


.. code-block:: python
    :linenos:
    :lineno-start: 81

    try:
        ns3_process = exp.run(ns3_args, show_output=True)

        while not var.isFinish():
            with var as data:
                if data is None:
                    break

                if data.env.type == 0:
                    data.act.station_id = rl.init(seed)

                elif data.env.type == 1:
                    observation = {
                        'time': data.env.time,
                        'n_successful': data.env.n_successful,
                        'n_failed': data.env.n_failed,
                        'n_wifi': data.env.n_wifi,
                        'power': data.env.power,
                        'cw': data.env.cw,
                        'mcs': data.env.mcs
                    }

                    data.act.station_id = data.env.station_id
                    data.act.mcs = rl.sample(data.env.station_id, **observation)

        ns3_process.wait()
    finally:
        del exp

The final step to make the example work is to define the agent-environment interaction loop. We loop while the ns3 simulation is running (line 84)
and if there is any data to be read (line 86). We differentiate the environment observation by a type attribute which
indicates whether it is an initialization frame or not. On initialization (line 89), we have to initialize our RL agent with
some seed. In the opposite case we translate the observation to a dictionary (lines 93-101) and override the action structure
with the received station ID (line 103) and the appropriate MCS selected by the RL agent (line 104). The last thing is to
clean up the shared memory environment when the simulation is finished (lines 106 and 108).


Example experiments
===================

We supply the ``$REINFORCED_LIB/examples/ns-3/main.py`` script with the CLI so that you can test the rate adaptation manager in different
scenarios. We reflect all the command line arguments listed in :ref:`ns3 scenario <rlib-sim>` ``rlib-sim\sim.cc``
with the ``--under_score`` style. There are only two additional arguments:

  * path to the ns3 root directory ``--ns3_path``, default is ``$HOME/ns-3-dev/``,
  * shared memory pool key - arbitrary integer large than 1000 ``--mempool_key``, default is 1234.

You can try running the following commands to test the Reinforced-lib rate adaptation manager in different example scenarios:

  a. Static scenario with 1 AP and 1 STA both positioned in the same place

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH"

  b. Static scenario with 1 AP and 1 STA both positioned in the same place, with a ``ra-results.csv`` output file and ``ra-experiment-0-0.pcap`` file saved in the ``$HOME\`` directory

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH" --csv_path="$HOME/ra-results.scv" --pcap_path="$HOME/ra-experiment"

  c. Static scenario with 1 AP and 16 stations at a 10 m distance

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH" --n_wifi=16 --initial_position=10

  d. Dynamic scenario with 1 AP and 1 STA starting at 0 m and moving away from AP with a velocity of 1 m/s

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH" --velocity=1


.. _gym_integration:

***************************
Gym environment integration
***************************


Our library supports defining RL environments in the OpenAI gym format. We want to show you how well our
agents are suited to work with the gym environments using an example of a simple recommender system.


Recommender system example
==========================

Suppose that we have some goods to sell but for each user we can present a single product at a time. We assume that
each user has some unknown to us preferences about our goods and we want to fit the presentation of the product to their
taste. The situation can be modeled as a `multi-armed bandit problem <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_
and we can use our agents (for example the :ref:`epsilon-greedy <Epsilon-greedy>` one) to satisfy it.


Environment definition
----------------------

We recommend to define the environment class in a separate python file. After the imports section, you should register your
new environment by assigning some id and a path of the class relative to the project root like this:

.. code-block:: python
    :linenos:
    :lineno-start: 7

    gym.envs.registration.register(
        id='RecommenderSystemEnv-v1',
        entry_point='examples.recommender_system.env:RecommenderSystemEnv'
    )

Then you define the environment class with an appropriate constructor, which provides the dictionary of user preferences, the observation
and action space.

.. code-block:: python
    :linenos:
    :lineno-start: 13

    class RecommenderSystemEnv(gym.Env):

        def __init__(self, preferences: Dict) -> None:

            self.action_space = gym.spaces.Discrete(len(preferences))
            self.observation_space = gym.spaces.Space()
            self._preferences = list(preferences.values())

Because we inherit from the `gym.Env` class, we must provide the `reset()` and the `step()` methods at least, which are also necessary
to make our recommender system environment work. The reset method is responsible only for seed setting. The step method
pulls the bandit's arm and returns the reward.

.. code-block:: python
    :linenos:
    :lineno-start: 27

    def reset(
            self,
            seed: int = None,
            options: Dict = None
    ) -> Tuple[gym.spaces.Space, Dict]:

        seed = seed if seed else np.random.randint(1000)
        super().reset(seed=seed)
        np.random.seed(seed)

        return None, {}
    
    def step(self, action: int) -> Tuple[gym.spaces.Dict, int, bool, bool, Dict]:

        reward = int(np.random.rand() < self._preferences[action])

        return None, reward, False, False, {}


Extension definition
--------------------

To fully benefit from the Reinforced-lib's functionalities we recommend to implement an extension which will improve the
communication between the agent and the environment, as described in the :ref:`Custom extensions <custom_extensions>`
section. The source code with the implemented extension to our simple recommender system can be found in our
`official repository <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/recommender_system/ext.py>`_.


Agent - environment interaction
-------------------------------

Once you have defined the environment (and optionally the extension), you can train the agent to act in it efficiently. As
usual, we begin with necessary imports:

.. code-block:: python
    :linenos:
    :lineno-start: 1

    from reinforced_lib import RLib
    from reinforced_lib.agents.mab import EGreedy
    from reinforced_lib.logs import PlotsLogger, SourceType
    from ext import RecommenderSystemExt

    import gymnasium as gym
    import env

We define a `run()` function that constructs the recommender system extension, creates, and resets the appropriate
environment with user preferences derived from the extension. We also create and initialize the `RLib` instance with the selected
agent, previously constructed extension and optionally some loggers to visualise the decision making process.

.. code-block:: python
    :linenos:
    :lineno-start: 10

    def run(episodes: int, seed: int) -> None:

        # Construct the extension
        ext = RecommenderSystemExt()

        # Create and reset the environment which will simulate user behavior
        env = gym.make("RecommenderSystemEnv-v1", preferences=ext.preferences)
        _ = env.reset(seed=seed)

        # Wrap everything under RLib object with designated agent
        rl = RLib(
            agent_type=EGreedy,
            agent_params={'e': 0.25},
            ext_type=RecommenderSystemExt,
            loggers_type=PlotsLogger,
            loggers_sources=[('action', SourceType.METRIC), ('cumulative', SourceType.METRIC)],
            loggers_params={'scatter': True}
        )
        rl.init(seed)

Finally we finish the `run()` function with a training loop that asks the agent to select an action, acts on the environment
and receives some reward. Beforehand, we select an arbitrary action from the action space and perform the first rewarded step.

.. code-block:: python
    :linenos:
    :lineno-start: 30

        # Loop through each episode and update prior knowledge
        act = env.action_space.sample()
        _, reward, *_ = env.step(act)

        for i in range(1, episodes):
            act = rl.sample(action=act, reward=reward, time=i)
            _, reward, *_ = env.step(act)

Evaluating the `run()` function, with some finite number of episodes and a seed, should result in two plots,
one representing the actions selected by the agent, and the second one representing the cumulative reward versus time.
