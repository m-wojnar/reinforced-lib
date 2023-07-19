.. _examples_page:

########
Examples
########


.. _ns3_connection:

********************
Connection with ns-3
********************

We will demonstrate the cooperation of Reinforced-lib with an external WiFi simulation software based on an example of
an ML-controlled rate adaptation (RA) manager. To simulate the WiFi environment, we will use the popular, research oriented
network simulator -- ns-3. To learn more about the simulator, we encourage to visit the
`official website <https://www.nsnam.org/>`_ or read the
`ns-3 tutorial <https://www.nsnam.org/docs/release/3.36/tutorial/html/index.html>`_.


Environment setup
=================

To perform experiments with Python-based Reinforced-lib and C++-based ns-3, you need to setup an environment which
consists of the following:

  * favourite C++ compiler (we assume that you already have one in your dev stack),
  * ns-3 (connection tested on the ns-3.37 version),
  * ns3-ai (`GitHub repository <https://github.com/hust-diangroup/ns3-ai/>`_).

Since the ns-3 requires the compilation, we will install all the required modules, transfer ns-3 files required for the
communication with Reinforced-lib, and copile everything once at the very end.


Installing ns-3
---------------

There are a few ways to install ns-3, all described in the `ns-3 wiki <https://www.nsnam.org/wiki/Installation>`_,
but we recommend to install ns-3 by cloning the git dev repository:

.. code-block:: bash

    git clone https://gitlab.com/nsnam/ns-3-dev.git

We recommend setting the simulator to the 3.37 version, since we do not guarantee the compatibility with other versions.
To set the ns-3 to the 3.37:

.. code-block:: bash

    cd ns-3-dev     # this directory will be referenced as YOUR_NS3_PATH since now on
    git reset --hard 4407a9528eac81476546a50597cc6e016a428f43


Installing ns3-ai
-----------------

The ns3-ai module interconnects ns-3 and Reinforced-lib (or any other python-writen software) by transferring data through
the shared memory pool. The memory is accessed by both sides thus making the connection. You can read more about the ns3-ai on the
`ns3-ai official repository <https://github.com/hust-diangroup/ns3-ai>`_. Unfortunately, ns3-ai (as of 18.07.2023) is not compatible with the ns-3.36 or later. We have forked and modified the official ns3-ai repository to make it compatible with the 3.37 version. To install the compatible, forked version run the following commands

.. code-block:: bash

    cd $YOUR_NS3_PATH/contrib/
    git clone --single-branch --branch ml4wifi https://github.com/m-wojnar/ns3-ai.git
    pip install "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"


Transferring ns3 files
----------------------

In ``$REINFORCED_LIB/examples/ns-3-ra/`` there are two directories. The ``scratch`` contains an
example RA scenario, which will be described in the :ref:`next section <rlib-sim>`. The ``contrib`` directory
contains a ``rlib-wifi-manager`` module with the specification of a custom rate adaptation manager that communicates with python
with the use of ns3-ai. You need to transfer both of these directories in the appropriate locations by running the
following commands:

.. code-block:: bash

    cp $REINFORCED_LIB/examples/ns-3-ra/scratch/* $YOUR_NS3_PATH/scratch/
    cp -r $REINFORCED_LIB/examples/ns-3-ra/contrib/rlib-wifi-manager $YOUR_NS3_PATH/contrib/

.. note::

    To learn more about adding contrib modules to ns-3, visit
    the `ns-3 manual <https://www.nsnam.org/docs/manual/html/new-modules.html>`_.


Compiling ns3
-------------

To have the simulator working and fully integrated with the Reinforced-lib, we need to compile it. We do this from the ``YOUR_NS3_PATH`` in two steps, by first configuring the compilation and than by building ns-3:

.. code-block:: bash

    cd $YOUR_NS3_PATH
    ./ns3 configure --build-profile=optimized --enable-examples --enable-tests
    ./ns3 build

Once you have built ns-3, you can test the ns-3 and Reinforced-lib integration by executing the script that runs an example
rate adaptation scenario controlled by the UCB agent.

.. code-block:: bash

    cd $REINFORCED_LIB
    ./test/test_ns3_integration.sh

On success, in your home directory, there should be a ``rlib-ns3-integration-test.csv`` file generated filled with some data.

.. _rlib-sim:

Simulation scenario
===================


ns-3 (C++) part
---------------

In ``rscratch`` directory we supply an example scenario ``rlib-sim.cc`` to test the rate adaptation manager in the 802.11ax
environment. The scenario is highly customizable but the key points
are that there is one access point (AP) and a variable number (``--nWifi``) of stations (STA); there is an uplink, saturated
communication (from stations to AP) and the AP is in line of sight with all the stations; all the stations are at the point of
:math:`(0, 0)~m` and the AP can either be at :math:`(0, 0)~m` as well or in some distance (``--initialPosition``)
from the stations. The AP can also be moving with a constant velocity (``--velocity``) to simulate dynamic scenarios.
Other assumptions from the simulation are the A-MPDU frame aggregation, 5 Ghz frequency band, and single spatial stream.

By typing ``$YOUR_NS3_PATH/build/scratch/ns3.37-ra-sim-optimized --help`` you can go over the simulation parameters and
learn what is the function of each.

.. code-block:: bash

    ./build/scratch/ns3.37-ra-sim-optimized --help
    [Program Options] [General Arguments]

    Program Options:
        --area:             Size of the square in which stations are wandering (m) [RWPM mobility type] [40]
        --channelWidth:     Channel width (MHz) [20]
        --csvPath:          Save an output file in the CSV format
        --dataRate:         Aggregate traffic generators data rate (Mb/s) [125]
        --deltaPower:       Power change (dBm) [0]
        --initialPosition:  Initial position of the AP on X axis (m) [Distance mobility type] [0]
        --intervalPower:    Interval between power change (s) [4]
        --logEvery:         Time interval between successive measurements (s) [1]
        --lossModel:        Propagation loss model to use [LogDistance, Nakagami] [LogDistance]
        --minGI:            Shortest guard interval (ns) [3200]
        --mobilityModel:    Mobility model [Distance, RWPM] [Distance]
        --nodeSpeed:        Maximum station speed (m/s) [RWPM mobility type] [1.4]
        --nodePause:        Maximum time station waits in newly selected position (s) [RWPM mobility type] [20]
        --nWifi:            Number of transmitting stations [1]
        --pcapPath:         Save a PCAP file from the AP
        --simulationTime:   Duration of the simulation excluding warmup stage (s) [20]
        --velocity:         Velocity of the AP on X axis (m/s) [Distance mobility type] [0]
        --warmupTime:       Duration of the warmup stage (s) [2]
        --wifiManager:      Rate adaptation manager [ns3::RLibWifiManager]
        --wifiManagerName:  Name of the Wi-Fi manager in CSV

    General Arguments:
        --PrintGlobals:              Print the list of globals.
        --PrintGroups:               Print the list of groups.
        --PrintGroup=[group]:        Print all TypeIds of group.
        --PrintTypeIds:              Print all TypeIds.
        --PrintAttributes=[typeid]:  Print all attributes of typeid.
        --PrintVersion:              Print the ns-3 version.
        --PrintHelp:                 Print this help message.


Reinforced-lib (python) end
---------------------------

The provided rate adaptation manager is implemented in the file ``$REINFORCED_LIB/examples/ns-3-ra/main.py``. Here we specify the
communication with the ns-3 simulator by defining the environment's observation space and the action space, we create the ``RLib``
agent, we provide the agent-environment interaction loop which reacts to the incoming (aggregated) frames by responding with an appropriate MCS,
and cleans up the environment when the simulation is done. Below we include and explain the essential fragments from the ``main.py`` script.

.. code-block:: python
    :linenos:
    :lineno-start: 4

    from ext import IEEE_802_11_ax_RA
    from particle_filter import ParticleFilter
    from py_interface import *   # Import the ns3-ai structures

    from reinforced_lib import RLib
    from reinforced_lib.agents.mab import *

We import the RA extension, agents and the RLib module. Line 6 is responsible for importing the structures from the ns3-ai
library.

.. code-block:: python
    :linenos:
    :lineno-start: 12

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
`header file <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3-ra/contrib/rlib-wifi-manager/model/rlib-wifi-manager.h>`_
``contrib/rlib-wifi-manager/model/rlib-wifi-manager.h`` because it is the interface of the shared memory data bridge between
python and C++. You can learn more about the data exchange model
`here <https://github.com/hust-diangroup/ns3-ai/tree/master/examples/a_plus_b>`_.


.. code-block:: python
    :linenos:
    :lineno-start: 73

    rl = RLib(
        agent_type=agent_type,
        agent_params=agent_params,
        ext_type=IEEE_802_11_ax_RA
    )

    exp = Experiment(mempool_key, memory_size, 'ra-sim', ns3_path)
    var = Ns3AIRL(memblock_key, Env, Act)

In line 73, we create an instance of RLib by supplying the appropriate, parametrized agent and the 802.11ax environment extension.
We define the ns3-ai experiment in line 79 by setting the memory key, the memory size, the name of the ns-3 scenario, and the path
to the ns3 root directory. In line 80, we create a handler to the shared memory interface by providing an arbitrary key and
the previously defined environment and action structures.


.. code-block:: python
    :linenos:
    :lineno-start: 82

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
                        'cw': data.env.cw
                    }

                    data.act.station_id = data.env.station_id
                    data.act.mcs = rl.sample(agent_id=data.env.station_id, **observation)

        ns3_process.wait()
    finally:
        del exp

The final step to make the example work is to define the agent-environment interaction loop. We loop while the ns3 simulation is running (line 85)
and if there is any data to be read (line 86). We differentiate the environment observation by a type attribute which
indicates whether it is an initialization frame or not. On initialization (line 90), we have to initialize our RL agent with
some seed. In the opposite case we translate the observation to a dictionary (lines 94-102) and override the action structure
with the received station ID (line 104) and the appropriate MCS selected by the RL agent (line 105). The last thing is to
clean up the shared memory environment when the simulation is finished (lines 107 and 107).


Example experiments
===================

We supply the ``$REINFORCED_LIB/examples/ns-3-ra/main.py`` script with the CLI so that you can test the rate adaptation manager in different
scenarios. We reflect all the command line arguments listed in :ref:`ns3 scenario <rlib-sim>` ``scratch/ra-sim.cc`` with four additional arguments:

  * ``--agent`` -- the type of RL agent responsible for the RA, a required argument,
  * ``--mempoolKey`` -- shared memory pool key, which is an arbitrary integer, greater than 1000, default is 1234.
  * ``--ns3Path`` -- path to the ns3 root directory, a required argument,

You can try running the following commands to test the Reinforced-lib rate adaptation manager in different example scenarios:

  a. Static scenario with 1 AP and 1 STA both positioned in the same place, RA handled by the *UCB* agent

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="UCB" --ns3Path="$YOUR_NS3_PATH"

  b. Static scenario with 1 AP and 1 STA both positioned in the same place, RA handled by the *UCB* agent. Output
  saved to the ``$HOME/ra-results.csv`` file and ``.pcap`` saved to the ``$HOME/ra-experiment-0-0.pcap``.

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="UCB" --ns3Path="$YOUR_NS3_PATH" --csvPath="$HOME/ra-results.scv" --pcapPath="$HOME/ra-experiment"

  c. Static scenario with 1 AP and 16 stations at a 10 m distance, RA handled by the *ThompsonSampling* agent.

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="ThompsonSampling" --ns3_path="$YOUR_NS3_PATH" --nWifi=16 --initialPosition=10

  d. Dynamic scenario with 1 AP and 1 STA starting at 0 m and moving away from AP with a velocity of 1 m/s, RA handled by the *ParticleFilter* agent.

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3-ra/main.py --agent="ParticleFilter" --ns3Path="$YOUR_NS3_PATH" --velocity=1


.. _gym_integration:

*********************************
Gymnasium environment integration
*********************************


Our library supports defining RL environments in the `Gymnasium <https://gymnasium.farama.org/>`_ (former OpenAI Gym)
format. We want to show you how well our agents are suited to work with the Gymnasium environments using an example
of a simple recommender system.


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
            logger_types=PlotsLogger,
            logger_sources=[('action', SourceType.METRIC), ('cumulative', SourceType.METRIC)],
            logger_params={'scatter': True}
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
