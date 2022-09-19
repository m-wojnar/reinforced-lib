.. _examples_page:

########
Examples
########


.. _ns3_connection:

*******************
Connetion with ns-3
*******************

We will demonstrate the cooperation of Reinforced-lib with external WiFi simulating software based on the example of
ML controlled rate adaptation manager. To simulate the WiFi environment, we will use the a popular, research oriented
network simulator - ns-3. To learn more about the simulator, we encourage to visit the
`official website <https://www.nsnam.org/>`_ or read the
`ns-3 tutorial <https://www.nsnam.org/docs/release/3.36/tutorial/html/index.html>`_.


Environment setup
=================

In order to perform experiments with pythonic Reinforced-lib and C++ written ns-3, you need to setup the environment in
following order:

  * Favourite C++ compiler (we assume that you already have one in your dev stack)
  * ns-3 (connection tested on ns-3.36 version)
  * ns3-ai


Installing ns-3
---------------

There are a few ways to install ns-3, all described on the `ns-3 wiki <https://www.nsnam.org/wiki/Installation>`_,
but we recomend to install the ns-3 by cloning their git dev repository:

.. code-block:: bash

    git clone https://gitlab.com/nsnam/ns-3-dev.git

Then change directory to newly created ``ns-3-dev/`` and build the ns-3:

.. code-block:: bash

    ./ns3 configure --enable-examples --build-profile=optimized
    ./ns3 build

Once you have built ns-3 (with examples enabled), it should be easy to run the sample programs with the following command,
such as:

.. code-block:: bash

    ./ns3 run wifi-simple-adhoc

If the installation process was successful, you should now have two new ``.pcap`` files in the ``ns-3-dev/`` directory, namely
``wifi-simple-adhoc-0-0.pcap`` and ``wifi-simple-adhoc-1-0.pcap``


Installing ns3-ai
-----------------

The ns3-ai module interconnects the ns-3 and Reinforced-lib (or other python-writen software) by transferring data through
the shared memory pool. The memory can be accessed by both sides thus making the connection. Read more on ns3-ai on the
`official repository <https://github.com/hust-diangroup/ns3-ai>`_. To install the module colone the ns3-ai repository to the
``ns-3-dev/contrib/`` directory and install the ``py_interface`` module with pip:

.. code-block:: bash

    cd $YOUR_NS3_PATH/contrib/
    git clone https://github.com/hust-diangroup/ns3-ai.git
    pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"

To have ns3-ai working, you need to rebuild the ns-3, but before doing so, we will transfer some neccessary files to
enable experimenting with the provided rate adaptation manager.

.. warning::

    The ns3-ai (as of 18.09.2022) is not compatible with the ns-3.36 or later. We have forked and modified the official
    ns3-ai repository to make it compatible with the 3.36 version. In order to install our compatible version run the
    following commands instead:

    .. code-block:: bash

        cd $YOUR_NS3_PATH/contrib/
        git clone https://github.com/m-wojnar/ns3-ai.git
        cd "$YOUR_NS3_PATH/contrib/ns3-ai/"
        git checkout ml4wifi
        pip install --user "$YOUR_NS3_PATH/contrib/ns3-ai/py_interface"


Transfering RA example files
----------------------------

In the ``$REINFORCED_LIB/examples/ns-3/`` there are two directories. The ``rlib-sim`` contains the
example scenario, which will be described in the :ref:`next section <rlib-sim>`. The ``rlib-wifi-manager`` directory
contains an ns-3 contrib module with the specification of a custom rate adaptation manager that comunicates with python
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


ns-3 (C++) end
--------------

We supply an example scenario ``rlib-sim\sim.cc`` to test the rate adaptation manager in 802.11ax environment. The scenario is highly customizable but the key points
are that there is one access point AP and a variable number (``--nWifi``) of stations STA; there is an uplink, saturated
comunication (from STAs to AP) and the AP is in clear line of sight from all the STAs; All the STAs are in the point of 0m
and the AP can be either in 0m as well or in some distance (``--initialPosition``) from the STAs. The AP can also be moving
with a constant velocity (``--velocity``) to simulate dynamic scenarios. Other assumptions from the simulation are the
log-distance propagation `loss model <https://www.nsnam.org/docs/models/html/propagation.html>`_,  AMPDU frames aggregation,
5 Ghz frequency band and single spatial stream.
  
  Changable simulation parameters:
  
  * Duration of the simulation; excluding warmup stage (s) ``--simulationTime``, default to 20 s
  * Duration of the warmup stage (s) - a time for the simulator to enable all the mechanisms before the traffic begins ``--warmupTime``, default to 2 s
  * Time interval between successive measurements (s) ``--logEvery``, default to 1 s
  * Simulation Seed ``--RngRun``
  
---------------

  * Aggregated traffic generators data rate (Mb/s) ``--dataRate``, default to 125 Mb/s
  * Channel width (MHz) ``--channelWidth``, default to 20 MHz
  * Shortest guard interval (ns) ``--minGI``, default to 3200 ns
  * Rate adaptation manager ``--wifiManager``, default to ``"ns3::RLibWifiManager"``, meaning that the manager is on the Reinforced-lib side
  
---------------

  * Relative path where the simulation output file will be saved in the CSV format ``--csvPath``, default to ``""``, meaning no save at all
  * Name of the Wi-Fi manager in CSV ``wifiManagerName``, default to ``"RLib"``
  * Relative path where the PCAP file from the AP will be saved ``--pcapPath``, default to ``""``, meaning no pcap at all


Reinforced-lib (python) end
---------------------------

The provided Rate Adaptation manager is implemented in the file ``$REINFORCED_LIB/examples/ns-3/main.py``. Here we specify the
comunication with the ns-3 simulator by defining the environment observation space and the action space, we create the ``RLib``
agent, we provide the agent-environment interaction loop which reacts to the incomming (aggregated) frames by responding with the appropriate MCS
and clean up the environment when the simulation is done. Below we include and explain the essential code snippets.

.. code-block:: python
    :linenos:
    :lineno-start: 6

    from py_interface import *

    from reinforced_lib import RLib
    from reinforced_lib.agents import ThompsonSampling
    from reinforced_lib.exts import IEEE_802_11_ax

In line 6 we include the ns3-ai structures which enables us the use of the shared memory comunication.
Next we import the ``RLib`` class which is the main interface of the library that merges the agent with the environment.
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

Next we define the ns3-ai structures that describes the environment space and acion space accordingly. The structures must
strictly reflect the ones defined in the 
`header file <https://github.com/m-wojnar/reinforced-lib/blob/main/examples/ns-3/rlib-wifi-manager/model/rlib-wifi-manager.h>`_
``rlib-wifi-manager/model/rlib-wifi-manager.h`` becouse it is the very interface of the shared memory data bridge between
python and C++. You can learn more about the data exchange model
`here <https://github.com/hust-diangroup/ns3-ai/tree/master/examples/a_plus_b>`_.


.. code-block:: python
    :linenos:
    :lineno-start: 73

    rl = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax
    )

    exp = Experiment(mempool_key, mem_size, "rlib-sim", ns3_path)
    var = Ns3AIRL(memblock_key, Env, Act)

In line 73 we create an instance of the RLib by supplying the Thompson sampling agent and 802.11ax environment extension.
We define the ns3-ai experiment in line 78 by setting the memory key, memory size, name of the ns3 scenario and the path
to the ns3 root directory. In line 79 we create a handler to the shared memory interface by providing an arbitral key and
previously defined environment and action structures.


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

The final step to make the example working is to define the agent-environment interaction loop. We loop while the ns3 simulation is running (line 84)
and there is any data to be read (line 86). We differentiated the environment observation by the type attribute which
indicates whether it is and initialization frame or not. On initialization (line 89), we have to init our RL agenet with
some seed. In the other case we translate the observation to a dictionary (lines 93-101) and override the action structure
with the received station ID (line 103) and appropriate MCS selected by the RL agent (line 104). The last thing to do, is to
clean up the shared memory environment when the simulation is finished (lines 106 and 108).


Example experiments
===================

We have supplied the ``$REINFORCED_LIB/examples/ns-3/main.py`` script with the CLI so that you can test the rate adaptation manager in different
scenarios. We reflected all the command lines arguments listed in :ref:`ns3 scenario <rlib-sim>` ``rlib-sim\sim.cc``
with the ``--under_score`` style. There are only two additional arguments:

  * Path to the ns3 root directory ``--ns3_path``, default to ``$HOME/ns-3-dev/``
  * Shared memory pool key - arbitrary integer large than 1000 ``--mempool_key``, default to 1234

You can try running the following commands to test the Reinforced-lib rate adaptation manager in example scenarios:

  a. Static scenario with 1 AP and 1 STA both positioned in the same place

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH"

  b. Static scenario with 1 AP and 1 STA both positioned in the same place, with a ``ra-results.csv`` output file and ``ra-experiment-0-0.pcap`` file saved in the ``$HOME\`` directory

    .. code-block:: bash
        
        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH" --csv_path="$HOME/ra-results.scv" --pcap_path="$HOME/ra-experiment"

  c. Static scenario with 1 AP and 16 STAs in a 10 m distance

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH" --n_wifi=16 --initial_position=10

  d. Dynamic scenario with 1 AP and 1 STA starting in 0m and moving away from AP with the velocity 1 m/s

    .. code-block:: bash

        python $REINFORCED_LIB/examples/ns-3/main.py --ns3_path="$YOUR_NS3_PATH" --velocity=1


.. _gym_integration:

***************************
Gym environment integration
***************************


TODO
====

