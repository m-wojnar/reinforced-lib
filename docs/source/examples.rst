.. _examples_page:

########
Examples
########


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

    ./ns3 configure --enable-examples
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
`official repository <https://github.com/hust-diangroup/ns3-ai>`_. To install the module colone the ns-3 repository to the
``ns-3-dev/contrib/`` directory:

.. code-block:: bash

    cd $YOUR_NS3_PATH/contrib/
    git clone https://github.com/hust-diangroup/ns3-ai.git

To have ns3-ai working, you need to rebuild the ns-3, but before doing so, we will transfer some neccessary files to
enable experimenting with the provided rate adaptation manager.


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

We supply an example scenario ``rlib-sim\sim.cc`` to test the rate adaptation manager. The scenario is quite customizable but the main idea
is that there is one access point AP and a variable number (``--nWifi``) of stations STA. There is an uplink, saturated
comunication (from AP to STAs) and the AP is in clear line of sight from all the STAs. All the STAs are in the point of 0m
and the AP can be either in 0m as well or in some distance (``--initialPosition``) from the STAs. The AP can also be moving
with a constant velocity (``--velocity``) to simulate dynamic scenarios. Other assumptions in the simulation are the
log-distance propagation loss `model <https://www.nsnam.org/docs/models/html/propagation.html>`_ and AMPDU frames aggregation.
  
  Changable simulatin parameters:
  
  * Duration of the simulation; excluding warmup stage (s) ``--simulationTime``, default to 20s
  * Duration of the warmup stage (s) - a time for the simulator to enable all the mechanisms before the traffic begins ``--warmupTime``, default to 2s
  * Time interval between successive measurements (s) ``--logEvery``, default to 1s
  * Simulation Seed ``--RngRun``
  
---------------

  * Aggregate traffic generators data rate (Mb/s) ``--dataRate``, default to 125Mb/s
  * Channel width (MHz) ``--channelWidth``, default to 20MHz
  * Shortest guard interval (ns) ``--minGI``, default to 3200ns
  * Rate adaptation manager ``--wifiManager``, default to ``"ns3::RLibWifiManager"``, meaning that the manager is on the Reinforced-lib side
  
---------------

  * Relative path where the simulation output file will be saved in the CSV format ``--csvPath``, default to ``""``, meaning no save at all
  * Name of the Wi-Fi manager in CSV ``wifiManagerName``, default to ``"RLib"``
  * Relative path where the PCAP file from the AP will be saved ``--pcapPath``, default to ``""``, meaning no pcap at all


Reinforced-lib (python) end
---------------------------

The provided Rate Adaptation manager is implemented in the file ``$REINFORCED_LIB/examples/ns-3/main.py``. Here we specify the
comunication with the ns-3 simulator by defining the environment observation space and the action space, we create the ``RLib``
agent, we provide the listening loop which reacts to the incomming (aggregated) frames by responding with the appropriate MCS
and clean up the environment when the simulation is done. Below we include and explain the essential code snippets.

.. code-block:: python
    :linenos:

    from py_interface import *

    from reinforced_lib import RLib
    from reinforced_lib.agents import ThompsonSampling
    from reinforced_lib.exts import IEEE_802_11_ax

In the first line we include the ns3-ai structures which enables us the use of the shared memory comunication.
Next we import the ``RLib`` class which is the main interface of the library that merges the agent with the environment.
We chose the ThompsonSampling agent to demonstrate the manager performance. The environment will be of course 802.11ax,
so we use the appropriate extension.

.. code-block:: python
    :linenos:

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



.. code-block:: python
    :linenos:

    rl = RLib(
        agent_type=ThompsonSampling,
        ext_type=IEEE_802_11_ax
    )

    exp = Experiment(mempool_key, mem_size, scenario, ns3_path)
    var = Ns3AIRL(memblock_key, Env, Act)


.. code-block:: python
    :linenos:

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

TODO
----
  * Describe code
  * Describe shortly main.py arguments
  * provide examples commands to run



***************************
Gym environment integration
***************************
