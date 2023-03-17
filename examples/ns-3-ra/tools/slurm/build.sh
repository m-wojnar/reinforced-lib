#!/usr/bin/scl enable devtoolset-11 rh-python38 -- /bin/bash -l

NS3_DIR="${NS3_DIR:=$HOME/ns-3.38}"

cd $NS3_DIR
./ns3 configure -d optimized --disable-python
./ns3
