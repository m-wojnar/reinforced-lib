FROM ubuntu:22.04

WORKDIR /home
RUN apt update
RUN apt install -y python3.10 python3-pip g++-11 cmake git vim
RUN git clone https://github.com/m-wojnar/reinforced-lib.git \
    && cd reinforced-lib \
    && pip install -U pip wheel setuptools \
    && pip install -e . 

WORKDIR /home
RUN git clone https://gitlab.com/nsnam/ns-3-dev.git \
    && cd ns-3-dev \
    && git reset --hard 4407a9528eac81476546a50597cc6e016a428f43

WORKDIR /home/ns-3-dev/contrib
RUN git clone https://github.com/hust-diangroup/ns3-ai.git \
    && cd ns3-ai \
    && git reset --hard 86453e840c6e5df849d8c4e9c7f88eade637798c \
    && cd .. \
    && pip install "ns3-ai/py_interface"

WORKDIR /home/reinforced-lib/examples/ns-3-ra
RUN cp scratch/* /home/ns-3-dev/scratch \
    && cp -r contrib/rlib-wifi-manager /home/ns-3-dev/contrib

WORKDIR /home/ns-3-dev
RUN ./ns3 configure --build-profile=optimized --disable-examples --disable-tests \
    && ./ns3 build

WORKDIR /home
