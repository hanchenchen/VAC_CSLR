FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime 
RUN mkdir /install
COPY ctcdecode /install/ctcdecode
## if 429, download third_party/* separately
COPY kaldi /install/kaldi
WORKDIR /install/ctcdecode
RUN apt-get update && apt-get install g++ apt-utils -y && pip install .
WORKDIR /install/kaldi/tools
RUN apt-get install make automake autoconf patch unzip wget git sox gfortran libtool subversion python2.7 -y && extras/check_dependencies.sh && make
# docker build -t ctc:v1 .
WORKDIR /install/kaldi/src
RUN ../tools/extras/install_mkl.sh
RUN ./configure --shared 
# --mkl-root=/usr --mkl-libdir=/usr/local/lib
RUN make depend -j 8
RUN make -j 8
RUN pip install opencv-python opencv-contrib-python-headless pandas matplotlib scipy -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com