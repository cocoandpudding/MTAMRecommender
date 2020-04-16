ARG IMAGE_NAME
FROM ${IMAGE_NAME}:10.0-devel-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 7.6.0.64
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
#更换了apt的源
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


#给出相关的操作 更python与pip
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y software-properties-common \
&&  apt-get install -y apt-file \
&&  apt-file update \
&&  apt-get update && apt-get -y upgrade\
&&  apt --fix-broken install python-pycurl -y python-apt
#&&  apt-get install -y python-software-properties

RUN apt-get update \
#&&  add-apt-repository ppa:jonathonf/python-3.6 \
#&&  apt-get update \
&&  apt-get install -y --no-install-recommends python3.6 \
&&  apt-get install -y --no-install-recommends python3-pip \
&&  pip3 install --upgrade pip

#&&  hash -d pip
    
    
    
    
 
#安装Python的相关库,与cuda10.1相匹配
RUN apt-get install -y --no-install-recommends python3-setuptools \
&&  pip3 install torch torchvision -i https://mirrors.aliyun.com/pypi/simple


RUN pip3 install numpy -i  https://mirrors.aliyun.com/pypi/simple \
&&pip3 install pandas -i  https://mirrors.aliyun.com/pypi/simple \
&&pip3 install scikit-learn -i  https://mirrors.aliyun.com/pypi/simple \
&&pip3 install tensorflow-gpu==1.14.0

RUN apt-get install -y --no-install-recommends python3-dev \
&&pip3 install xgboost  -i https://mirrors.aliyun.com/pypi/simple  \
&&pip3 install requests \
&&pip3 install nni \
&&pip3 install keras -i  https://mirrors.aliyun.com/pypi/simple \
&&pip3 install pymysql\
&&pip3 install jieba \
&&pip3 install gevent \
&&pip3 install flask \
&&pip3 install configparser
