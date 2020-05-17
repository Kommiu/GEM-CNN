FROM pytorch/pytorch
RUN apt update && apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
		vim \
		sudo \
		graphviz \
		ffmpeg \
		curl \
#		libeigen3-dev \
#        libgmp-dev \
#        libgmpxx4ldbl \
#        libmpfr-dev \
#        libboost-dev \
#        libboost-thread-dev \
#        libtbb-dev \
#        python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy
RUN mkdir /home/kommiu

RUN groupadd  -r -g 1000 kommiu && \
	useradd -r -g 1000 -u 1000 kommiu -d /home/kommiu/

ENV HOME=/home/kommiu
ENV APP_HOME=/home/kommiu/app

RUN mkdir $APP_HOME
RUN chown -R kommiu:kommiu $HOME

#RUN git clone https://github.com/PyMesh/PyMesh.git &&\
#    cd PyMesh && \
#    git submodule update --init && \
#    pip install --no-cache-dir -r ./python/requirements.txt && \
#    ./setup.py build && \
#    ./setup.py install

RUN pip install --no-cache-dir https://github.com/PytorchLightning/pytorch-lightning/archive/master.zip --upgrade
RUN pip install --no-cache-dir \
	pandas \
	notebook\
	jupyter \
	jupyterlab\
	scikit-image \
	scikit-learn \
	scipy \
	tensorboard \
	pillow \
	dask \
	flask \
	ipywidgets \
	cython \
	numba \
	python-dotenv  \
	numpy-stl
RUN conda install -c conda-forge trimesh

WORKDIR /home/kommiu/app

USER kommiu
