# docker build --no-cache  multitasking .
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 

RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            git \
            ssh \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build python=3.6.3 numpy pyyaml mkl&& \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/bin:$PATH

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

RUN conda install -c pytorch pytorch cuda90

RUN pip install tqdm
RUN pip install requests
RUN pip install git+https://github.com/pytorch/text.git
ADD ./README.md /README.md
ADD ./cove/ /cove/
ADD ./setup.py /setup.py
RUN python setup.py develop

ADD ./test/ /test/

CMD python /test/example.py
