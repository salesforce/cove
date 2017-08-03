# Start from base python/pytorch container
FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         locales \
         curl \
         ca-certificates && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build python=3.6 numpy pyyaml mkl&& \
     /opt/conda/bin/conda clean -ya 

ENV PATH /opt/conda/bin:$PATH
RUN conda install -c soumith pytorch=0.1.12 cuda80

# Default to utf-8 encodings in python
# Can verify in container with:
# python -c 'import locale; print(locale.getpreferredencoding(False))'
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

ADD ./ /cove/
RUN cd cove && pip install -r requirements.txt && python setup.py develop
