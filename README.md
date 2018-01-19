# Context Vectors (CoVe)

This repo provides the best MT-LSTM from the paper [Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107).
For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors).

example.py uses [torchtext](https://github.com/pytorch/text/tree/master/torchtext) to load the [Stanford Natural Language Inference Corpus](https://nlp.stanford.edu/projects/snli/) and [GloVe](https://nlp.stanford.edu/projects/glove/).

It uses a [PyTorch](http://pytorch.org/) implementation of the MTLSTM class in mtlstm.py to load a pretrained encoder, 
which takes in sequences of vectors pretrained with GloVe and outputs CoVe.

A Keras/TensorFlow implementation of the MT-LSTM/CoVe can be found at https://github.com/rgsachin/CoVe.

## Running with Docker

Install [Docker](https://www.docker.com/get-docker).
Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you would like to use with with a GPU.

```bash
docker pull bmccann/cove   # pull the docker image
docker run -it cove        # start a docker container
python /cove/test/example.py
```

## Running without Docker

Install [PyTorch](http://pytorch.org/).

```bash 
git clone https://github.com/salesforce/cove.git # use ssh: git@github.com:salesforce/cove.git
cd cove
pip install -r requirements.txt
python setup.py develop
python test/example.py
```


## References

If using this code, please cite:

   B. McCann, J. Bradbury, C. Xiong, R. Socher, [*Learned in Translation: Contextualized Word Vectors*](https://arxiv.org/abs/1708.00107)

```
@article{McCann2017LearnedIT,
  title={Learned in Translation: Contextualized Word Vectors},
  author={Bryan McCann and James Bradbury and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:1708.00107},
  year={2017}
}
```

Contact: [bmccann@salesforce.com](mailto:bmccann@salesforce.com)
