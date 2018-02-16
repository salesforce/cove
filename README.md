# Context Vectors (CoVe)

This repo provides the best MT-LSTM from the paper [Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107).
For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors).

example.py uses [torchtext](https://github.com/pytorch/text/tree/master/torchtext) to load the [Stanford Natural Language Inference Corpus](https://nlp.stanford.edu/projects/snli/) and [GloVe](https://nlp.stanford.edu/projects/glove/).

It uses a [PyTorch](http://pytorch.org/) implementation of the MTLSTM class in mtlstm.py to load a pretrained encoder, 
which takes in sequences of vectors pretrained with GloVe and outputs CoVe.

A Keras/TensorFlow implementation of the MT-LSTM/CoVe can be found at https://github.com/rgsachin/CoVe.

## Unknown Words

Out of vocabulary words for CoVe are also out of vocabulary for GloVe, which should be rare for most use cases. During training the CoVe encoder would have received a zero vector for any words that were not in GloVe, and it used zero vectors for unknown words in our classification and question answering experiments, so that is recommended.

You could also try initializing unknown inputs to something close to GloVe vectors instead, but we have no experiments suggesting that this would work better than zero vectors. If you wanted to try this, GloVe vectors follow (very roughly) a Gaussian with mean 0 and standard deviation 0.4. You could initialize by randomly drawing from that distrubtion, but you would probably want to train those embeddings while keeping the CoVe encoder (MTLSTM) and GloVe fixed.

There is also the third option if you are operating in an entirely different context -- retrain the bidirectional LSTM using trained embeddings. If you are mostly encoding a non-English language, that might be the best option. Check out the paper for details.

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
