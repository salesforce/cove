# Contextualized Word Vectors (CoVe)

This repo provides the best, pretrained MT-LSTM from the paper [Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf).
For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors).

This repository uses a [PyTorch](http://pytorch.org/) implementation of the MTLSTM class in mtlstm.py to load a pretrained encoder, 
which takes in sequences of vectors pretrained with GloVe and outputs CoVe.

## Need CoVe in Tensorflow?

A Keras/TensorFlow implementation of the MT-LSTM/CoVe can be found at https://github.com/rgsachin/CoVe.

## Unknown Words

Out of vocabulary words for CoVe are also out of vocabulary for GloVe, which should be rare for most use cases. During training the CoVe encoder would have received a zero vector for any words that were not in GloVe, and it used zero vectors for unknown words in our classification and question answering experiments, so that is recommended.

You could also try initializing unknown inputs to something close to GloVe vectors instead, but we have no experiments suggesting that this would work better than zero vectors. If you wanted to try this, GloVe vectors follow (very roughly) a Gaussian with mean 0 and standard deviation 0.4. You could initialize by randomly drawing from that distrubtion, but you would probably want to train those embeddings while keeping the CoVe encoder (MTLSTM) and GloVe fixed.

## Example Usage

The following example can be found in `test/example.py`. It demonstrates a few different variations of how to use the pretrained MTLSTM class that generates contextualized word vectors (CoVe) programmatically.

### Running with Docker

Install [Docker](https://www.docker.com/get-docker).
Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you would like to use with with a GPU.

```bash
docker pull bmccann/cove   # pull the docker image
# On CPU
docker run -it --rm -v `pwd`/.embeddings:/.embeddings/ -v `pwd`/.data/:/.data/ bmccann/cove bash -c "python /test/example.py --device -1" 
# On GPU
nvidia-docker run -it --rm -v `pwd`/.embeddings:/.embeddings/ -v `pwd`/.data/:/.data/ bmccann/cove bash -c "python /test/example.py"
```

### Running without Docker

Install [PyTorch](http://pytorch.org/).

```bash 
git clone https://github.com/salesforce/cove.git # use ssh: git@github.com:salesforce/cove.git
cd cove
pip install -r requirements.txt
python setup.py develop
# On CPU
python test/example.py --device -1
# On GPU
python test/example.py
```
## Re-training CoVe

There is also the third option if you are operating in an entirely different context -- retrain the bidirectional LSTM using trained embeddings. If you are mostly encoding a non-English language, that might be the best option. Check out the paper for details; code for this is included in the directory OpenNMT-py, which was forked from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) a long while back and includes changes we made to the repo internally.

## References

If using this code, please cite:

   B. McCann, J. Bradbury, C. Xiong, R. Socher, [*Learned in Translation: Contextualized Word Vectors*](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf)

```
@inproceedings{mccann2017learned,
  title={Learned in translation: Contextualized word vectors},
  author={McCann, Bryan and Bradbury, James and Xiong, Caiming and Socher, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6297--6308},
  year={2017}
}
```

Contact: [bmccann@salesforce.com](mailto:bmccann@salesforce.com)
