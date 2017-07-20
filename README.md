# Context Vectors (CoVe)

This repo provides the best MT-LSTM from the paper [Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://einstein.ai/research/static/images/layouts/research/cove/McCann2017LearnedIT.pdf).
For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors).

example.py uses [torchtext](https://github.com/pytorch/text/tree/master/torchtext) to load the [Stanford Natural Language Inference Corpus](https://nlp.stanford.edu/projects/snli/) and [GloVe](https://nlp.stanford.edu/projects/glove/).

It uses a [PyTorch](http://pytorch.org/) implementation of the MTLSTM class in mtlstm.py to load a pretrained encoder, 
which takes in sequences of vectors pretrained with GloVe and outputs CoVe.

## Running with Docker

We have included a Dockerfile that covers all dependencies.  
We typically use this code on a machine with a GPU, 
so we use `nvidia-docker`.

Once you have installed [Docker](https://www.docker.com/get-docker), 
pull the docker image with `docker pull bmccann/cove`.
Then you can use `nvidia-docker run -it -v /path/to/cove/:/cove cove` to start a docker container using that image.
Once the container is running, 
you can use `nvidia-docker ps` to find the `container_name` and
`nvidia-docker exec -it container_name bash -c "cd cove && python example.py"` to run example.py.

## Running without Docker

You will need to install PyTorch and then run `pip install -r requirements.txt`
Run the example with `python example.py`.


## References

When the paper is live on arxiv, we will update the link and reference below. If using this code, please cite:

   B. McCann, J. Bradbury, C. Xiong, R. Socher, [*Learned in Translation: Contextualized Word Vectors*](https://einstein.ai/research/static/images/layouts/research/cove/McCann2017LearnedIT.pdf)

```
@article{McCann2017LearnedIT,
  title={Learned in Translation: Contextualized Word Vectors},
  author={Bryan McCann and James Bradbury and Caiming Xiong and Richard Socher},
  journal={arXiv preprint arXiv:?},
  year={2017}
}
```

Contact: [bmccann@salesforce.com](mailto:bmccann@salesforce.com)
