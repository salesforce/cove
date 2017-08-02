# Context Vectors (CoVe)

This repo provides the best MT-LSTM from the paper [Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107).
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
