# README

This repository includes the data and tools used in the paper [Fine-grained Topic Detection and Tracking on Twitter](https://www.scitepress.org/PublicationsDetail.aspx?ID=o+Iys1RHmPU=&t=1).
This README.md file describes how you can use the repository to experiment with the data yourself.

## Configuration

You can use the configuration file at `config/conf.py` to control the logging level.
This logging level is used while detecting participants, and can be restricted to reduce verbosity.

## Data

This repository contains the data used in the evaluation of _Fine-grained Topic Detection and Tracking on Twitter_.
Due to [Twitter's Developer Policy](https://developer.twitter.com/en/developer-terms/policy), it is not possible for us to share the original tweets.
Instead, we are only allowed to share the tweet IDs.

The tweet IDs are separated in two folders: `data/understanding` and `data/event`.
These two folders represent the understanding and event periods, described in the paper.
Each folder contains an `ids` folder, which contains the tweet IDs of each corpus, and an `idmeta` folder, which contains the metadata about each ID file.

An additional file—`data/sample.json`—includes tweet IDs collected using Twitter's Streaming API.
This sample of tweets was used in the paper to construct the TF-ICF table.
Instead of downloading it anew, you can use the `data/idf.json` file instead¸ which stores the TF-ICF scheme.
**The TF-IDF scheme is only used if you do not provide an understanding period tweet corpus.**
**It is recommended that you collect an understanding period tweet corpus for best results.**

If you prefer to download a new sample and use it to construct the TF-ICF scheme, download the original tweets and use the `idf` tool:

    ./tools/idf.py --file data/sample.json \
    --output data/idf.json \
    --stem \
    --remove-unicode-entities \
    --normalize-words

To learn more about this tool and the options it accepts, use `./tools/idf.py --help`.

## Generating timelines

### Baseline

To generate event timelines, use the `consume` tool available in the `tools` directory after downloading a tweet corpus.
The [Zhao et al. (2011)](https://arxiv.org/abs/1106.4300) baselines can be generated using the following snippets.
In all cases, the tools expect the file to contain tweets: one tweet per line.
Therefore before running the following, you need to download the tweets anew from the provided IDs.

    ./tools/consume.py \
    --event data/event/CRYCHE.json \
    --output results/CRYCHE-Zhao.json \
    --consumer ZhaoConsumer \
    --skip 10 --periodicity 1 --post-rate 1.7

The following parameters are passed on to the consumer:

- `--event`: The event tweet corpus, with one JSON-encoded tweet per line
- `--output`: The generated timeline, stored in a JSON file, which includes the tweets published when the algorithm detected a topic
- `--consumer`: The algorithm to use, in this case [Zhao et al. (2011)'s](https://arxiv.org/abs/1106.4300)
- `--skip`: The number of minutes to skip from the beginning of the event
- `--periodicity`: How often to check whether a new topic has occured (in seconds)
-  `--post-rate`: The increase in tweeting volume for the algorithm to detect a topic (a ratio)

### ELD

You can create timelines using ELD similarly.
ELD uses different parameters and expects datasets for an understanding period and an event period.

    ./tools/consume.py \
    --understanding data/understanding/CRYCHE.json \
    --event data/event/CRYCHE.json \
    --consumer ELDConsumer \
    --min-size 3 --scheme data/idf.json \
    --freeze-period 20 --max-intra-similarity 0.85 --speed 0.5

The following parameters are passed on to the consumer:

- `--event`: The understanding tweet corpus, collected shortly before the event, with one JSON-encoded tweet per line
- `--event`: The event tweet corpus, with one JSON-encoded tweet per line
- `--output`: The generated timeline, stored in a JSON file, which includes the tweets in topical clusters and the topical keywords
- `--consumer`: The algorithm to use, in this case ELD
- `--min-size`: The minimum acceptable size of a cluster to be considered a potential topic
- `--scheme`: The TF-IDF term-weighting scheme to use **if you do not have an understanding corpus (which is not recommended)**
- `--freeze-period`: The number of seconds after which an inactive cluster is frozen and no longer considered
- `--max-intra-similarity`: The maximum similarity between tweets and their cluster centroid; if the similarity exceeds this threshold, the cluster is discarded for being too singular
- `--speed`: The speed at which to feed tweets to the consumer (a value of 0.5 means the event is simulated at half the speed); necessary in large events which overwhelm ELD

More details about the parameters available in the [original paper](https://www.scitepress.org/PublicationsDetail.aspx?ID=o+Iys1RHmPU=&t=1).
