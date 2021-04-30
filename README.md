# README

This repository includes the data and tools used in the paper _Fine-grained Topic Detection and Tracking on Twitter_.
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
Each folder contains an `ids` folder, which contains the tweet IDs of each corpus.

An additional file—`data/ids/sample.json`—includes tweet IDs collected using Twitter's Streaming API.
This sample of tweets was used in the paper to construct the TF-ICF table.
Instead of downloading it anew, you can use the `data/idf.json` file instead¸ which stores the TF-ICF scheme.

If you prefer to download a new sample and use it to construct the TF-ICF scheme, you can use the `idf` tool:

    ./tools/idf.py --file ~/DATA/evaluation/apd/data/sample.json \
    --output results/tfidf.json \
    --stem \
    --remove-unicode-entities \
    --normalize-words

To learn more about this tool and the options it accepts, use `./tools/idf.py --help`.
