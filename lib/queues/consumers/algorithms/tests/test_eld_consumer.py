"""
Test the functionality of the ELD consumer.
"""

import asyncio
import json
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from queues import Queue
from queues.consumers.algorithms import ELDConsumer
from nlp.document import Document
from nlp.weighting import TF
from vsm import vector_math
from vsm.clustering import Cluster
import twitter

class TestELDConsumer(unittest.TestCase):
    """
    Test the implementation of the ELD consumer.
    """

    def async_test(f):
        def wrapper(*args, **kwargs):
            coro = asyncio.coroutine(f)
            future = coro(*args, **kwargs)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(future)
        return wrapper

    def test_init_name(self):
        """
        Test that the ELD consumer passes on the name to the base class.
        """

        name = 'Test Consumer'
        consumer = ELDConsumer(Queue(), 10, name=name)
        self.assertEqual(name, str(consumer))

    def test_create_consumer(self):
        """
        Test that when creating a consumer, all the parameters are saved correctly.
        """

        queue = Queue()
        consumer = ELDConsumer(queue, 60, scheme=TF())
        self.assertEqual(queue, consumer.queue)
        self.assertEqual(0, consumer.queue.length())
        self.assertEqual(60, consumer.time_window)
        self.assertEqual(TF, type(consumer.scheme))

    def test_create_consumer_buffer_empty(self):
        """
        Test that when creating a consumer, an empty buffer is created.
        """

        queue = Queue()
        consumer = ELDConsumer(queue, 60)
        self.assertEqual(Queue, type(consumer.buffer))
        self.assertEqual(0, consumer.buffer.length())

    @async_test
    async def test_construct_idf_documents(self):
        """
        Test that when constructing the IDF, it uses all documents.
        """

        queue = Queue()
        consumer = ELDConsumer(queue, 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            queue.enqueue(*tweets)
            consumer._started()
            scheme = await consumer._construct_idf(1)
            self.assertEqual(len(lines), scheme.global_scheme.documents)

    @async_test
    async def test_construct_idf_terms(self):
        """
        Test that when constructing the IDF, the correct terms are registered.
        """

        queue = Queue()
        consumer = ELDConsumer(queue, 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            queue.enqueue(*tweets)
            consumer._started()
            scheme = await consumer._construct_idf(1)

            documents = consumer._to_documents(tweets)
            terms = set([ term for document in documents
                               for term in document.dimensions ])

            self.assertEqual(terms, set(scheme.global_scheme.idf))

    @async_test
    async def test_construct_idf_counts(self):
        """
        Test that when constructing the IDF, the correct term counts are registered.
        """

        queue = Queue()
        consumer = ELDConsumer(queue, 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            queue.enqueue(*tweets)
            consumer._started()
            scheme = await consumer._construct_idf(1)

            documents = consumer._to_documents(tweets)
            terms = set([ term for document in documents
                               for term in document.dimensions ])

            for term in terms:
                count = len([ document for document in documents if term in document.dimensions ])
                self.assertEqual(count, scheme.global_scheme.idf[term])

    @async_test
    async def test_construct_idf_buffer(self):
        """
        Test that when constructing the IDF, the documents are added to the buffer.
        """

        queue = Queue()
        consumer = ELDConsumer(queue, 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            queue.enqueue(*tweets)
            consumer._started()
            scheme = await consumer._construct_idf(1)

            documents = consumer._to_documents(tweets)
            for document, buffered in zip(documents, consumer.buffer.queue):
                self.assertEqual(document.text, buffered.text)

    def test_filter_tweets_empty(self):
        """
        Test that when filtering a list of empty tweets, another empty list is returned.
        """

        consumer = ELDConsumer(Queue(), 60)
        self.assertEqual([ ], consumer._filter_tweets([ ]))

    def test_filter_tweets_english(self):
        """
        Test that when filtering a list of tweets, only English tweets are returned.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertTrue(all(tweet['lang'] == 'en' for tweet in tweets))
            self.assertGreater(count, len(tweets))

    def test_filter_tweets_hashtags(self):
        """
        Test that when filtering tweets, all returned tweets have no more than 2 hashtags.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertTrue(all(len(tweet['entities']['hashtags']) <= 2 for tweet in tweets))
            self.assertGreater(count, len(tweets))

    def test_filter_tweets_no_favourites(self):
        """
        Test that when filtering tweets, all returned tweets' authors have favourited at least one tweet.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertTrue(all(tweet['user']['favourites_count'] > 0 for tweet in tweets))
            self.assertGreater(count, len(tweets))

    def test_filter_tweets_follower_ratio(self):
        """
        Test that when filtering tweets, all users have at least one follower for every thousand tweets they've published.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertTrue(all(tweet['user']['followers_count'] / tweet['user']['statuses_count'] >= 1./1000. for tweet in tweets))
            self.assertGreater(count, len(tweets))

    def test_filter_tweets_urls(self):
        """
        Test that when filtering tweets, they can have no more than one URL.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertTrue(all(len(tweet['entities']['urls']) <= 1 for tweet in tweets))
            self.assertGreater(count, len(tweets))

    def test_filter_tweets_bio(self):
        """
        Test that when filtering tweets, their authors must have a non-empty biography.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertTrue(all(tweet['user']['description'] for tweet in tweets))
            self.assertGreater(count, len(tweets))

    def test_filter_tweets_repeat(self):
        """
        Test that when filtering tweets twice, the second time has no effect.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]

            """
            The first time, the number of tweets should decrease.
            """
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertGreater(count, len(tweets))

            """
            The second time, the number of tweets should remain the same.
            """
            count = len(tweets)
            tweets = consumer._filter_tweets(tweets)
            self.assertEqual(count, len(tweets))

    def test_filter_tweets_unchanged(self):
        """
        Test that when filtering tweets, the tweet data does not change.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            filtered = consumer._filter_tweets(tweets)
            self.assertTrue(all(tweet in tweets for tweet in filtered))

    def test_filter_tweets_document(self):
        """
        Test that when filtering a list of documents, the function looks for the tweet in the attributes.
        """

        consumer = ELDConsumer(Queue(), 60, scheme=TF())
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = [ Document('', attributes={ 'tweet': tweet }) for tweet in tweets ]

            tweets = consumer._filter_tweets(tweets)
            documents = consumer._filter_tweets(documents)
            self.assertEqual(len(tweets), len(documents))
            self.assertTrue(all( document.attributes['tweet'] in tweets for document in documents ))

    def test_to_documents_tweet(self):
        """
        Test that when creating a document from a tweet, the tweet is saved as an attribute.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            tweet = json.loads(f.readline())
            document = consumer._to_documents([ tweet ])[0]
            self.assertEqual(tweet['id'], document.attributes['id'])
            self.assertEqual(len(tweet['entities']['urls']), document.attributes['urls'])

    def test_to_documents_ellipsis(self):
        """
        Test that when the text has an ellipsis, the full text is used.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if '…' in tweet['text']:
                    document = consumer._to_documents([ tweet ])[0]

                    """
                    Make an exception for a special case.
                    """
                    if not ('retweeted_status' in tweet and tweet['retweeted_status']['id_str'] == '1238513167573147648'):
                        self.assertFalse(document.text.endswith('…'))

    def test_to_documents_quoted(self):
        """
        Test that when the tweet is a quote, the text is used, not the quoted tweet's text.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if 'retweeted_status' in tweet:
                    timestamp = tweet['timestamp_ms']
                    tweet = tweet['retweeted_status']
                    tweet['timestamp_ms'] = timestamp

                if 'quoted_status' in tweet:
                    document = consumer._to_documents([ tweet ])[0]

                    if 'extended_tweet' in tweet:
                        self.assertEqual(tweet["extended_tweet"].get("full_text", tweet.get("text", "")), document.text)
                    else:
                        self.assertEqual(tweet.get('text'), document.text)

    def test_to_documents_retweeted(self):
        """
        Test that when the tweet is a quote, the retweet's text is used.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if 'retweeted_status' in tweet:
                    document = consumer._to_documents([ tweet ])[0]

                    retweet = tweet['retweeted_status']
                    if 'extended_tweet' in retweet:
                        self.assertEqual(retweet["extended_tweet"].get("full_text", retweet.get("text", "")), document.text)
                    else:
                        self.assertEqual(retweet.get('text'), document.text)

                    """
                    Tweets shouldn't start with 'RT'.
                    """
                    self.assertFalse(document.text.startswith('RT'))

    def test_to_documents_normal(self):
        """
        Test that when the tweet is not a quote or retweet, the full text is used.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                if not 'retweeted_status' in tweet and not 'quoted_status' in tweet:
                    document = consumer._to_documents([ tweet ])[0]

                    if 'extended_tweet' in tweet:
                        self.assertEqual(tweet["extended_tweet"].get("full_text", tweet.get("text", "")), document.text)
                    else:
                        self.assertEqual(tweet.get('text'), document.text)

                    """
                    There should be no ellipsis in the text now.
                    """
                    self.assertFalse(document.text.endswith('…'))

    def test_to_documents_normalized(self):
        """
        Test that the documents are returned normalized.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                document = consumer._to_documents([ tweet ])[0]
                self.assertEqual(1, round(vector_math.magnitude(document), 10))

    def test_to_documents_documents(self):
        """
        Test that when converting a list of documents to documents, they are retained.
        """

        consumer = ELDConsumer(Queue(), 60, TF())
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            self.assertEqual(documents, consumer._to_documents(documents))

    def test_to_documents_documents_with_attributes(self):
        """
        Test that when converting a list of documents to documents, their attributes are updated.
        """

        consumer = ELDConsumer(Queue(), 60, TF())
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = [ Document('', attributes={ 'tweet': tweet}) for tweet in tweets ]
            documents = consumer._to_documents(tweets)
            self.assertTrue(all( 'urls' in document.attributes for document in documents ))

    def test_latest_timestamp_empty(self):
        """
        Test that when getting the timestamp from an empty set, a ValueError is raised.
        """

        consumer = ELDConsumer(Queue(), 60)
        self.assertRaises(ValueError, consumer._latest_timestamp, [ ])

    def test_latest_timestamp(self):
        """
        Test getting the latest timestamp from a corpus of documents.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            self.assertEqual(documents[-1].attributes['timestamp'], consumer._latest_timestamp(documents))

    def test_latest_timestamp_reversed(self):
        """
        Test that when getting the latest timestamp from a corpus of reversed documents, the actual latest timestamp is given.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)[::-1]
            self.assertEqual(documents[0].attributes['timestamp'], consumer._latest_timestamp(documents))

    def test_create_checkpoint_first(self):
        """
        Test that when creating the first checkpoint, the nutrition is created from scratch.
        """

        consumer = ELDConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.store.all())
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertTrue(consumer.store.get(timestamp))
            self.assertEqual(set(documents[0].dimensions), set(consumer.store.get(timestamp)))

    def test_create_checkpoint_empty(self):
        """
        Test that when creating an empty checkpoint, it is still recorded.
        """

        consumer = ELDConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.store.all())
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            timestamp = twitter.extract_timestamp(tweet)
            consumer._create_checkpoint(timestamp)
            self.assertEqual({ }, consumer.store.get(timestamp))

    def test_create_checkpoint_timestamp(self):
        """
        Test that when creating checkpoints, the correct timestamp is recorded.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))

    def test_create_checkpoint_scale(self):
        """
        Test that when creating checkpoints, they are rescaled correctly.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            timestamp = twitter.extract_timestamp(tweets[-1])
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertLessEqual(0, min(consumer.store.get(timestamp).values()))
            self.assertEqual(1, max(consumer.store.get(timestamp).values()))

    def test_create_checkpoint_filter_empty(self):
        """
        Test that when creating a checkpoint with the timestamp before any published documents, it is empty.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer.buffer.enqueue(*documents)
            timestamp = twitter.extract_timestamp(tweets[0])
            consumer._create_checkpoint(timestamp - 1)

    def test_create_checkpoint_filter_inclusive(self):
        """
        Test that when creating a checkpoint, the timestamp filter is inclusive.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer.buffer.enqueue(*documents)
            timestamp = twitter.extract_timestamp(tweets[0])
            consumer._create_checkpoint(timestamp)

            """
            Work out which dimensions should be in the checkpoint.
            """
            dimensions = [ dimension for document in documents
                                      for dimension in document.dimensions
                                     if document.attributes['timestamp'] <= timestamp ]
            self.assertEqual(set(dimensions), set(consumer.store.get(timestamp)))

    def test_create_checkpoint_removes_documents_from_buffer(self):
        """
        Test that when creating a checkpoint, the documents are removed from the buffer..
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer.buffer.enqueue(*documents)
            timestamp = twitter.extract_timestamp(tweets[0])
            self.assertEqual(len(tweets), consumer.buffer.length())
            consumer._create_checkpoint(timestamp)
            self.assertEqual(len(tweets) - 100, consumer.buffer.length())

    def test_create_checkpoint_reorders_buffer(self):
        """
        Test that when creating a checkpoint and the buffer has mixed-up documents, the buffer is re-ordered.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer.buffer.enqueue(*documents[::-1])
            timestamp = twitter.extract_timestamp(tweets[0])
            consumer._create_checkpoint(timestamp)
            self.assertTrue(all(consumer.buffer.queue[i].attributes['timestamp'] <= consumer.buffer.queue[i + 1].attributes['timestamp'])
                                for i in range(len(consumer.buffer.queue) - 1))

    def test_create_checkpoint_wrong_order(self):
        """
        Test that when creating a checkpoint and the buffer has mixed-up documents, the correct documents are used.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer.buffer.enqueue(*documents[::-1])
            timestamp = twitter.extract_timestamp(tweets[0])
            consumer._create_checkpoint(timestamp)

            """
            Work out which dimensions should be in the checkpoint.
            """
            dimensions = [ dimension for document in documents
                                      for dimension in document.dimensions
                                     if document.attributes['timestamp'] <= timestamp ]
            self.assertEqual(set(dimensions), set(consumer.store.get(timestamp)))

    def test_checkpoint_empty(self):
        """
        Test that when creating a checkpoint from a document with no dimensions, an empty checkpoint is returned.
        """

        consumer = ELDConsumer(Queue(), 60, log_nutrition=True)
        document = Document('is not', { }, attributes={ 'timestamp': 10 })
        checkpoint = consumer._checkpoint(document)
        self.assertEqual({ }, checkpoint)

    def test_checkpoint_log_nutrition(self):
        """
        Test that when creating checkpoints with logarithmic nutrition, the scaling uses the logarithm.
        """

        consumer = ELDConsumer(Queue(), 60, log_nutrition=True)
        document = Document('joe biden', { 'joe': 10, 'biden': 1000 }, attributes={ 'timestamp': 10 })
        checkpoint = consumer._checkpoint(document)
        self.assertEqual(round(1/3, 10), round(checkpoint.get('joe'), 10))
        self.assertEqual(1, checkpoint.get('biden'))

    def test_checkpoint_no_log_nutrition(self):
        """
        Test that when creating checkpoints without logarithmic nutrition, the scaling does not use the logarithm.
        """

        consumer = ELDConsumer(Queue(), 60, log_nutrition=False)
        document = Document('joe biden', { 'joe': 10, 'biden': 1000 }, attributes={ 'timestamp': 10 })
        checkpoint = consumer._checkpoint(document)
        self.assertEqual(round(1/100, 10), round(checkpoint.get('joe'), 10))
        self.assertEqual(1, checkpoint.get('biden'))

    def test_create_checkpoint_scale(self):
        """
        Test that when creating checkpoints, they are rescaled correctly.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            document = Document.concatenate(*documents, tokenizer=consumer.tokenizer)
            checkpoint = consumer._checkpoint(document)
            self.assertLessEqual(0, min(checkpoint.values()))
            self.assertEqual(1, max(checkpoint.values()))

    def test_remove_old_checkpoints_empty(self):
        """
        Test that when removing checkpoints from an empty store, nothing happens.
        """

        consumer = ELDConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.store.all())
        consumer._remove_old_checkpoints(100)
        self.assertEqual({ }, consumer.store.all())

    def test_remove_old_checkpoints_zero_timestamp(self):
        """
        Test that when removing checkpoints at timestamp 0, nothing is removed.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))
            consumer._remove_old_checkpoints(0)

    def test_remove_old_checkpoints_small_timestamp(self):
        """
        Test that when removing checkpoints with a small timestamp that does not cover the entire sets, nothing is removed.
        """

        consumer = ELDConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(10)
            self.assertEqual([ 10 ], list(consumer.store.all().keys()))
            consumer._remove_old_checkpoints(9)

    def test_remove_old_checkpoints_exclusive(self):
        """
        Test that when removing checkpoints, the removal is exclusive.
        """

        consumer = ELDConsumer(Queue(), 60, sets=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))
            consumer._remove_old_checkpoints(timestamp + 600)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))

    def test_remove_old_checkpoints(self):
        """
        Test that when removing checkpoints, any nutrition data out of frame is removed.
        """

        consumer = ELDConsumer(Queue(), 60, sets=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))
            consumer._remove_old_checkpoints(timestamp + 600 + 1)
            self.assertEqual([ ], list(consumer.store.all().keys()))

    def test_filter_clusters_empty(self):
        """
        Test that when filtering an empty list of clusters, another empty list is returned.
        """

        consumer = ELDConsumer(Queue(), 60)
        self.assertEqual([ ], consumer._filter_clusters([ ], 0))

    def test_filter_clusters_copy(self):
        """
        Test that when filtering a list of clusters, the list itself doesn't change, but a copy is returned.
        The test creates singleton clusters for all documents so they are all filtered out.
        """

        consumer = ELDConsumer(Queue(), 60, sets=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(document) for document in documents ]
            original = len(clusters)
            timestamp = documents[-1].attributes['timestamp']
            self.assertEqual(0, len(consumer._filter_clusters(clusters, timestamp)))
            self.assertEqual(original, len(clusters))

    def test_filter_clusters_size_inclusive(self):
        """
        Test that when filtering a list of clusters, the minimum size is inclusive.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:3]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_small(self):
        """
        Test that when filtering a list of clusters, small clusters are filtered out.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:2]) ]
            self.assertEqual([ ], consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_large(self):
        """
        Test that when filtering a list of clusters, large clusters are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:4]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_recently_checked(self):
        """
        Test that when filtering a list of clusters, clusters that have been recently checked are filtered out.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=0, cooldown=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:2], { 'last_checked': 10 }) ]
            self.assertEqual([ ], consumer._filter_clusters(clusters, 11))

    def test_filter_clusters_never_checked(self):
        """
        Test that when filtering a list of clusters, clusters that have never been checked are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=0, cooldown=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:2]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 11))

    def test_filter_clusters_recently_checked_exclusive(self):
        """
        Test that when filtering a list of clusters, the checked filter is exclusive.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=0, cooldown=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:2], { 'last_checked': 10 }) ]
            self.assertEqual([ ], consumer._filter_clusters(clusters, 20))

    def test_filter_clusters_checked_long_ago(self):
        """
        Test that when filtering a list of clusters, clusters that were checked a long time ago are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=0, cooldown=10)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:2], { 'last_checked': 10 }) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 21))

    def test_filter_clusters_intra_similarity_low(self):
        """
        Test that when filtering a list of clusters, clusters with a low intra-similarity are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:3]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_intra_similarity_high(self):
        """
        Test that when filtering a list of clusters, clusters with a high intra-similarity are not retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster([ documents[0] ] * 3) ]
            self.assertEqual([ ], consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_not_bursty(self):
        """
        Test that when filtering a list of clusters, clusters that are explicitly not bursty are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            cluster = Cluster(documents[:3], attributes={ 'bursty': False })
            self.assertEqual([ cluster ], consumer._filter_clusters([ cluster ], 10))

    def test_filter_clusters_bursty(self):
        """
        Test that when filtering a list of clusters, clusters that are explicitly bursty are removed.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            cluster = Cluster(documents[:3], attributes={ 'bursty': True })
            self.assertEqual([ ], consumer._filter_clusters([ cluster ], 10))

    def test_filter_clusters_unknown_bursty(self):
        """
        Test that when filtering a list of clusters, clusters that are implicitly not bursty are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            cluster = Cluster(documents[:3])
            self.assertEqual([ cluster ], consumer._filter_clusters([ cluster ], 10))

    def test_filter_clusters_bursty_attribute_unchanged(self):
        """
        Test that when filtering a list of clusters, the bursty attribute is unchanged.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)

            cluster = Cluster(documents[:3], attributes={ 'bursty': True })
            self.assertEqual([ ], consumer._filter_clusters([ cluster ], 10))
            self.assertTrue(cluster.attributes['bursty'])

            cluster = Cluster(documents[:3], attributes={ 'bursty': False })
            self.assertEqual([ cluster ], consumer._filter_clusters([ cluster ], 10))
            self.assertFalse(cluster.attributes['bursty'])

            cluster = Cluster(documents[:3])
            self.assertEqual([ cluster ], consumer._filter_clusters([ cluster ], 10))
            self.assertEqual(None, cluster.attributes.get('bursty'))

    def test_filter_clusters_no_urls(self):
        """
        Test that when filtering a list of clusters, clusters whose tweets have no URLs are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            tweets = [ tweet for tweet in tweets if not tweet['entities']['urls'] ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:3]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_few_urls(self):
        """
        Test that when filtering a list of clusters, clusters with a few URLs are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:20]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_urls_inclusive(self):
        """
        Test that when filtering a list of clusters, the check for URLs is inclusive.
        This test adds documents having exactly one URL to a cluster.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            tweets = [ tweet for tweet in tweets if len(tweet['entities']['urls']) == 1 ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:50]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_urls_average(self):
        """
        Test that when filtering a list of clusters, the check for URLs is an average.
        This test adds documents having no URLs and documents having 2 URLs to a cluster.
        This brings the average to 1 URL per document.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            no_url_tweets = [ tweet for tweet in tweets if not len(tweet['entities']['urls']) ]
            url_tweets = [ tweet for tweet in tweets if len(tweet['entities']['urls']) == 2 ]
            no_url_documents = consumer._to_documents(no_url_tweets)
            url_documents = consumer._to_documents(url_tweets)
            clusters = [ Cluster(no_url_documents[:50] + url_documents[:50]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_many_urls(self):
        """
        Test that when filtering a list of clusters, clusters with many URLs are filtered out.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            tweets = [ tweet for tweet in tweets if len(tweet['entities']['urls']) == 2 ]
            documents = consumer._to_documents(tweets)
            clusters = [ Cluster(documents[:50]) ]
            self.assertEqual([ ], consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_no_replies(self):
        """
        Test that when filtering a list of clusters, clusters without replies are retained.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = [ document for document in documents if not document.text.startswith('@') ]
            clusters = [ Cluster(documents[:3]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_few_replies(self):
        """
        Test that when filtering a list of clusters, clusters with few replies are retained.
        This test adds one document with a reply and the rest without replies to the cluster.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = [ document for document in documents if not document.text.startswith('@') ]
            reply_documents = [ document for document in documents if document.text.startswith('@') ]
            clusters = [ Cluster(documents[:3] + reply_documents[:1]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_replies_inclusive(self):
        """
        Test that when filtering a list of clusters, the check for replies is inclusive.
        This test adds three documents that are replies and three others that aren't to a clustser.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = [ document for document in documents if not document.text.startswith('@') ]
            reply_documents = [ document for document in documents if document.text.startswith('@') ]
            clusters = [ Cluster(documents[:3] + reply_documents[:3]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_replies_average(self):
        """
        Test that when filtering a list of clusters, the check for replies is an average.
        This test adds many documents that aren't mentions and one document that is to a cluster.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = [ document for document in documents if not document.text.startswith('@') ]
            reply_documents = [ document for document in documents if document.text.startswith('@') ]
            clusters = [ Cluster(documents[:3] + reply_documents[:1]) ]
            self.assertEqual(clusters, consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_many_replies(self):
        """
        Test that when filtering a list of clusters, clusters with many replies are filtered out.
        The proportion of documents added to a cluster is three being replies, and two that aren't.
        """

        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = [ document for document in documents if not document.text.startswith('@') ]
            reply_documents = [ document for document in documents if document.text.startswith('@') ]
            clusters = [ Cluster(documents[:2] + reply_documents[:3]) ]
            self.assertEqual([ ], consumer._filter_clusters(clusters, 10))

    def test_filter_clusters_mix(self):
        """
        Test that when filtering a list of clusters, only those that need to be filtered out are removed.
        In this test, one cluster is too small, one was checked recently, the other has identical documents, another is bursty, one more has many URLs and a final one with many mentions.
        A valid cluster is also among the clusters.
        """

        clusters = [ ]
        consumer = ELDConsumer(Queue(), 60, min_size=3, max_intra_similarity=0.8)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            clusters.append(Cluster(documents[:2]))
            clusters.append(Cluster(documents[:50], { 'last_checked': 10 }))
            clusters.append(Cluster([ documents[0] ] * 3))
            clusters.append(Cluster(documents[:50], { 'bursty': True }))

            no_url_documents = [ document for document in documents if document.attributes['urls'] == 0 ]
            url_documents = [ document for document in documents if document.attributes['urls'] >= 2 ]
            clusters.append(Cluster(no_url_documents[:1] + url_documents[:3]))

            no_reply_documents = [ document for document in documents if not document.text.startswith('@') ]
            reply_documents = [ document for document in documents if document.text.startswith('@') ]
            clusters.append(Cluster(no_reply_documents[:2] + reply_documents[:3]))

            cluster = Cluster(documents[:50])
            clusters.append(cluster)
            self.assertEqual([ cluster ], consumer._filter_clusters(clusters, 10))

    def test_detect_topics_breaking(self):
        """
        Test that when detecting topics, the returned terms should be breaking.
        """

        consumer = ELDConsumer(Queue(), 30, min_burst=0)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))
            self.assertEqual(documents[0].dimensions.keys(), consumer.store.get(timestamp).keys())

            """
            Create a new cluster with a sligtly different tweet.
            The function should return some of the different dimensions as breaking terms.
            """
            document = documents[0].copy()
            document.text = document.text + ' pipe'
            cluster = Cluster(document)
            terms = consumer._detect_topics(cluster, timestamp + 60)
            self.assertEqual([ 'pipe' ], list(terms))
            self.assertEqual(0.5, terms.get('pipe'))

    def test_detect_topics_dict(self):
        """
        Test that when detecting topics, the returned terms are returned as a dictionary.
        """

        consumer = ELDConsumer(Queue(), 30, min_burst=0)
        with open(os.path.join(os.path.dirname(__file__), 'corpus.json'), 'r') as f:
            line = f.readline()
            tweet = json.loads(line)
            documents = consumer._to_documents([ tweet ])
            timestamp = twitter.extract_timestamp(tweet)
            consumer.buffer.enqueue(*documents)
            consumer._create_checkpoint(timestamp)
            self.assertEqual([ timestamp ], list(consumer.store.all().keys()))
            self.assertEqual(documents[0].dimensions.keys(), consumer.store.get(timestamp).keys())

            """
            Create a new cluster with a sligtly different tweet.
            The function should return some of the different dimensions as breaking terms.
            """
            document = documents[0].copy()
            document.text = document.text + ' pipe'
            cluster = Cluster(document)
            terms = consumer._detect_topics(cluster, timestamp + 60)
            self.assertEqual(dict, type(terms))

    def test_score_documents_empty(self):
        """
        Test that when scoring an empty list of documents, an empty list is returned.
        """

        consumer = ELDConsumer(Queue(), 30)
        self.assertEqual([ ], consumer._score_documents([ ]))

    def test_score_documents_sorted(self):
        """
        Test that when scoring documents, the returned list is sorted.
        """

        consumer = ELDConsumer(Queue(), 30)
        documents = [ Document('THIS IS A PIPE'),
                      Document('This is a pipe and this is a cigar'),
                      Document('this is a pipe'), ]
        self.assertEqual([ documents[1], documents[2], documents[0] ], consumer._score_documents(documents))

    def test_score_documents(self):
        """
        Test that when scoring documents, the same list of documents is returned, albeit in a different order.
        """

        consumer = ELDConsumer(Queue(), 30)
        documents = [ Document('THIS IS A PIPE'),
                      Document('This is a pipe and this is a cigar'),
                      Document('this is a pipe'), ]
        self.assertEqual(set(documents), set(consumer._score_documents(documents)))

    def test_brevity_score_empty(self):
        """
        Test that the brevity score is 0 when the text is empty.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = ''
        self.assertEqual(0, consumer._brevity_score(text))

    def test_brevity_score(self):
        """
        Test the calculation of the brevity score.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'this is a pipe'
        self.assertEqual(0.00012, round(consumer._brevity_score(text, r=10), 5))

    def test_brevity_score_equal(self):
        """
        Test that when the text has as many tokens as required, the score is 1.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'a pipe is not a cigar and a cigar is not a pipe'
        self.assertEqual(1, consumer._brevity_score(text, r=4))

    def test_brevity_score_long(self):
        """
        Test that when the text has more tokens than required, the score is 1.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'a pipe is not a cigar and a cigar is not a pipe'
        self.assertEqual(1, consumer._brevity_score(text, r=3))

    def test_brevity_score_bounds(self):
        """
        Test that the bounds of the brevity score are between 0 and 1.
        """

        consumer = ELDConsumer(Queue(), 30)

        text = ''
        self.assertEqual(0, consumer._brevity_score(text))
        text = 'a pipe is not a cigar and a cigar is not a pipe'
        self.assertEqual(1, consumer._brevity_score(text, r=3))

    def test_brevity_score_custom_r(self):
        """
        Test that when a custom ideal length is given, it is used.
        """

        consumer = ELDConsumer(Queue(), 30)

        text = 'a pipe is not a cigar'
        self.assertEqual(0.60653, round(consumer._brevity_score(text, r=3), 5))
        text = 'a pipe is not a cigar'
        self.assertEqual(0.36788, round(consumer._brevity_score(text, r=4), 5))

    def test_emotion_score_empty(self):
        """
        Test that the emotional score of an empty string is 0.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = ''
        self.assertEqual(0, consumer._emotion_score(text))

    def test_emotion_score_all_lower(self):
        """
        Test that the emotional score of a string that is lowercase is 1.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'this is not a pipe'
        self.assertEqual(1, consumer._emotion_score(text))

    def test_emotion_score_all_upper(self):
        """
        Test that the emotional score of a string that is uppercase is 0.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'THIS IS NOT A PIPE'
        self.assertEqual(0, consumer._emotion_score(text))

    def test_emotion_score_numbers(self):
        """
        Test that numbers in a string do not count when calculating the emotional score.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'This is not a pipe 500'
        self.assertEqual(13/14, consumer._emotion_score(text))

    def test_emotion_score_punctuation(self):
        """
        Test that numbers in a string count as lowercase when calculating the emotional score.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'This is not a pipe.'
        self.assertEqual(13/14, consumer._emotion_score(text))

    def test_emotion_score_bounds(self):
        """
        Test that the bounds of the emotional score are between 0 and 1.
        """

        consumer = ELDConsumer(Queue(), 30)

        text = 'THIS IS NOT A PIPE'
        self.assertEqual(0, consumer._emotion_score(text))
        text = 'this is not a pipe'
        self.assertEqual(1, consumer._emotion_score(text))

    def test_emotion_score(self):
        """
        Test the emotion score calculation.
        """

        consumer = ELDConsumer(Queue(), 30)
        text = 'This is not a pipe'
        self.assertEqual(13/14, consumer._emotion_score(text))
