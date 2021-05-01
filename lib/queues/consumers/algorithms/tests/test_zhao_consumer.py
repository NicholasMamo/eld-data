"""
Test the functionality of the Zhao et al. consumer.
"""

import json
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
if path not in sys.path:
    sys.path.append(path)

from queues import Queue
from queues.consumers.algorithms import ZhaoConsumer
from vsm import vector_math

class TestZhaoConsumer(unittest.TestCase):
    """
    Test the implementation of the Zhao et al. consumer.
    """

    def test_init_name(self):
        """
        Test that the Zhao consumer passes on the name to the base class.
        """

        name = 'Test Consumer'
        consumer = ZhaoConsumer(Queue(), periodicity=10, name=name)
        self.assertEqual(name, str(consumer))

    def test_create_consumer(self):
        """
        Test that when creating a consumer, all the parameters are saved correctly.
        """

        queue = Queue()
        consumer = ZhaoConsumer(queue, 60)
        self.assertEqual(queue, consumer.queue)
        self.assertEqual(60, consumer.periodicity)

    def test_create_consumer_store(self):
        """
        Test that when creating a consumer, an empty nutrition store is created.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.store.all())

    def test_create_consumer_default_post_rate(self):
        """
        Test that the default post rate is 1.7.
        This constant is what is used by Zhao et al. in their paper.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual(1.7, consumer.tdt.post_rate)

    def test_create_consumer_custom_post_rate(self):
        """
        Test that when passing on a post rate, it is passed on to the TDT algorithm.
        """

        consumer = ZhaoConsumer(Queue(), 60, post_rate=1.9)
        self.assertEqual(1.9, consumer.tdt.post_rate)

    def test_to_documents_tweet(self):
        """
        Test that when creating a document from a tweet, the tweet is saved as an attribute.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            tweet = json.loads(f.readline())
            document = consumer._to_documents([ tweet ])[0]
            self.assertEqual(tweet, document.attributes['tweet'])

    def test_to_documents_ellipsis(self):
        """
        Test that when the text has an ellipsis, the full text is used.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
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

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
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

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
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

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
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

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                document = consumer._to_documents([ tweet ])[0]
                if vector_math.magnitude(document) == 0:
                    continue
                self.assertEqual(1, round(vector_math.magnitude(document), 10))

    def test_latest_timestamp_empty(self):
        """
        Test that when getting the timestamp from an empty set, a ValueError is raised.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertRaises(ValueError, consumer._latest_timestamp, [ ])

    def test_latest_timestamp(self):
        """
        Test getting the latest timestamp from a corpus of documents.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            self.assertEqual(documents[-1].attributes['timestamp'], consumer._latest_timestamp(documents))

    def test_latest_timestamp_reversed(self):
        """
        Test that when getting the latest timestamp from a corpus of reversed documents, the actual latest timestamp is given.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)[::-1]
            self.assertEqual(documents[0].attributes['timestamp'], consumer._latest_timestamp(documents))

    def test_add_documents_all_timestamps(self):
        """
        Test that when adding documents to the historical data, the correct timestamps are saved.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer._add_documents(documents)
            self.assertEqual(set(document.attributes['timestamp'] for document in documents), set(consumer.documents.keys()))

    def test_add_documents_empty(self):
        """
        Test that when adding no documents to the historical data, it remains unchanged.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer._add_documents(documents)
            self.assertEqual(set(document.attributes['timestamp'] for document in documents), set(consumer.documents.keys()))
            consumer._add_documents([ ])
            self.assertEqual(set(document.attributes['timestamp'] for document in documents), set(consumer.documents.keys()))

    def test_add_documents_multiple(self):
        """
        Test that when adding documents to the historical data, each timestamp can have multiple documents.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer._add_documents(documents)
            self.assertTrue(all( len(documents) for documents in consumer.documents.values() ))
            self.assertTrue(any( len(documents) > 1 for documents in consumer.documents.values() ))

    def test_documents_since_empty(self):
        """
        Test that getting the documents when there are no documents returns an empty list.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
            self.assertEqual([ ], consumer._documents_since(0))

    def test_documents_since_all(self):
        """
        Test that getting the documents since the first timestamp returns all documents.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
            consumer._add_documents(documents)
            self.assertEqual(documents, consumer._documents_since(0))

    def test_documents_since_order(self):
        """
        Test that when getting all documents since a timestamp, the documents are ordered chronologically.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer._add_documents(documents[::-1])
            documents = consumer._documents_since(0)
            self.assertTrue(all(documents[i].attributes['timestamp'] <= documents[i + 1].attributes['timestamp']
                            for i in range(0, len(documents) - 1)))

    def test_documents_since_inclusive(self):
        """
        Test that getting the documents since a timestamp returns an inclusive result set.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
            consumer._add_documents(documents)
            self.assertEqual(documents, consumer._documents_since(documents[0].attributes['timestamp']))

    def test_documents_since_last(self):
        """
        Test that getting the documents since the last timestamp returns all documents published at the same time.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        self.assertEqual({ }, consumer.documents)

        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
            consumer._add_documents(documents)
            self.assertEqual([ document for document in documents if document.attributes['timestamp'] == documents[-1].attributes['timestamp'] ],
                             consumer._documents_since(documents[-1].attributes['timestamp']))

    def test_documents_since_none(self):
         """
         Test that getting the documents beyond the last timestamp returns an empty set.
         """

         consumer = ZhaoConsumer(Queue(), 60)
         self.assertEqual({ }, consumer.documents)

         with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
             lines = f.readlines()
             tweets = [ json.loads(line) for line in lines ]
             documents = consumer._to_documents(tweets)
             documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
             consumer._add_documents(documents)
             self.assertEqual([ ], consumer._documents_since(documents[-1].attributes['timestamp'] + 1))

    def test_remove_documents_before_empty(self):
         """
         Test that when removing documents when there are no documents, nothing is removed.
         """

         consumer = ZhaoConsumer(Queue(), 60)
         self.assertEqual({ }, consumer.documents)

         with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
             lines = f.readlines()
             tweets = [ json.loads(line) for line in lines ]
             documents = consumer._to_documents(tweets)
             documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
             consumer._remove_documents_before(documents[0].attributes['timestamp'] - 1)
             self.assertEqual([ ], consumer._documents_since(0))

    def test_remove_documents_before_none(self):
         """
         Test that when removing documents that were published before the first document, nothing is removed.
         """

         consumer = ZhaoConsumer(Queue(), 60)
         self.assertEqual({ }, consumer.documents)

         with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
             lines = f.readlines()
             tweets = [ json.loads(line) for line in lines ]
             documents = consumer._to_documents(tweets)
             documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
             consumer._add_documents(documents)
             consumer._remove_documents_before(documents[0].attributes['timestamp'] - 1)
             self.assertEqual(documents, consumer._documents_since(0))

    def test_remove_documents_before_exclusive(self):
         """
         Test that when removing documents, the removal is exclusive.
         """

         consumer = ZhaoConsumer(Queue(), 60)
         self.assertEqual({ }, consumer.documents)

         with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
             lines = f.readlines()
             tweets = [ json.loads(line) for line in lines ]
             documents = consumer._to_documents(tweets)
             documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
             consumer._add_documents(documents)
             consumer._remove_documents_before(documents[0].attributes['timestamp'])
             self.assertEqual(documents, consumer._documents_since(0))
             consumer._remove_documents_before(documents[-1].attributes['timestamp'])
             self.assertTrue(all( document in consumer._documents_since(0)
                                           for document in documents
                                           if document.attributes['timestamp'] >= documents[-1].attributes['timestamp'] ))
             self.assertTrue(all( document.attributes['timestamp'] >= documents[-1].attributes['timestamp']
                                           for document in consumer._documents_since(0) ))

    def test_remove_documents_before_all(self):
         """
         Test that when removing all documents, no documents remain.
         """

         consumer = ZhaoConsumer(Queue(), 60)
         self.assertEqual({ }, consumer.documents)

         with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
             lines = f.readlines()
             tweets = [ json.loads(line) for line in lines ]
             documents = consumer._to_documents(tweets)
             documents = sorted(documents, key=lambda document: document.attributes['timestamp'])
             consumer._add_documents(documents)
             consumer._remove_documents_before(documents[-1].attributes['timestamp'] + 1)
             self.assertEqual([ ], consumer._documents_since(0))

    def test_create_checkpoint_empty(self):
        """
        Test that when creating the first checkpoint, the nutrition is created from scratch.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            for line in f:
                tweet = json.loads(line)
                documents = consumer._to_documents([ tweet ])
                consumer._create_checkpoint(documents)
                self.assertEqual([ 1 ], list(consumer.store.all().values()))
                break

    def test_create_checkpoint_multiple_empty(self):
        """
        Test that when creating the first checkpoint with multiple tweets, the nutrition is created from scratch.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()[:10]
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            self.assertEqual([ ], list(consumer.store.all().values()))
            consumer._create_checkpoint(documents)
            self.assertEqual(set(document.attributes['timestamp'] for document in documents), set(consumer.store.all()))
            self.assertTrue(all( volume > 0 for volume in consumer.store.all().values() ))

    def test_create_checkpoint_increment(self):
        """
        Test that when creating checkpoints, the nutrition increments.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            for i, document in enumerate(documents):
                volume = consumer.store.get(document.attributes['timestamp']) or 0
                consumer._create_checkpoint([ document ])
                self.assertEqual(volume + 1, consumer.store.get(document.attributes['timestamp']))

    def test_create_checkpoint_all_documents(self):
        """
        Test that when creating checkpoints, all documents are stored.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer._create_checkpoint(documents)
            self.assertEqual(len(lines), sum(consumer.store.all().values()))

    def test_create_checkpoint_all_timestamps(self):
        """
        Test that when creating checkpoints, the correct timestamp is recorded.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            timestamps = [ document.attributes['timestamp'] for document in documents ]
            consumer._create_checkpoint(documents)
            self.assertEqual(sorted(set(timestamps)), sorted(consumer.store.all().keys()))

    def test_create_checkpoint_range(self):
        """
        Test that when creating checkpoints, the correct range of timestamps is created.
        """

        consumer = ZhaoConsumer(Queue(), 60)
        with open(os.path.join(os.path.dirname(__file__), '../../../../tests/corpora/CRYCHE-500.json'), 'r') as f:
            lines = f.readlines()
            tweets = [ json.loads(line) for line in lines ]
            documents = consumer._to_documents(tweets)
            consumer._create_checkpoint(documents)
            self.assertEqual(documents[0].attributes['timestamp'], min(consumer.store.all()))
            self.assertEqual(documents[-1].attributes['timestamp'], max(consumer.store.all()))
