import unittest
import ast
import numpy as np
import pdb
import nltk

from homework_06.src.classifier import NaiveBayes
import homework_06.src.tagger as tagger

class TaggerTest(unittest.TestCase):

    def setUp(self):
        self.review = open('test_data/review.txt').read()
        self.short_review = "wow, Hamburger is a great movie"
        self.review_words = nltk.word_tokenize(self.review)
        self.features = ['screenplay', 'heart-warming', 'writing',
            'unattractive', 'dfsdf']
        self.top_words = [
            'the', 'and', 'a', 'of', 'is', 'to', 'it', 'movie', 'as', 'this',
            'he',  'in', 'john', 'that', 'malkovich', 'movies']
        self.review_pos = sorted(ast.literal_eval(
            open('test_data/review_pos.data').read()))
        self.review_pos_vb = sorted(ast.literal_eval(
            open('test_data/review_pos_vb.data').read()))
        self.review_feat = ast.literal_eval(
            open('test_data/review_feature.data').read())
        self.short_review_pos = sorted(
            [('great', 'JJ'), ('Hamburger', 'NNP'), ('wow', 'NN'),
            ('movie', 'NN'), ('is', 'VBZ'), ('a', 'DT')])

    def test_onlyAlpha(self):
        data = tagger.onlyAlpha(self.review_words)
        self.assertIs(all([word.isalpha() for word in data]), True)

    def test_checkFeatures(self):
        ans = [True, True, True, True, False]
        testAns = dict(zip(self.features, ans))
        self.assertEqual(
            tagger.checkFeatures(self.review_words, self.features), testAns)

    def test_getTopWords(self):
        top_words = tagger.getTopWords(self.review_words, .05)
        self.assertEqual(top_words, self.top_words)

    def test_posTagger(self):
        #import pdb; pdb.set_trace()
        pos = tagger.posTagger([self.review, self.short_review])
        pos_vb = tagger.posTagger([self.review], 'VB')
        total_pos = sorted(set(self.review_pos).union(self.short_review_pos))
        self.assertEqual(sorted(pos), total_pos)
        self.assertEqual(sorted(pos_vb), self.review_pos_vb)

    def test_document_features(self):
        features = tagger.document_features(self.review, self.review_pos)
        self.assertEqual(features, self.review_feat)


class TestClassifier(unittest.TestCase):

    def setUp(self):
        #Training data
        sentences1 = [list(set(x.split(' '))) for x in
                      ['do they speak english in what',
                       'I wrote you many many times',
                       ('If you strike me down I shall become more '
                        'powerful than you can possibly imagine'),
                       ('I hate to disappoint you but rubber lips '
                        'are immune to your charms'),
                       ('how do you get a farmer to sell his horse '
                        'when he does not want to sell')]]
        self.data = zip(['good', 'bad', 'good', 'bad', 'good'],
                        sentences1)

        self.test_data = list(set('I am django wroteto rubber charm'.split(' ')))

        self.classifier = NaiveBayes(self.data)
        self.classifier_log = NaiveBayes(self.data, log=True)
        self.classifier_smoothed = NaiveBayes(self.data, smoothing=0.5)
        self.classifier_full = NaiveBayes(self.data, log=True,
                                          smoothing=0.5)

    def test_basic(self):
        expected = "bad"
        prediction = self.classifier.classify(self.test_data)

        self.assertEqual(expected, prediction)

        prediction, prob = self.classifier.classify(self.test_data, prob=True)
        self.assertEqual(prediction, expected)

    def test_log_transform(self):
        expected = "bad"
        prediction = self.classifier_log.classify(self.test_data)

        self.assertEqual(expected, prediction)

        prediction, prob = self.classifier_log.classify(self.test_data,
                                                        prob=True)
        self.assertEqual(prediction, expected)

    def test_smoothing(self):
        expected = "bad"
        prediction = self.classifier_smoothed.classify(self.test_data)

        self.assertEqual(expected, prediction)

        prediction, prob = self.classifier_smoothed.classify(self.test_data,
                                                             prob=True)
        self.assertEqual(prediction, expected)

    def test_both(self):
        expected = "bad"
        prediction = self.classifier_full.classify(self.test_data)

        self.assertEqual(expected, prediction)

        prediction, prob = self.classifier_full.classify(self.test_data,
                                                         prob=True)
        self.assertEqual(prediction, expected)

    def test_malformed_data(self):
        data = None
        self.assert_(self.classifier.classify(data) is None)

        data = []
        rs, p = self.classifier.classify(data, prob=True)
        self.assert_(rs is None)
        self.assert_(np.isnan(p))

        # this would just be silly
        data = lambda x: x
        self.assertRaises(TypeError, self.classifier.classify, data)


suite = unittest.TestLoader().loadTestsFromTestCase(TestClassifier)
unittest.TextTestRunner(verbosity=2).run(suite)
