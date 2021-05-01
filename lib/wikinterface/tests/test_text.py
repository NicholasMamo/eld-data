"""
Run unit tests on the text module.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from wikinterface import text

class TestText(unittest.TestCase):
    """
    Test the text module.
    """

    def test_collect_none(self):
        """
        Test that when no pages are given, an empty dictionary is returned.
        """

        extracts = text.collect([ ])
        self.assertFalse(len(extracts))

    def test_redirection(self):
        """
        Test that pages may redirect, but the original pages are retained.
        """

        page = 'Olympique Lyon'
        extracts = text.collect(page)
        self.assertTrue(page in extracts)

    def test_get_multiple_full(self):
        """
        Test that when multiple full pages are requested, they are all returned, and in full.
        """

        pages = [ 'Olympique Lyonnais', 'Borussia Dortmund' ]
        extracts = text.collect(pages)
        self.assertEqual(2, len(extracts))
        self.assertEqual(set(pages), set(list(extracts.keys())))
        self.assertTrue(all(len(text) for text in extracts.values()))

    def test_get_introduction_only(self):
        """
        Test that when only the introduction is requested, it is returned.
        """

        page = 'Olympique Lyon'
        introduction = text.collect(page, introduction_only=True)
        extract = text.collect(page, introduction_only=False)
        self.assertLess(len(introduction[page]), len(extract[page]))

    def test_get_multiple_introductions(self):
        """
        Test that when multiple introductions are requested, they are all returned.
        """

        pages = [ 'Olympique Lyonnais', 'Borussia Dortmund' ]
        extracts = text.collect(pages, introduction_only=True)
        self.assertEqual(2, len(extracts))
        self.assertEqual(set(pages), set(list(extracts.keys())))
        self.assertTrue(all(len(text) for text in extracts.values()))

    def test_get_page_with_accent(self):
        """
        Test that pages that contain an accent in their title are retrieved normally.
        """

        page = 'Ciprian Tătărușanu'
        extracts = text.collect(page, introduction_only=True)
        self.assertEqual(1, len(extracts))
        self.assertTrue(page in extracts)
        self.assertGreater(len(extracts[page]), 100)

    def test_get_long_list(self):
        """
        Test that when getting a long list (greater than the stagger value), all pages are retrieed.
        """

        pages = [ 'Anthony Lopes', 'Mapou Yanga-Mbiwa', 'Joachim Andersen',
                  'Rafael',  'Jason Denayer', 'Marcelo', 'Martin Terrier',
                  'Houssem Aouar',  'Moussa Dembélé', 'Bertrand Traoré',
                  'Memphis Depay', 'Thiago Mendes', 'Léo Dubois', 'Oumar Solet',
                  'Jeff Reine-Adélaïde', 'Rayan Cherki', 'Bruno Guimarães',
                  'Amine Gouiri', 'Marçal', 'Karl Toko Ekambi', 'Jean Lucas',
                  'Kenny Tete', 'Maxence Caqueret', 'Camilo Reijers de Oliveira',
                  'Maxwel Cornet', 'Youssouf Koné', 'Lucas Tousart',
                  'Ciprian Tătărușanu', 'Boubacar Fofana']

        extracts = text.collect(pages, introduction_only=True)
        self.assertEqual(len(pages), len(extracts))
        self.assertEqual(set(pages), set(list(extracts.keys())))
        self.assertTrue(all(len(text) > 100 for text in extracts.values()))
