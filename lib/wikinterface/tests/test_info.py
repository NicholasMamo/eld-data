"""
Run unit tests on the info module.
"""

import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), '..', '..')
if path not in sys.path:
    sys.path.append(path)

from wikinterface import info

class TestInfo(unittest.TestCase):
    """
    Test the info module.
    """

    def test_get_type_no_pages(self):
        """
        Test getting the type of a simple page.
        """

        type = info.types([ ])
        self.assertEqual({ }, type)

    def test_get_type(self):
        """
        Test getting the type of a simple page.
        """

        page = 'Rudi Garcia'
        type = info.types(page)
        self.assertEqual(info.ArticleType.NORMAL, type[page])

    def test_get_type_missing_page(self):
        """
        Test getting the type of a page that doesn't exist.
        """

        page = 'Rudi Garcia (French coach)'
        type = info.types(page)
        self.assertEqual(info.ArticleType.MISSING, type[page])

    def test_get_type_disambiguation_page(self):
        """
        Test getting the type of a page that is actually a disambiguation.
        """

        page = 'Garcia'
        type = info.types(page)
        self.assertEqual(info.ArticleType.DISAMBIGUATION, type[page])

    def test_get_type_redirect(self):
        """
        Test getting the type of a page that redirects returns the input page.
        """

        page = 'Olympique Lyon'
        type = info.types(page)
        self.assertEqual(info.ArticleType.NORMAL, type[page])

    def test_get_type_list(self):
        """
        Test getting the type of a page that is a list of articles.
        """

        page = 'List of works by Michelangelo'
        type = info.types(page)
        self.assertEqual(info.ArticleType.LIST, type[page])

    def test_get_type_multiple_pages(self):
        """
        Test getting the types of multiple pages returns information about all pages.
        """

        pages = [ 'Bordeaux', 'Lyon' ]
        types = info.types(pages)
        self.assertEqual(len(pages), len(types))
        self.assertEqual(set(pages), set(list(types.keys())))
        self.assertTrue(all(type == info.ArticleType.NORMAL for type in types.values()))

    def test_get_type_many_pages(self):
        """
        Test getting the types of many pages returns (more than the stagger value) information about all pages.
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
        types = info.types(pages)
        self.assertEqual(len(pages), len(types))
        self.assertEqual(set(pages), set(list(types.keys())))
        self.assertEqual(info.ArticleType.DISAMBIGUATION, types['Rafael'])

    def test_is_person_no_articles(self):
        """
        Test that when no articles are given, an empty dictionary is returned.
        """

        self.assertFalse(info.is_person([ ]))

    def test_is_person(self):
        """
        Test checking whether an individual article is about a person.
        """

        page = 'Rudi Garcia'
        classes = info.is_person(page)
        self.assertTrue(classes[page])

    def test_is_person_pattern2(self):
        """
        Test that an article with pattern 2 is correctly captured as a person.
        """

        page = 'Dario Benedetto'
        classes = info.is_person(page)
        self.assertTrue(classes[page])

    def test_is_person_pattern4(self):
        """
        Test that an article with pattern 2 is correctly captured as a person.
        """

        page = 'Valentina Shevchenko (fighter)'
        classes = info.is_person(page)
        self.assertTrue(classes[page])

    def test_is_person_multiple_pages(self):
        """
        Test checking whether a number of articles are about persons.
        """

        pages = [ 'Lyon', 'Bordeaux' ]
        classes = info.is_person(pages)
        self.assertEqual(len(pages), len(classes))
        self.assertFalse(any(classes[page] for page in pages))

    def test_is_person_with_redirects(self):
        """
        Test checking whether an article that redirects still checks that the subject is a person.
        """

        page = 'Messi'
        classes = info.is_person(page)
        self.assertTrue(page in classes)
        self.assertEqual(2, len(classes))
        self.assertTrue(all(classes.values()))

    def test_is_person_many_pages(self):
        """
        Test checking whether many pages are about persons.
        """

        pages = [ 'Anthony Lopes', 'Mapou Yanga-Mbiwa', 'Joachim Andersen (footballer)',
                  'Rafael (footballer, born 1990)',  'Jason Denayer',
                  'Houssem Aouar',  'Moussa Dembélé (French_footballer)',
                  'Memphis Depay', 'Thiago Mendes', 'Léo Dubois', 'Oumar Solet',
                  'Jeff Reine-Adélaïde', 'Rayan Cherki', 'Bruno Guimarães',
                  'Amine Gouiri', 'Fernando Marçal', 'Karl Toko Ekambi',
                  'Kenny Tete', 'Maxence Caqueret', 'Camilo Reijers de Oliveira',
                  'Maxwel Cornet', 'Youssouf Koné (footballer, born 1995)',
                  'Ciprian Tătărușanu', 'Boubacar Fofana', 'Bertrand Traoré',
                  'Martin Terrier', 'Marcelo (footballer, born 1987)',
                  'Jean Lucas Oliveira', 'Lucas Tousart' ]

        classes = info.is_person(pages)
        self.assertEqual(len(pages), len(classes))
        self.assertTrue(all(classes.values()))

    def test_is_person_with_accent(self):
        """
        Test checking whether a page that has an accent in it can still be assessed.
        """

        page = 'Jeff Reine-Adélaïde'
        classes = info.is_person(page)
        self.assertEqual(1, len(classes))
        self.assertTrue(all(classes.values()))
