import unittest
from inverted_index import InvertedIndex


class TestInvertedIndex(unittest.TestCase):
    def test_basic_search(self):
        index = InvertedIndex()
        index.add_document(1, "The Great Gatsby is a novel")
        index.add_document(2, "A great detective story")
        index.add_document(3, "Gatsby investigates a crime")

        # Тест пошуку
        self.assertEqual(index.search("great"), {1, 2})
        self.assertEqual(index.search("gatsby"), {1, 3})
        self.assertEqual(index.search("crime"), {3})
        self.assertEqual(index.search("nonexistent"), set())

    def test_tokenization(self):
        index = InvertedIndex()
        index.add_document(1, "The Great Gatsby, by F. Scott Fitzgerald!")
        index.add_document(2, "A GREAT detective story.")

        # Тест пошуку з урахуванням токенізації
        self.assertEqual(index.search("great"), {1, 2})
        self.assertEqual(index.search("gatsby"), {1})
        self.assertEqual(index.search("detective"), {2})
        self.assertEqual(index.search("fitzgerald"), {1})

    def test_ranking(self):
        index = InvertedIndex()
        index.add_document(1, "The Great Gatsby is a novel about Gatsby.")
        index.add_document(2, "A great detective story")
        index.add_document(3, "Gatsby investigates a crime in this novel")

        # Тест пошуку з ранжуванням
        results = index.search_with_ranking("gatsby novel")
        self.assertEqual(results, [(1, 2), (3, 2)])
