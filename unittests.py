import unittest

from inverted_index import *


class TestBookEmbeddings(unittest.TestCase):
    def setUp(self):
        """Налаштування для тестів"""
        self.embedding_generator = BookEmbeddings()

    def test_get_embedding(self):
        """Тестування методу генерації ембеддінгу"""
        text = "Sample book description"
        embedding = self.embedding_generator.get_embedding(text)
        self.assertEqual(len(embedding), 768)


class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        """Налаштування для тестів"""
        self.embedding_generator = BookEmbeddings()
        self.inverted_index = InvertedIndex(self.embedding_generator, index_file="test_index.csv",
                                            books_file="test_books.csv", embeddings_file="test_embeddings.csv")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test_index.csv'):
            os.remove('test_index.csv')
        if os.path.exists('test_books.csv'):
            os.remove('test_books.csv')
        if os.path.exists('test_embeddings.csv'):
            os.remove('test_embeddings.csv')

    def test_add_book(self):
        """Тестування додавання книги до індексу"""
        book = {
            "title": "Book Title",
            "author": "Author Name",
            "description": "This is a sample description for the book.",
            "isbn": 1234567890,
            "publish_year": 2020,
            "genre": "Fiction"
        }
        book_id = self.inverted_index.add_book(book)
        self.assertIn(book_id, self.inverted_index.get_books())

    def test_delete_book(self):
        """Тестування видалення книги з індексу"""
        book = {
            "title": "Book Title",
            "author": "Author Name",
            "description": "This is a sample description for the book.",
            "isbn": 1234567890,
            "publish_year": 2020,
            "genre": "Fiction"
        }
        book_id = self.inverted_index.add_book(book)
        self.inverted_index.delete_book(book_id)
        self.assertNotIn(book_id, self.inverted_index.get_books())

    def test_update_book(self):
        """Тестування оновлення книги в індексі"""
        book = {
            "title": "Book Title",
            "author": "Author Name",
            "description": "This is a sample description for the book.",
            "isbn": 1234567890,
            "publish_year": 2020,
            "genre": "Fiction"
        }
        book_id = self.inverted_index.add_book(book)
        updated_data = {
            "title": "Updated Book Title",
            "author": "Updated Author Name",
            "description": "Updated description of the book."
        }
        self.inverted_index.update_book(book_id, updated_data)
        books = self.inverted_index.get_books()
        self.assertEqual(books[book_id]['title'], "Updated Book Title")
        self.assertEqual(books[book_id]['author'], "Updated Author Name")

    def test_search_books(self):
        """Тестування пошуку книг за запитом"""
        self.inverted_index.books = {}
        self.inverted_index.index = defaultdict(set)
        self.inverted_index.embeddings = {}
        book1 = {
            "title": "Book One",
            "author": "Author One",
            "description": "This is a book about Python programming.",
            "isbn": 1234567890,
            "publish_year": 2020,
            "genre": "Programming"
        }
        book2 = {
            "title": "Book Two",
            "author": "Author Two",
            "description": "This is a book about Machine Learning.",
            "isbn": 1234567891,
            "publish_year": 2021,
            "genre": "Technology"
        }
        self.inverted_index.add_book(book1)
        self.inverted_index.add_book(book2)

        result = self.inverted_index.search_books("Python")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['book_information']['title'], "Book One")

    def test_concurrent_updates_with_thread_pool(self):
        self.tearDownClass()
        self.setUp()

        """Тестування з пулом потоків для одночасних оновлень"""
        def add_book():
            book = {
                "title": "Concurrent Book",
                "author": "Author Name",
                "description": "This is a book added concurrently.",
                "isbn": 1234567890,
                "publish_year": 2020,
                "genre": "Fiction"
            }
            self.inverted_index.add_book(book)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_book) for _ in range(10)]
            for future in as_completed(futures):
                future.result()

        books = self.inverted_index.get_books()
        self.assertEqual(len(books), 10)

if __name__ == "__main__":
    unittest.main()