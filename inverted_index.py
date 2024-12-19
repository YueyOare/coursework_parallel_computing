import os
import threading
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pybloom_live import BloomFilter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
from sklearn.preprocessing import normalize
from transformers import DistilBertModel, DistilBertTokenizer


class BookEmbeddings:
    """
    Клас для генерації ембеддінгів книг за допомогою моделі DistilBERT.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        """
        Ініціалізація моделі та токенізатора DistilBERT.

        :param model_name: Назва моделі для завантаження. За замовчуванням "distilbert-base-uncased".
        """
        try:
            self.model = DistilBertModel.from_pretrained(model_name)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model.eval()
            self.lock = threading.Lock()
        except Exception as e:
            raise RuntimeError(f"Error loading model or tokenizer: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Генерація ембеддінгу для заданого тексту.

        :param text: Текст для генерації ембеддінгу.
        :return: Ембеддінг у вигляді numpy масиву.
        """
        try:
            with self.lock:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embedding_norm = normalize([embedding])[0]
                return embedding_norm
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}")


def format_book_description(book: Dict[str, Union[str, int]]) -> str:
    """
    Формує опис книги з її атрибутів.

    :param book: Словник з даними книги.
    :return: Форматований опис книги.
    """
    return f"{book['title']} {book['author']} {book['description']}"


class InvertedIndex:
    """
    Клас для побудови та управління інвертованим індексом для книг, що зберігаються у CSV.
    Включає методи для додавання, видалення та оновлення книг, а також пошуку по індексах.
    """

    def __init__(
            self,
            embedding_generator: BookEmbeddings,
            index_file: str = "inverted_index.csv",
            books_file: str = "books.csv",
            embeddings_file: str = "books_features.csv",
    ) -> None:
        """
        Ініціалізація індексу та завантаження даних.

        :param embedding_generator: Об'єкт для генерації ембеддінгів для тексту книг.
        :param index_file: Файл для зберігання інвертованого індексу.
        :param books_file: Файл для зберігання інформації про книги.
        :param embeddings_file: Файл для зберігання ембеддінгів.
        """
        self.index: Dict[str, set[int]] = defaultdict(set)
        self.books: Dict[int, Dict[str, Union[str, int]]] = {}
        self.embeddings: Dict[int, np.ndarray] = {}
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.embedding_generator = embedding_generator
        self.tokenizer = embedding_generator.tokenizer
        self.index_file = index_file
        self.books_file = books_file
        self.embeddings_file = embeddings_file
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)

        self.lock = threading.Lock()
        self.file_lock = threading.Lock()

        try:
            self.load_data()
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def load_data(self) -> None:
        """
        Завантаження книг, ембеддінгів та індексу з файлів.
        """
        self.load_books()
        self.load_embeddings()
        self.load_index()

    def load_books(self) -> None:
        """
        Завантаження даних книг з CSV файлу.

        Якщо файл не існує, книги не завантажуються.
        Якщо є кілька книг з однаковою назвою, автором та роком, але різними жанрами, вони об'єднуються в одну.
        """
        if os.path.exists(self.books_file):
            try:
                books_df = pd.read_csv(self.books_file)
                seen_books = {}

                for _, row in books_df.iterrows():
                    book_key = (row['title'], row['author'], row['publish_year'])
                    if book_key in seen_books:
                        seen_books[book_key]['genre'] += f", {str(row['genre'])}"
                    else:
                        seen_books[book_key] = row.to_dict()
                        seen_books[book_key]['genre'] = str(seen_books[book_key]['genre'])

                for new_id, (key, book_data) in enumerate(seen_books.items()):
                    book_data['id'] = new_id
                    self.books[new_id] = book_data

            except Exception as e:
                raise RuntimeError(f"Error loading books: {e}")

    def load_embeddings(self) -> None:
        """
        Завантаження ембеддінгів з CSV файлу.

        Якщо файл не існує або кількість ембеддінгів не відповідає кількості книг,
        ембеддінги перегенеровуються та зберігаються.
        """
        if os.path.exists(self.embeddings_file):
            try:
                embeddings_df = pd.read_csv(self.embeddings_file)
                for _, row in embeddings_df.iterrows():
                    self.embeddings[row['id']] = np.fromstring(row['embedding'].strip("[]"), sep=" ")

                # Перевірка, чи кількість ембеддінгів відповідає кількості книг
                if len(self.embeddings) != len(self.books):
                    print("Mismatch between embeddings and books. Regenerating embeddings...")
                    self.recalculate_and_save_embeddings()
            except Exception as e:
                raise RuntimeError(f"Error loading embeddings: {e}")
        else:
            # Якщо файл не існує, генеруємо ембеддінги заново
            print("Embeddings file not found. Generating embeddings...")
            self.recalculate_and_save_embeddings()

    def recalculate_and_save_embeddings(self) -> None:
        """
        Перегенерація ембеддінгів для всіх книг і збереження їх у файл.
        """
        try:
            with self.lock:
                self.embeddings = {}
                for doc_id, book in self.books.items():
                    description = format_book_description(book)
                    self.embeddings[doc_id] = self.embedding_generator.get_embedding(description)
            self.save_embeddings()
            print("Embeddings successfully regenerated and saved.")
        except Exception as e:
            raise RuntimeError(f"Error recalculating embeddings: {e}")

    def load_index(self) -> None:
        """
        Завантаження інвертованого індексу з CSV файлу, або його створення, якщо файл не існує.
        """
        if os.path.exists(self.index_file):
            try:
                index_df = pd.read_csv(self.index_file)
                for _, row in index_df.iterrows():
                    if type(row['doc_ids']) == str:
                        self.index[row['token']] = set(map(int, row['doc_ids'].split(',')))
                        doc_ids = set()
                        for doc_id in self.index[row['token']]:
                            if doc_id in self.books:
                                doc_ids.add(doc_id)
                        self.index[row['token']] = doc_ids
                    else:
                        self.index[row['token']] = set()
                    self.bloom_filter.add(row['token'])
            except Exception as e:
                raise RuntimeError(f"Error loading index: {e}")
        else:
            self.create_index()

    def save_books(self) -> None:
        """
        Збереження книг у CSV файл.
        """
        try:
            with self.file_lock:
                books_data = [{"id": doc_id, **book_data} for doc_id, book_data in self.books.items()]
                pd.DataFrame(books_data).to_csv(self.books_file, index=False)
        except Exception as e:
            raise RuntimeError(f"Error saving books: {e}")

    def save_embeddings(self) -> None:
        """
        Збереження ембеддінгів у CSV файл.
        """
        try:
            with self.file_lock:
                embeddings_data = [{"id": int(doc_id), "embedding": embedding} for doc_id, embedding in
                                   self.embeddings.items()]
                pd.DataFrame(embeddings_data).to_csv(self.embeddings_file, index=False)
        except Exception as e:
            raise RuntimeError(f"Error saving embeddings: {e}")

    def save_index(self) -> None:
        """
        Збереження інвертованого індексу у CSV файл.
        """
        try:
            with self.file_lock:
                index_data = [{"token": token, "doc_ids": ','.join(map(str, doc_ids))} for token, doc_ids in
                              self.index.items()]
                pd.DataFrame(index_data).to_csv(self.index_file, index=False)
        except Exception as e:
            raise RuntimeError(f"Error saving index: {e}")

    def create_index(self) -> None:
        """
        Створення інвертованого індексу на основі описів книг.
        """
        with self.lock:
            for doc_id, book in self.books.items():
                description = format_book_description(book)
                self.add_tokens_to_index(doc_id, description)
        self.save_index()

    def tokenize(self, text: str) -> List[str]:
        """
        Токенізація та препроцесинг тексту (включає лематизацію та видалення стоп-слів).

        :param text: Текст для токенізації.
        :return: Список токенів.
        """
        tokens = self.tokenizer.tokenize(text.lower())
        return [self.lemmatizer.lemmatize(token) for token in tokens if
                token not in self.stop_words and token.isalpha()]

    def add_tokens_to_index(self, doc_id: int, description: str) -> None:
        """
        Додавання токенів з опису книги в інвертований індекс.

        :param doc_id: Ідентифікатор книги.
        :param description: Опис книги для токенізації.
        """
        tokens = self.tokenize(description)
        for token in tokens:
            self.bloom_filter.add(token)
            self.index[token].add(int(doc_id))

    def update_embeddings(self, doc_id: int, description: str) -> None:
        """
        Оновлення ембеддінгу для книги.

        :param doc_id: Ідентифікатор книги.
        :param description: Опис книги для генерації нового ембеддінгу.
        """
        try:
            self.embeddings[doc_id] = self.embedding_generator.get_embedding(description)
        except Exception as e:
            raise RuntimeError(f"Error updating embedding: {e}")

    def add_book(self, new_book: Dict[str, Union[str, int]], given_id=None) -> int:
        """
        Додавання нової книги до індексу та ембеддінгів.

        :param new_book: Словник з атрибутами нової книги.
        :param given_id: Id з яким треба зберегти книгу (за замовчуванням встановлюється рівним останньому значенню в базі +1).
        :return: Ідентифікатор доданої книги.
        """
        try:
            with self.lock:
                doc_id = len(self.books) if given_id is None else given_id
                default_book = {"title": "", "author": "", "description": "", "isbn": 0, "publish_year": 0, "genre": ""}
                book_data = {key: new_book.get(key, default_book[key]) for key in default_book}
                self.books[doc_id] = book_data
                description = format_book_description(book_data)
                self.add_tokens_to_index(doc_id, description)
                self.update_embeddings(doc_id, description)
            self.save_books()
            self.save_index()
            self.save_embeddings()
            return doc_id
        except Exception as e:
            raise e

    def delete_book(self, doc_id: int) -> None:
        """
        Видалення книги з індексу та ембеддінгів.

        :param doc_id: Ідентифікатор книги для видалення.
        """
        try:
            if doc_id in self.books:
                with self.lock:
                    description = format_book_description(self.books[doc_id])
                    tokens = self.tokenize(description)
                    for token in tokens:
                        self.index[token].discard(doc_id)
                        if not self.index[token]:
                            del self.index[token]
                    del self.books[doc_id]
                    del self.embeddings[doc_id]
                    self.regenerate_bloom_filter()
                self.save_books()
                self.save_index()
                self.save_embeddings()
            else:
                raise ValueError(f"Book with ID {doc_id} not found.")
        except Exception as e:
            raise e

    def regenerate_bloom_filter(self) -> None:
        """
        Перегенерация Bloom Filter после удаления книги.
        """
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)
        # Добавляем все токены обратно в Bloom Filter
        for doc_id, book in self.books.items():
            description = format_book_description(book)
            tokens = self.tokenize(description)
            for token in tokens:
                self.bloom_filter.add(token)

    def update_book(self, doc_id: int, updated_data: Dict[str, Union[str, int]]) -> None:
        """
        Оновлення інформації про книгу.

        :param doc_id: Ідентифікатор книги для оновлення.
        :param updated_data: Нові дані для оновлення.
        """
        try:
            if doc_id in self.books:
                with self.lock:
                    current_data = self.books[doc_id]
                    for key, value in updated_data.items():
                        current_data[key] = value
                    old_tokens = self.tokenize(format_book_description(self.books[doc_id]))
                    for token in old_tokens:
                        if doc_id in self.index.get(token, set()):
                            self.index[token].remove(doc_id)
                            if not self.index[token]:
                                del self.index[token]
                    self.books[doc_id] = current_data
                    self.add_tokens_to_index(doc_id, format_book_description(current_data))
                    self.update_embeddings(doc_id, format_book_description(current_data))
                    self.regenerate_bloom_filter()
                self.save_books()
                self.save_index()
                self.save_embeddings()
            else:
                raise ValueError(f"Book with ID {doc_id} not found.")
        except Exception as e:
            raise e

    def search_books(self, query: str, top_n: int = 100) -> List[
        Dict[str, Union[Dict[str, Union[str, int]], int, float]]]:
        """
        Пошук книг, що відповідають запиту.

        :param query: Запит для пошуку.
        :param top_n: Кількість повертаємих результатів.
        :return: Список книг, відсортованих за релевантністю.
        """
        try:
            tokens = self.tokenize(query)
            valid_tokens = [token for token in tokens if token in self.bloom_filter]
            if not valid_tokens:
                return []
            doc_ids = set()
            for token in valid_tokens:
                doc_ids.update(self.index.get(token, set()))
            query_embedding = self.embedding_generator.get_embedding(query)
            embeddings = [self.embeddings[book_id] for book_id in doc_ids]

            similarities = cosine_similarity([query_embedding], embeddings).flatten()
            # similarities = euclidean([query_embedding], embeddings).flatten()

            sorted_books = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
            return [{'book_information': self.books[book_id], 'book_id': book_id, 'similarity': similarity} for
                    book_id, similarity in sorted_books[:top_n]]
        except Exception as e:
            raise RuntimeError(f"Error searching for books: {e}")


if __name__ == "__main__":
    embedding_generator = BookEmbeddings()
    index = InvertedIndex(embedding_generator, books_file="books_database_with_descriptions.csv")

    test_query = "story about genius detective"
    tokens = index.tokenize(test_query)
    print(tokens)

    book_id = index.add_book(
        {"title": "New Book", "author": "Author Name", "description": "A fascinating tale.", "isbn": "12345",
         "publish_year": 2024, "genre": "Fiction"})

    index.update_book(book_id, {"title": "Updated Book"})

    index.delete_book(book_id)

    result = index.search_books(test_query)
    print(result)

    # data = pd.read_csv('books_database_with_descriptions.csv')
    # for i, _ in data.iterrows():
    #     data.loc[i, 'id'] = i
    #
    # data['id'].astype(int)
    # data.to_csv('books_database_with_descriptions.csv', index=False)
