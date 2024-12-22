import logging
import os
import threading
from collections import defaultdict
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from colorama import Fore, Style, init
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pybloom_live import BloomFilter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import DistilBertModel, DistilBertTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

def log_info(message: str) -> None:
    logging.info(Fore.GREEN + message + Style.RESET_ALL)

def log_warning(message: str) -> None:
    logging.warning(Fore.YELLOW + message + Style.RESET_ALL)

def log_error(message: str) -> None:
    logging.error(Fore.RED + message + Style.RESET_ALL)

def check_nan(val: float | str | int) -> int:
    return str(val) == 'nan'

class BookEmbeddings:
    """
    Клас для генерації ембеддінгів книг за допомогою моделі DistilBERT.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        """
        Ініціалізація моделі та токенізатора DistilBERT.

        :param model_name: Назва моделі для завантаження. За замовчуванням "distilbert-base-uncased".
        :throws RuntimeError: Якщо виникла помилка при завантаженні моделі або токенізатора.
        """
        try:
            self.__model = DistilBertModel.from_pretrained(model_name)
            self.__tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.__model.eval()
            self.__lock = threading.Lock()
        except Exception as e:
            log_error(f"Помилка завантаження моделі або токенізатора: {e}")
            raise RuntimeError(f"Помилка завантаження моделі або токенізатора: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Генерація ембеддінгу для заданого тексту.

        :param text: Текст для генерації ембеддінгу.
        :return: Ембеддінг у вигляді numpy масиву.
        :throws RuntimeError: Якщо виникла помилка при генерації ембеддінгу.
        """
        try:
            with self.__lock:
                inputs = self.__tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.__model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embedding_norm = normalize([embedding])[0]
                return embedding_norm
        except Exception as e:
            log_error(f"Помилка генерації ембеддінгу: {e}")
            raise RuntimeError(f"Помилка генерації ембеддінгу: {e}")

    def get_tokenizer(self) -> DistilBertTokenizer:
        """
        Повертає токенізатор.

        :return: Об'єкт токенізатора.
        """
        return self.__tokenizer

def format_book_description(book: Dict[str, Union[str, int]]) -> str:
    """
    Формує опис книги з її атрибутів.

    :param book: Словник з даними книги.
    :return: Форматований опис книги.
    """
    return f"Title: {book['title']}, Author: {book['author']}, Description: {book['description']}"

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
            num_threads: int = 1,
    ) -> None:
        """
        Ініціалізація індексу та завантаження даних.

        :param embedding_generator: Об'єкт для генерації ембеддінгів для тексту книг.
        :param index_file: Файл для зберігання інвертованого індексу.
        :param books_file: Файл для зберігання інформації про книги.
        :param embeddings_file: Файл для зберігання ембеддінгів.
        :param num_threads: Кількість робочих потоків у ThreadPool.
        :throws RuntimeError: Якщо виникла помилка при завантаженні даних.
        """
        self.__index: Dict[str, set[int]] = defaultdict(set)
        self.__books: Dict[int, Dict[str, Union[str, int]]] = {}
        self.__embeddings: Dict[int, np.ndarray] = {}
        self.__stop_words = set(stopwords.words("english"))
        self.__lemmatizer = WordNetLemmatizer()
        wordnet.ensure_loaded()
        self.__embedding_generator = embedding_generator
        self.__tokenizer = embedding_generator.get_tokenizer()
        self.__index_file = index_file
        self.__books_file = books_file
        self.__embeddings_file = embeddings_file
        self.__bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)

        self.__lock = threading.Lock()
        self.__file_lock = threading.Lock()
        self.__thread_pool_lock = threading.Lock()

        self.__num_threads = num_threads
        self.__executor = ThreadPoolExecutor(max_workers=num_threads) if num_threads > 1 else None
        log_info(f"Використовується {"ThreadPool" if self.__executor else "послідовна обробка"}.")

        try:
            self.__load_data()
        except Exception as e:
            log_error(f"Помилка завантаження даних: {e}")
            raise RuntimeError(f"Помилка завантаження даних: {e}")

    def __load_data(self) -> None:
        """
        Завантаження книг, ембеддінгів та індексу з файлів.
        """
        self.__load_books()
        self.__load_embeddings()
        self.__load_index()

    def __load_books(self) -> None:
        """
        Завантаження даних книг з CSV файлу.

        Якщо файл не існує, книги не завантажуються.
        Якщо є кілька книг з однаковою назвою, автором та роком, але різними жанрами, вони об'єднуються в одну.
        :throws RuntimeError: Якщо виникла помилка при завантаженні книг.
        """
        if os.path.exists(self.__books_file):
            try:
                books_df = pd.read_csv(self.__books_file)
                seen_books = {}

                def process_row(row):
                    book_key = (row['title'] if not check_nan(row['title']) else None,
                                row['author'] if not check_nan(row['author']) else None,
                                row['publish_year'] if not check_nan(row['publish_year']) else None)
                    with self.__thread_pool_lock:
                        if book_key in seen_books:
                            seen_books[book_key]['genre'] += f", {str(row['genre'])}"
                        else:
                            seen_books[book_key] = row.to_dict()
                            seen_books[book_key]['genre'] = str(seen_books[book_key]['genre'])

                if self.__executor:
                    futures = [self.__executor.submit(process_row, row) for _, row in books_df.iterrows()]
                    for future in as_completed(futures):
                        future.result()
                else:
                    for _, row in books_df.iterrows():
                        process_row(row)

                for new_id, (key, book_data) in enumerate(seen_books.items()):
                    book_data['id'] = new_id
                    self.__books[new_id] = book_data

            except Exception as e:
                log_error(f"Помилка завантаження книг: {e}")
                raise RuntimeError(f"Помилка завантаження книг: {e}")

    def __load_embeddings(self) -> None:
        """
        Завантаження ембеддінгів з CSV файлу.

        Якщо файл не існує або кількість ембеддінгів не відповідає кількості книг,
        ембеддінги перегенеровуються та зберігаються.
        :throws RuntimeError: Якщо виникла помилка при завантаженні ембеддінгів.
        """
        if os.path.exists(self.__embeddings_file):
            try:
                embeddings_df = pd.read_csv(self.__embeddings_file)

                def process_row(row):
                    with self.__thread_pool_lock:
                        self.__embeddings[row['id']] = np.fromstring(row['embedding'].strip("[]"), sep=" ")

                if self.__executor:
                    futures = [self.__executor.submit(process_row, row) for _, row in embeddings_df.iterrows()]
                    for future in as_completed(futures):
                        future.result()
                else:
                    for _, row in embeddings_df.iterrows():
                        process_row(row)

                # Перевірка, чи кількість ембеддінгів відповідає кількості книг
                if len(self.__embeddings) != len(self.__books):
                    log_info("Невідповідність між ембеддінгами та книгами. Перегенерація ембеддінгів...")
                    self.__recalculate_and_save_embeddings()
            except Exception as e:
                log_error(f"Помилка завантаження ембеддінгів: {e}")
                raise RuntimeError(f"Помилка завантаження ембеддінгів: {e}")
        else:
            # Якщо файл не існує, генеруємо ембеддінги заново
            log_info("Файл ембеддінгів не знайдено. Генерування ембеддінгів...")
            self.__recalculate_and_save_embeddings()

    def __recalculate_and_save_embeddings(self) -> None:
        """
        Перегенерація ембеддінгів для всіх книг і збереження їх у файл.
        :throws RuntimeError: Якщо виникла помилка при перегенерації ембеддінгів.
        """
        try:
            with self.__lock:
                self.__embeddings = {}
                if self.__executor:
                    futures = [self.__executor.submit(self.__generate_embedding, doc_id, book) for doc_id, book in self.__books.items()]
                    for future in as_completed(futures):
                        with self.__thread_pool_lock:
                            doc_id, embedding = future.result()
                            self.__embeddings[doc_id] = embedding
                else:
                    for doc_id, book in self.__books.items():
                        doc_id, embedding = self.__generate_embedding(doc_id, book)
                        self.__embeddings[doc_id] = embedding
            self.__save_embeddings()
            log_info("Ембеддінги успішно перегенеровано та збережено.")
        except Exception as e:
            log_error(f"Помилка перегенерації ембеддінгів: {e}")
            raise RuntimeError(f"Помилка перегенерації ембеддінгів: {e}")

    def __generate_embedding(self, doc_id: int, book: Dict[str, Union[str, int]]) -> Tuple[int, np.ndarray]:
        """
        Допоміжний метод для генерації ембеддінгу для однієї книги.
        """
        description = format_book_description(book)
        embedding = self.__embedding_generator.get_embedding(description)
        return doc_id, embedding

    def __load_index(self) -> None:
        """
        Завантаження інвертованого індексу з CSV файлу, або його створення, якщо файл не існує.
        :throws RuntimeError: Якщо виникла помилка при завантаженні індексу.
        """
        if os.path.exists(self.__index_file):
            try:
                index_df = pd.read_csv(self.__index_file)

                def process_row(row):
                    with self.__thread_pool_lock:
                        if type(row['doc_ids']) == str:
                            self.__index[row['token']] = set(map(int, row['doc_ids'].split(',')))
                            doc_ids = set()
                            for doc_id in self.__index[row['token']]:
                                if doc_id in self.__books:
                                    doc_ids.add(doc_id)
                            self.__index[row['token']] = doc_ids
                        else:
                            self.__index[row['token']] = set()
                        self.__bloom_filter.add(row['token'])

                if self.__executor:
                    futures = [self.__executor.submit(process_row, row) for _, row in index_df.iterrows()]
                    for future in as_completed(futures):
                        future.result()
                else:
                    for _, row in index_df.iterrows():
                        process_row(row)
            except Exception as e:
                log_error(f"Помилка завантаження індексу: {e}")
                raise RuntimeError(f"Помилка завантаження індексу: {e}")
        else:
            self.__create_index()

    def __save_books(self) -> None:
        """
        Збереження книг у CSV файл.
        :throws RuntimeError: Якщо виникла помилка при збереженні книг.
        """
        try:
            with self.__file_lock:
                books_data = [{"id": doc_id, **book_data} for doc_id, book_data in self.__books.items()]
                pd.DataFrame(books_data).to_csv(self.__books_file, index=False)
        except Exception as e:
            log_error(f"Помилка збереження книг: {e}")
            raise RuntimeError(f"Помилка збереження книг: {e}")

    def __save_embeddings(self) -> None:
        """
        Збереження ембеддінгів у CSV файл.
        :throws RuntimeError: Якщо виникла помилка при збереженні ембеддінгів.
        """
        try:
            with self.__file_lock:
                embeddings_data = [{"id": int(doc_id), "embedding": embedding} for doc_id, embedding in
                                   self.__embeddings.items()]
                pd.DataFrame(embeddings_data).to_csv(self.__embeddings_file, index=False)
        except Exception as e:
            log_error(f"Помилка збереження ембеддінгів: {e}")
            raise RuntimeError(f"Помилка збереження ембеддінгів: {e}")

    def __save_index(self) -> None:
        """
        Збереження інвертованого індексу у CSV файл.
        :throws RuntimeError: Якщо виникла помилка при збереженні індексу.
        """
        try:
            with self.__file_lock:
                index_data = [{"token": token, "doc_ids": ','.join(map(str, doc_ids))} for token, doc_ids in
                              self.__index.items()]
                pd.DataFrame(index_data).to_csv(self.__index_file, index=False)
        except Exception as e:
            log_error(f"Помилка збереження індексу: {e}")
            raise RuntimeError(f"Помилка збереження індексу: {e}")

    def __create_index(self) -> None:
        """
        Створення інвертованого індексу на основі описів книг.
        """
        if self.__executor:
            futures = [self.__executor.submit(self.__add_tokens_to_index, doc_id, format_book_description(book)) for doc_id, book in
                       self.__books.items()]
            for future in as_completed(futures):
                future.result()
        else:
            for doc_id, book in self.__books.items():
                self.__add_tokens_to_index(doc_id, format_book_description(book))
        self.__save_index()

    def __tokenize(self, text: str) -> List[str]:
        """
        Токенізація та препроцесинг тексту (включає лематизацію та видалення стоп-слів).

        :param text: Текст для токенізації.
        :return: Список токенів.
        """
        tokens = self.__tokenizer.tokenize(text.lower())
        return [self.__lemmatizer.lemmatize(token) for token in tokens if
                token not in self.__stop_words and token.isalpha()]

    def __add_tokens_to_index(self, doc_id: int, description: str) -> None:
        """
        Додавання токенів з опису книги в інвертований індекс.

        :param doc_id: Ідентифікатор книги.
        :param description: Опис книги для токенізації.
        """
        tokens = self.__tokenize(description)
        with self.__thread_pool_lock:
            for token in tokens:
                self.__bloom_filter.add(token)
                self.__index[token].add(int(doc_id))

    def __update_embeddings(self, doc_id: int, description: str) -> None:
        """
        Оновлення ембеддінгу для книги.

        :param doc_id: Ідентифікатор книги.
        :param description: Опис книги для генерації нового ембеддінгу.
        :throws RuntimeError: Якщо виникла помилка при оновленні ембеддінгу.
        """
        try:
            self.__embeddings[doc_id] = self.__embedding_generator.get_embedding(description)
        except Exception as e:
            log_error(f"Помилка оновлення ембеддінгу: {e}")
            raise RuntimeError(f"Помилка оновлення ембеддінгу: {e}")

    def add_book(self, new_book: Dict[str, Union[str, int]], given_id=None) -> int:
        """
        Додавання нової книги до індексу та ембеддінгів.

        :param new_book: Словник з атрибутами нової книги.
        :param given_id: Id з яким треба зберегти книгу (за замовчуванням встановлюється рівним останньому значенню в базі +1).
        :return: Ідентифікатор доданої книги.
        :throws RuntimeError: Якщо виникла помилка при додаванні книги.
        """
        try:
            with self.__lock:
                doc_id = len(self.__books) if given_id is None else given_id
                default_book = {"title": "", "author": "", "description": "", "isbn": 0, "publish_year": 0, "genre": ""}
                book_data = {key: new_book.get(key, default_book[key]) for key in default_book}
                self.__books[doc_id] = book_data
                description = format_book_description(book_data)
                self.__add_tokens_to_index(doc_id, description)
                self.__update_embeddings(doc_id, description)
            self.__save_books()
            self.__save_index()
            self.__save_embeddings()
            return doc_id
        except Exception as e:
            log_error(f"Помилка додавання книги: {e}")
            raise e

    def delete_book(self, doc_id: int) -> None:
        """
        Видалення книги з індексу та ембеддінгів.

        :param doc_id: Ідентифікатор книги для видалення.
        :throws ValueError: Якщо книга з вказаним ідентифікатором не знайдена.
        :throws RuntimeError: Якщо виникла помилка при видаленні книги.
        """
        try:
            if doc_id in self.__books:
                with self.__lock:
                    description = format_book_description(self.__books[doc_id])
                    tokens = self.__tokenize(description)

                    def process_token(token):
                        with self.__thread_pool_lock:
                            self.__index[token].discard(doc_id)
                            if not self.__index[token]:
                                del self.__index[token]

                    if self.__executor:
                        futures = [self.__executor.submit(process_token, token) for token in tokens]
                        for future in as_completed(futures):
                            future.result()
                    else:
                        for token in tokens:
                            process_token(token)

                    del self.__books[doc_id]
                    del self.__embeddings[doc_id]
                    self.__regenerate_bloom_filter()
                self.__save_books()
                self.__save_index()
                self.__save_embeddings()
            else:
                log_error(f"Книга з ID {doc_id} не знайдена.")
                raise ValueError(f"Книга з ID {doc_id} не знайдена.")
        except Exception as e:
            log_error(f"Помилка видалення книги: {e}")
            raise e

    def __regenerate_bloom_filter(self) -> None:
        """
        Перегенерація Bloom Filter після видалення книги.
        """
        self.__bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)

        def process_book(book):
            description = format_book_description(book)
            tokens = self.__tokenize(description)
            with self.__thread_pool_lock:
                for token in tokens:
                    self.__bloom_filter.add(token)

        if self.__executor:
            futures = [self.__executor.submit(process_book, book) for doc_id, book in self.__books.items()]
            for future in as_completed(futures):
                future.result()
        else:
            for doc_id, book in self.__books.items():
                process_book(book)

    def update_book(self, doc_id: int, updated_data: Dict[str, Union[str, int]]) -> None:
        """
        Оновлення інформації про книгу.

        :param doc_id: Ідентифікатор книги для оновлення.
        :param updated_data: Нові дані для оновлення.
        :throws ValueError: Якщо книга з вказаним ідентифікатором не знайдена.
        :throws RuntimeError: Якщо виникла помилка при оновленні книги.
        """
        try:
            if doc_id in self.__books:
                with self.__lock:
                    current_data = self.__books[doc_id]
                    for key, value in updated_data.items():
                        current_data[key] = value
                    old_tokens = self.__tokenize(format_book_description(self.__books[doc_id]))

                    def process_old_token(token):
                        with self.__thread_pool_lock:
                            if doc_id in self.__index.get(token, set()):
                                self.__index[token].remove(doc_id)
                                if not self.__index[token]:
                                    del self.__index[token]

                    if self.__executor:
                        futures = [self.__executor.submit(process_old_token, token) for token in old_tokens]
                        for future in as_completed(futures):
                            future.result()
                    else:
                        for token in old_tokens:
                            process_old_token(token)

                    self.__books[doc_id] = current_data
                    self.__add_tokens_to_index(doc_id, format_book_description(current_data))
                    self.__update_embeddings(doc_id, format_book_description(current_data))
                    self.__regenerate_bloom_filter()
                self.__save_books()
                self.__save_index()
                self.__save_embeddings()
            else:
                log_error(f"Книга з ID {doc_id} не знайдена.")
                raise ValueError(f"Книга з ID {doc_id} не знайдена.")
        except Exception as e:
            log_error(f"Помилка оновлення книги: {e}")
            raise e

    def search_books(self, query: str, top_n: int = 100) -> List[
        Dict[str, Union[Dict[str, Union[str, int]], int, float]]]:
        """
        Пошук книг, що відповідають запиту.

        :param query: Запит для пошуку.
        :param top_n: Кількість повертаємих результатів.
        :return: Список книг, відсортованих за релевантністю.
        :throws RuntimeError: Якщо виникла помилка при пошуку книг.
        """
        try:
            tokens = self.__tokenize(query)
            valid_tokens = [token for token in tokens if token in self.__bloom_filter]
            if not valid_tokens:
                return []
            doc_ids = set()

            def process_token(token):
                with self.__thread_pool_lock:
                    doc_ids.update(self.__index.get(token, set()))

            if self.__executor:
                futures = [self.__executor.submit(process_token, token) for token in valid_tokens]
                for future in as_completed(futures):
                    future.result()
            else:
                for token in valid_tokens:
                    process_token(token)

            query_embedding = self.__embedding_generator.get_embedding(query)
            embeddings = [self.__embeddings[book_id] for book_id in doc_ids]

            similarities = cosine_similarity([query_embedding], embeddings).flatten()
            # similarities = euclidean([query_embedding], embeddings).flatten()

            sorted_books = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)
            return [{'book_information': self.__books[book_id], 'book_id': book_id, 'similarity': similarity} for
                    book_id, similarity in sorted_books[:top_n]]
        except Exception as e:
            log_error(f"Помилка пошуку книг: {e}")
            raise RuntimeError(f"Помилка пошуку книг: {e}")

    def get_books(self) -> Dict[int, Dict[str, Union[str, int]]]:
        """
        Повертає словник книг.

        :return: Словник книг.
        """
        return self.__books

    def get_embeddings(self) -> Dict[int, np.ndarray]:
        """
        Повертає словник ембеддінгів.

        :return: Словник ембеддінгів.
        """
        return self.__embeddings

    def get_index(self) -> Dict[str, set[int]]:
        """
        Повертає інвертований індекс.

        :return: Інвертований індекс.
        """
        return self.__index

    def get_tokenizer(self) -> DistilBertTokenizer:
        """
        Повертає токенізатор.

        :return: Об'єкт токенізатора.
        """
        return self.__tokenizer

if __name__ == "__main__":
    embedding_generator = BookEmbeddings()
    index = InvertedIndex(embedding_generator, books_file="books_database_with_descriptions.csv")

    test_query = "story about genius detective"

    book_id = index.add_book(
        {"title": "New Book", "author": "Author Name", "description": "A fascinating tale.", "isbn": "12345",
         "publish_year": 2024, "genre": "Fiction"})
    print(book_id)

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
