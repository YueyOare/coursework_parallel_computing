import os
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
        except Exception as e:
            raise RuntimeError(f"Error loading model or tokenizer: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Генерація ембеддінгу для заданого тексту.

        :param text: Текст для генерації ембеддінгу.
        :return: Ембеддінг у вигляді numpy масиву.
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
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
        """
        if os.path.exists(self.books_file):
            try:
                books_df = pd.read_csv(self.books_file)
                for _, row in books_df.iterrows():
                    self.books[row['id']] = row.to_dict()
            except Exception as e:
                raise RuntimeError(f"Error loading books: {e}")

    def load_embeddings(self) -> None:
        """
        Завантаження ембеддінгів з CSV файлу.

        Якщо файл не існує, ембеддінги не завантажуються.
        """
        if os.path.exists(self.embeddings_file):
            try:
                embeddings_df = pd.read_csv(self.embeddings_file)
                for _, row in embeddings_df.iterrows():
                    self.embeddings[row['id']] = np.fromstring(row['embedding'].strip("[]"), sep=" ")
            except Exception as e:
                raise RuntimeError(f"Error loading embeddings: {e}")

    def load_index(self) -> None:
        """
        Завантаження інвертованого індексу з CSV файлу, або його створення, якщо файл не існує.
        """
        if os.path.exists(self.index_file):
            try:
                index_df = pd.read_csv(self.index_file)
                for _, row in index_df.iterrows():
                    self.index[row['token']] = set(map(int, row['doc_ids'].split(',')))
            except Exception as e:
                raise RuntimeError(f"Error loading index: {e}")
        else:
            self.create_index()

    def save_books(self) -> None:
        """
        Збереження книг у CSV файл.
        """
        try:
            books_data = [{"id": doc_id, **book_data} for doc_id, book_data in self.books.items()]
            pd.DataFrame(books_data).to_csv(self.books_file, index=False)
        except Exception as e:
            raise RuntimeError(f"Error saving books: {e}")

    def save_embeddings(self) -> None:
        """
        Збереження ембеддінгів у CSV файл.
        """
        try:
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
            index_data = [{"token": token, "doc_ids": ','.join(map(str, doc_ids))} for token, doc_ids in
                          self.index.items()]
            pd.DataFrame(index_data).to_csv(self.index_file, index=False)
        except Exception as e:
            raise RuntimeError(f"Error saving index: {e}")

    def create_index(self) -> None:
        """
        Створення інвертованого індексу на основі описів книг.
        """
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
        :param given_id: Id з яким треба зберегти книгу (за замовчанням встановлюється рівним останньому значенню в базі +1).
        :return: Ідентифікатор доданої книги.
        """
        try:
            doc_id = len(self.books) if given_id is None else given_id
            default_book = {"title": "", "author": "", "description": "", "isbn": 0, "publish_year": 0, "genre": ""}
            book_data = {key: new_book.get(key, default_book[key]) for key in default_book}
            self.books[doc_id] = book_data
            description = format_book_description(new_book)
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
                description = format_book_description(self.books[doc_id])
                tokens = self.tokenize(description)
                for token in tokens:
                    self.index[token].discard(doc_id)
                    if not self.index[token]:
                        del self.index[token]
                del self.books[doc_id]
                del self.embeddings[doc_id]
                self.save_books()
                self.save_index()
                self.save_embeddings()
            else:
                raise ValueError(f"Book with ID {doc_id} not found.")
        except Exception as e:
            raise e

    def update_book(self, doc_id: int, updated_data: Dict[str, Union[str, int]]) -> None:
        """
        Оновлення інформації про книгу.

        :param doc_id: Ідентифікатор книги для оновлення.
        :param updated_data: Нові дані для оновлення.
        """
        try:
            if doc_id in self.books:
                current_data = self.books[doc_id]
                new_data = {key: updated_data.get(key, current_data[key]) for key in current_data}
                self.delete_book(doc_id)
                self.add_book(new_data, doc_id)
            else:
                raise ValueError(f"Book with ID {doc_id} not found.")
        except Exception as e:
            raise e


if __name__ == "__main__":

    embedding_generator = BookEmbeddings()
    index = InvertedIndex(embedding_generator, books_file="books_database_with_descriptions.csv")

    test_query = "story about genius detective"
    tokens = index.tokenize(test_query)
    print(tokens)

    # Додавання нової книги
    book_id = index.add_book(
        {"title": "New Book", "author": "Author Name", "description": "A fascinating tale.", "isbn": "12345",
         "publish_year": 2024, "genre": "Fiction"})
    #
    # # Оновлення книги
    index.update_book(book_id, {"title": "Updated Book"})

    # Видалення книги
    index.delete_book(book_id)

    # data = pd.read_csv('books_database_with_descriptions.csv')
    # for i, _ in data.iterrows():
    #     data.loc[i, 'id'] = i
    #
    # data['id'].astype(int)
    # data.to_csv('books_database_with_descriptions.csv', index=False)