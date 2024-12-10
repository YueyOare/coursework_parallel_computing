import time
import re
import random
import pandas as pd
from fuzzywuzzy import fuzz
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from main import get_api_key

# Створення клієнта для роботи з API моделі
client = InferenceClient(api_key=get_api_key())

# Завантаження бази даних книг і жанрів
try:
    books_df = pd.read_csv('books_features.csv')
    genre_mapping = pd.read_csv('genre_mapping.csv')
except Exception as e:
    print(f"Помилка при завантаженні даних: {e}")
    books_df = None

genres_list = [
    'fiction', 'science', 'history', 'fantasy', 'mystery',
    'romance', 'biography', 'self-help', 'cookbook', 'travel',
    'children', 'young adult', 'horror', 'philosophy', 'poetry',
    'classic literature', 'graphic novels', 'science fiction', 'adventure', 'true crime'
]
genres_str = ", ".join(genres_list)

# Функція для аналізу запиту користувача
def analyze_user_query(query, titles, authors, genres, keywords, max_retries=50, retry_delay=5):
    print(f"----Query: {query}----")

    # Визначення ймовірності вибору моделі
    models = [
        ("mistralai/Mistral-Nemo-Instruct-2407", 0.75),
        ("mistralai/Mistral-7B-Instruct-v0.3", 0.25),
        ("google/gemma-2-2b-it", 0.0)
    ]

    messages = [
        {
            "role": "user",
            "content": (
                f"Please translate the following query to English and extract the relevant information:\n"
                f"1. Book title (if any, otherwise 'Not provided')\n"
                f"2. Author name (if any, otherwise 'Not provided')\n"
                f"3. Possible genres (only from the following list: {genres_str})\n"
                f"4. Main keywords for book description in English\n\n"
                f"Text: {query}\n\n"
                "Please respond only in English in any case, and ensure to translate the keywords to English if needed. "
                "Follow the specified format, and do not generate book titles or author names if they are not explicitly mentioned in the text. In that case say 'Not provided'. "
                "The response format should be:\n1. Book title: <title>\n2. Author name: <author>\n"
                "3. Genres: <genres>\n4. Keywords: <keywords>"
            )
        }
    ]

    print('Йде обробка, зачекайте...')
    for attempt in range(max_retries):
        print(f'Спроба {attempt + 1}/{max_retries}')

        # Вибір моделі з ймовірністю
        chosen_model = random.choices(
            [model for model, _ in models],
            [probability for _, probability in models]
        )[0]

        print(f"Вибрано модель: {chosen_model}")

        try:
            stream = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                max_tokens=500,
                stream=True,
                temperature=0.2
            )
        except Exception as e:
            print(f"Помилка при зверненні до API: {e}")
            time.sleep(retry_delay)
            continue

        result_text = ""
        for chunk in stream:
            if hasattr(chunk, 'error') and chunk.error:
                print(f"Error: {chunk.error}")
                time.sleep(retry_delay)
                continue

            result_text += chunk.choices[0].delta.content

        if not result_text.strip():
            print('Не вдалося отримати відповідь, пробую ще раз')
            time.sleep(retry_delay)
            continue

        # Перевірка на відповідність формату та наявність кирилиці
        print(f"----Result: {result_text}----")
        if (not re.search('[а-яА-Я]', result_text)) and \
           all(key in result_text.lower() for key in ["book title", "author name", "genres", "keywords"]):
            try:
                result_text = result_text.lower()
                result_text_splitted = result_text.split('\n')

                # Оновлення значень із запиту
                new_title = result_text_splitted[0].split(": ")[1] if "book title" in result_text_splitted[0] else None
                if new_title and new_title != "not provided":
                    titles.add(new_title)

                new_author = result_text_splitted[1].split(": ")[1] if "author name" in result_text_splitted[1] else None
                if new_author and new_author != "not provided":
                    authors.add(new_author)

                try:
                    # Перевірка жанрів на наявність у списку допустимих жанрів
                    new_genres = {genre.strip() for genre in result_text_splitted[2].split(": ")[1].split(", ") if genre.strip()}
                    valid_genres = new_genres.intersection(genres_list)  # Залишаємо лише жанри, які є у списку genres_list
                    genres.update(valid_genres)
                except Exception as e:
                    print(f"Помилка при обробці відповіді моделі: {e}")

                new_keywords = result_text_splitted[3].split(": ")[1] if "keywords" in result_text_splitted[3] else ""
                if new_keywords:
                    keywords += f", {new_keywords}" if keywords else new_keywords

                return titles, authors, genres, keywords
            except Exception as e:
                print(f"Помилка при обробці відповіді моделі: {e}")
        else:
            print("Відповідь не відповідає формату або містить кирилицю, пробую ще раз")
            time.sleep(retry_delay)

    print("Max retries exceeded. Unable to complete the operation.")
    return titles, authors, genres, keywords

# Завантаження моделі для векторизації опису книг
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Помилка при завантаженні моделі векторизації: {e}")

# Функція для пошуку книг за отриманими характеристиками
def search_books(titles, authors, genres, keywords, books_df=books_df):
    try:
        filtered_books = books_df

        results = {}

        # Об'єднання всіх назв та авторів у строки для порівняння
        title_str = ", ".join(titles) if titles else ""
        author_str = ", ".join(authors) if authors else ""
        query_combined_text = "Title: " + title_str + ". Author: " + author_str + ". Keywords: " + keywords
        query_vector = embedding_model.encode([query_combined_text], convert_to_numpy=True)
        print(f"----Searched query: {query_combined_text}----")
        # Обчислення схожості для кожної книги
        for index, book in filtered_books.iterrows():
            book_title = str(book['title'])
            book_author = str(book['author'])

            # Обчислення максимальної схожості за назвами
            title_similarity = max(
                [fuzz.token_set_ratio(title.lower(), book_title.lower()) for title in titles],
                default=0
            ) if titles else 0

            # Обчислення максимальної схожості за авторами
            author_similarity = max(
                [fuzz.token_set_ratio(author.lower(), book_author.lower()) for author in authors],
                default=0
            ) if authors else 0

            # Векторизація опису книги для обчислення схожості
            book_vector = book[[f'feature_{i}' for i in range(query_vector.shape[1])]].values.reshape(1, -1)
            description_similarity = cosine_similarity(query_vector, book_vector)[0][0]

            # Перевірка, чи книга вже в результатах
            book_key = (book_title, book_author)
            if book_key in results:
                results[book_key]['genre'].append(book['genre'])
                results[book_key]['isbn'] += ", " + str(book['isbn'])
            else:
                results[book_key] = {
                    'title': book_title,
                    'author': book_author,
                    'genre': [book['genre']],
                    'isbn': str(book['isbn']),
                    'description': str(book['description']),
                    'title_similarity': title_similarity,
                    'author_similarity': author_similarity,
                    'description_similarity': description_similarity
                }

        # Конвертація результатів у список і сортування за схожістю
        results = list(results.values())
        results = sorted(
            results,
            key=lambda x: (x['title_similarity'], x['author_similarity'], x['description_similarity']),
            reverse=True
        )

        return results
    except Exception as e:
        print(f"Помилка при пошуку книг: {e}")
        return []