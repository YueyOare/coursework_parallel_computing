import gzip
import json
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from colorama import init, Fore
from jinja2 import Environment, FileSystemLoader

from inverted_index import BookEmbeddings, InvertedIndex

init(autoreset=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')


def log_info(message: str) -> None:
    logging.info(Fore.GREEN + message)


def log_warning(message: str) -> None:
    logging.warning(Fore.YELLOW + message)


def log_error(message: str) -> None:
    logging.error(Fore.RED + message)


HOST: str = '127.0.0.1'
PORT: int = 8080
TEMPLATES_DIR: str = 'templates'
QUERIES_TO_SHOW: int = 10

env: Environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
template = env.get_template("index.html")

embedding_generator: BookEmbeddings = BookEmbeddings()
index: InvertedIndex = InvertedIndex(embedding_generator, books_file="books_database_with_descriptions.csv", num_threads=2)

thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=8)


def compress_html(content: str) -> bytes:
    """
    Стискає HTML-контент за допомогою Gzip.

    :param content: HTML-контент у вигляді рядка.

    :return: Стиснутий контент у форматі Gzip.
    """
    return gzip.compress(content.encode('utf-8'))


def handle_request(client_socket: socket.socket) -> None:
    """
    Обробляє запити клієнтів.

    :param client_socket: Сокет клієнта.
    """
    with client_socket:
        client_ip: str = client_socket.getpeername()[0]
        try:
            request: str = client_socket.recv(2048).decode('utf-8')
            log_info(f"Отримано запит від {client_ip}: {request.splitlines()[0]}")

            if request.startswith('POST /search'):
                handle_post_request(client_socket, request)
            elif request.startswith('POST /admin'):
                handle_admin_post_request(client_socket, request)
            elif request.startswith('GET /admin HTTP/'):
                handle_get_admin(client_socket)
            elif request.startswith('GET / HTTP/'):
                handle_get_root(client_socket)
            else:
                log_warning(f"Непідтримуваний запит від {client_ip}: {request.splitlines()[0]}")
                client_socket.sendall("HTTP/1.1 404 Not Found\n\n".encode('utf-8'))
        except Exception as e:
            log_error(f"Помилка при обробці запиту від {client_ip}: {e}")


def handle_admin_post_request(client_socket: socket.socket, request: str) -> None:
    """
    Обробляє POST-запити для адміністративних дій.

    :param client_socket: Сокет клієнта.
    :param request: HTTP-запит у вигляді рядка.
    """
    try:
        content_length = int(
            [header.split(":")[1].strip() for header in request.split("\r\n") if "Content-Length" in header][0]
        )
        body = request.split("\r\n\r\n", 1)[1][:content_length]
        data = json.loads(body)

        action = data.get("action")
        book_data = data.get("book_data", [])
        book_ids = data.get("book_ids", [])

        if action == "view_books":
            books = index.get_books()
            books_list = [{"id": book_id, **book_info} for book_id, book_info in books.items()]
            response_data = {"books": books_list}
            log_info("Список книг успішно відправлено.")

        elif action == "add_book" and book_data:
            new_ids = [index.add_book(book) for book in book_data]
            response_data = {"message": f"Books with IDs {new_ids} added successfully."}
            log_info(f"Книги з ID {new_ids} успішно додані.")

        elif action == "edit_book" and book_ids and book_data:
            for book_id, book in zip(book_ids, book_data):
                index.update_book(int(book_id), book)
            response_data = {"message": f"Books with IDs {book_ids} updated successfully."}
            log_info(f"Книги з ID {book_ids} успішно оновлені.")

        elif action == "delete_book" and book_ids:
            for book_id in book_ids:
                index.delete_book(int(book_id))
            response_data = {"message": f"Books with IDs {book_ids} deleted successfully."}
            log_info(f"Книги з ID {book_ids} успішно видалені.")

        else:
            response_data = {"error": "Invalid action or missing data."}
            log_warning("Отримано недійсну дію або неповні дані.")

        client_socket.sendall(
            "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
            json.dumps(response_data).encode('utf-8')
        )
    except Exception as e:
        log_error(f"Помилка обробки POST-запиту адміністратора: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def handle_get_root(client_socket: socket.socket) -> None:
    """
    Обробляє GET-запит для кореневого маршруту.

    :param client_socket: Сокет клієнта.
    """
    try:
        rendered_html = template.render(previous_queries=[])
        compressed_html = compress_html(rendered_html)

        response_headers = (
            'HTTP/1.1 200 OK\n'
            'Content-Type: text/html; charset=utf-8\n'
            'Content-Encoding: gzip\n'
            'Cache-Control: public, max-age=3600\n'
            'Connection: close\n\n'
        )

        client_socket.sendall(response_headers.encode('utf-8') + compressed_html)
        log_info("Успішно відправлено index.html.")
    except Exception as e:
        log_error(f"Помилка при відправці index.html: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def handle_get_admin(client_socket: socket.socket) -> None:
    """
    Обробляє GET-запит для отримання списку адміністративних дій.

    :param client_socket: Сокет клієнта.
    """
    try:
        actions: List[str] = ["view_books", "add_book", "edit_book", "delete_book"]
        response_data: str = json.dumps(actions)

        client_socket.sendall(
            "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
            response_data.encode('utf-8')
        )
        log_info("Список дій адміністратора успішно відправлено.")
    except Exception as e:
        log_error(f"Помилка при обробці GET-запиту адміністратора: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def handle_post_request(client_socket: socket.socket, request: str) -> None:
    """
    Обробляє POST-запити для пошуку книг.

    :param client_socket: Сокет клієнта.
    :param request: Запит клієнта.
    """
    try:
        content_length: int = int(
            [header.split(":")[1].strip() for header in request.split("\r\n") if "Content-Length" in header][0])
        body: str = request.split("\r\n\r\n", 1)[1][:content_length]
        data: Dict[str, Any] = json.loads(body)

        action: Optional[str] = data.get("action")
        user_query: Optional[List[str]] = data.get("user_query")

        if action == "search" and user_query:
            all_queries: str = ". ".join(user_query)
            log_info(f"Пошук книг за запитом: {all_queries}")
            recommendations = index.search_books(all_queries)[:QUERIES_TO_SHOW]

            formatted_recommendations: List[Dict[str, Any]] = [
                {
                    "isbn": book['book_information'].get('isbn'),
                    "title": book['book_information'].get('title'),
                    "author": book['book_information'].get('author'),
                    "year": book['book_information'].get('publish_year'),
                    "description": book['book_information'].get('description') if str(book['book_information'].get('description')).lower() != 'nan' else '',
                    "cover": book['book_information'].get('cover', ''),
                    "similarity": round(book['similarity'] * 100, 2)
                } for book in recommendations
            ]

            response_data: Dict[str, Any] = {"recommendations": formatted_recommendations}

            client_socket.sendall(
                "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
                json.dumps(response_data).encode('utf-8')
            )
            log_info(f"Успішно відправлено {len(formatted_recommendations)} рекомендацій")
        else:
            log_warning("Невірні дані для пошуку")
    except Exception as e:
        log_error(f"Помилка обробки POST-запиту: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def start_server() -> None:
    """
    Запускає сервер для обробки клієнтських запитів.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        log_info(f"Сервер запущено на http://{HOST}:{PORT}")

        try:
            while True:
                client_socket, address = server_socket.accept()
                log_info(f"Прийнято з'єднання від {address}")
                thread_pool.submit(handle_request, client_socket)
        except KeyboardInterrupt:
            log_info("Завершення роботи сервера.")
        except Exception as e:
            log_error(f"Сталася помилка на сервері: {e}")
        finally:
            thread_pool.shutdown(wait=True)
            log_info("Усі робочі потоки завершені. Сервер вимкнено.")


if __name__ == "__main__":
    start_server()
