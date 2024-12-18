import socket
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import threading
from jinja2 import Environment, FileSystemLoader
from inverted_index import BookEmbeddings, InvertedIndex
import logging

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

# Конфигурация сервера
HOST = '127.0.0.1'
PORT = 8080
TEMPLATES_DIR = 'templates'
queries_to_show = 10

# Инициализация окружения Jinja2
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
template = env.get_template("index.html")

# Инициализация BookEmbeddings и InvertedIndex
embedding_generator = BookEmbeddings()
index = InvertedIndex(embedding_generator, books_file="books_database_with_descriptions.csv")

lock = threading.Lock()

# Пул потоков для обработки клиентов
thread_pool = ThreadPoolExecutor(max_workers=10)

def compress_html(content):
    """Сжатие HTML-контента с использованием Gzip."""
    return gzip.compress(content.encode('utf-8'))

def handle_request(client_socket):
    """Handle incoming client requests."""
    with client_socket:
        try:
            request = client_socket.recv(2048).decode('utf-8')
            client_ip = client_socket.getpeername()[0]
            logging.info(f"Received request from {client_ip}: {request}")

            if request.startswith('POST /search'):
                handle_post_request(client_socket, request)

            elif request.startswith('POST /admin'):
                handle_admin_post_request(client_socket, request)

            elif request.startswith('GET /admin HTTP/'):
                handle_get_admin(client_socket)

            elif request.startswith('GET / HTTP/'):
                handle_get_root(client_socket)

            else:
                logging.warning(f"Received unsupported request from {client_ip}: {request.splitlines()[0]}")
                client_socket.sendall("HTTP/1.1 404 Not Found\n\n".encode('utf-8'))

        except Exception as e:
            logging.error(f"Error handling request from {client_ip}: {e}")



def handle_get_admin(client_socket):
    """Обработка GET-запроса для получения списка действий администратора."""
    try:
        actions = ["view_books", "add_book", "edit_book", "delete_book"]
        response_data = json.dumps(actions)

        client_socket.sendall(
            "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
            response_data.encode('utf-8')
        )
        logging.info("Sent admin actions list successfully")

    except Exception as e:
        logging.error(f"Error serving admin actions list: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def handle_post_request(client_socket, request):
    """Обработка POST-запросов для поиска."""
    try:
        content_length = int([header.split(":")[1].strip() for header in request.split("\r\n") if "Content-Length" in header][0])

        request_parts = request.split("\r\n\r\n", 1)
        if len(request_parts) < 2:
            raise ValueError("Invalid HTTP request format")

        body = request_parts[1][:content_length]
        data = json.loads(body)

        action = data.get("action")
        user_query = data.get("user_query")
        index_to_delete = data.get("index", -1)

        logging.info(f"Action: {action}, User Query: {user_query}, Index to Delete: {index_to_delete}")

        if action == "search" and user_query:
            all_queries = ". ".join(user_query)
            logging.info(f"Search performed with query: {all_queries}")
            recommendations = index.search_books(all_queries)[:queries_to_show]

            # Формирование списка рекомендаций для отправки на клиент
            formatted_recommendations = [{
                "isbn": book['book_information'].get('isbn'),
                "title": book['book_information'].get('title'),
                "author": book['book_information'].get('author'),
                "year": book['book_information'].get('publish_year'),
                "description": book['book_information'].get('description') if str(book['book_information'].get('description')) != 'nan' else '',
                "cover": book['book_information'].get('cover', ''),
                "similarity": round(book['similarity'] * 100, 2)
            } for book in recommendations]

            response_data = {
                "recommendations": formatted_recommendations
            }
            logging.info(f"Search results: {[(book['book_id'], 
                                              book['book_information'].get('title'),
                                              book['similarity']) for book in recommendations]}")

        client_socket.sendall(
            "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
            json.dumps(response_data).encode('utf-8')
        )
        logging.info(f"Response sent successfully for action: {action}")

    except Exception as e:
        logging.error(f"Error processing POST request: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def handle_admin_post_request(client_socket, request):
    """Handle POST requests for admin actions."""
    try:
        content_length = int(
            [header.split(":")[1].strip() for header in request.split("\r\n") if "Content-Length" in header][0])

        request_parts = request.split("\r\n\r\n", 1)
        if len(request_parts) < 2:
            raise ValueError("Invalid HTTP request format")

        body = request_parts[1][:content_length]
        data = json.loads(body)

        action = data.get("action")
        book_data = data.get("book_data")
        if not isinstance(book_data, list):
            book_data = [book_data]
        book_ids = data.get("book_ids", [])
        if not isinstance(book_ids, list):
            book_ids = [book_ids]

        if action == "view_books":
            books_list = [{"id": book_id, **book_info} for book_id, book_info in index.books.items()]
            response_data = {"books": books_list}

        elif action == "add_book" and book_data is not None:
            try:
                new_ids = [index.add_book(book) for book in book_data]
                response_data = {"message": f"Books with IDs {new_ids} added successfully."}
            except Exception as e:
                response_data = {"error": f"Error adding books: {e}"}
                logging.error(f"Error adding books: {e}")

        elif action == "edit_book" and book_ids and book_data is not None:
            try:
                for book_id, book in zip(book_ids, book_data):
                    index.update_book(int(book_id), book)
                response_data = {"message": f"Books with IDs {book_ids} updated successfully."}
            except Exception as e:
                response_data = {"error": f"Error updating books: {e}"}
                logging.error(f"Error updating books: {e}")

        elif action == "delete_book" and book_ids:
            try:
                for book_id in book_ids:
                    index.delete_book(int(book_id))
                response_data = {"message": f"Books with IDs {book_ids} deleted successfully."}
            except Exception as e:
                response_data = {"error": f"Error deleting books: {e}"}
                logging.error(f"Error deleting books: {e}")

        else:
            response_data = {"error": "Invalid action or missing data."}

        client_socket.sendall(
            "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
            json.dumps(response_data).encode('utf-8')
        )
        logging.info(f"Admin action '{action}' processed successfully.")

    except Exception as e:
        logging.error(f"Error processing admin POST request: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))


def handle_get_root(client_socket):
    """Обработка GET-запроса для корневого маршрута."""
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
        logging.info("Served index.html successfully")

    except Exception as e:
        logging.error(f"Error serving index.html: {e}")
        client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))

def start_server():
    """Запуск сервера для обработки клиентских соединений."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        logging.info(f"Server started on http://{HOST}:{PORT}")

        try:
            while True:
                client_socket, address = server_socket.accept()
                logging.info(f"Accepted connection from {address}")
                thread_pool.submit(handle_request, client_socket)
        except KeyboardInterrupt:
            logging.info("Server shutting down.")

if __name__ == "__main__":
    start_server()
