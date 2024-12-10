import socket
import threading
import gzip
import os
import json
from jinja2 import Environment, FileSystemLoader
from search import analyze_user_query, search_books

HOST = '127.0.0.1'
PORT = 8080

# Глобальные переменные
previous_queries = []
lock = threading.Lock()

# Путь к директории с CSS и HTML файлами
STATIC_DIR = 'static'
TEMPLATES_DIR = 'templates'

# Настроим Jinja2 для рендеринга шаблонов
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

def compress_html(content):
    """Функция для сжатия HTML с использованием Gzip"""
    return gzip.compress(content.encode('utf-8'))

def handle_client(client_socket):
    """Обработка запроса от клиента"""
    with client_socket:
        request = client_socket.recv(2048).decode('utf-8')
        print(f"[INFO] Клиент {client_socket} запросил: {request}")

        # Разбираем запрос клиента
        if request.startswith('POST /search'):
            try:
                # Извлекаем Content-Length из заголовков
                content_length = int([header.split(":")[1].strip() for header in request.split("\r\n") if "Content-Length" in header][0])

                # Находим начало тела запроса
                request_parts = request.split("\r\n\r\n", 1)
                if len(request_parts) < 2:
                    raise ValueError("Неверный формат HTTP-запроса")

                # Извлекаем тело запроса из request
                body = request_parts[1][:content_length]

                # Парсим JSON из тела запроса
                data = json.loads(body)

                action = data.get("action")
                user_query = data.get("user_query")
                index_to_delete = data.get("index")

                if action == "search" and user_query:
                    if user_query not in previous_queries:
                        with lock:
                            previous_queries.append(user_query)
                    all_queries = ". ".join(previous_queries)

                    titles, authors, genres, keywords = set(), set(), set(), ""
                    titles, authors, genres, keywords = analyze_user_query(all_queries, titles, authors, genres, keywords)
                    recommendations = search_books(titles, authors, genres, keywords)
                    recommendations = recommendations[:10]

                    response_data = {
                        "previous_queries": previous_queries,
                        "recommendations": recommendations
                    }

                elif action == "delete" and index_to_delete is not None:
                    with lock:
                        if 0 <= index_to_delete < len(previous_queries):
                            previous_queries.pop(index_to_delete)
                    all_queries = ". ".join(previous_queries)
                    titles, authors, genres, keywords = analyze_user_query(all_queries, set(), set(), set(), "")
                    recommendations = search_books(titles, authors, genres, keywords)
                    recommendations = recommendations[:10]

                    response_data = {
                        "previous_queries": previous_queries,
                        "recommendations": recommendations
                    }

                elif action == "clear":
                    with lock:
                        previous_queries.clear()
                    response_data = {
                        "previous_queries": [],
                        "recommendations": []
                    }

                # Отправляем ответ клиенту
                client_socket.sendall(
                    "HTTP/1.1 200 OK\nContent-Type: application/json\n\n".encode('utf-8') +
                    json.dumps(response_data).encode('utf-8')
                )

            except Exception as e:
                print(f"[ERROR] Ошибка обработки POST-запроса: {e}")
                client_socket.sendall("HTTP/1.1 500 Internal Server Error\n\n".encode('utf-8'))

        elif request.startswith('GET / HTTP/'):
            template = env.get_template("index.html")
            rendered_html = template.render(previous_queries=previous_queries)
            compressed_html = compress_html(rendered_html)
            response_headers = (
                'HTTP/1.1 200 OK\n'
                'Content-Type: text/html; charset=utf-8\n'
                'Content-Encoding: gzip\n'
                'Cache-Control: public, max-age=3600\n'
                'Connection: close\n\n'
            )
            client_socket.sendall(response_headers.encode('utf-8') + compressed_html)

        elif request.startswith('GET /static/'):
            filename = request.split(' ')[1][8:]
            send_static_file(client_socket, filename)

        else:
            client_socket.sendall("HTTP/1.1 404 Not Found\n\n".encode('utf-8'))
    print(f'Клиент {client_socket} отключился.')


def send_static_file(client_socket, filename):
    """Функция отправки статического файла (например, CSS)"""
    try:
        file_path = os.path.join(STATIC_DIR, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                file_data = file.read()
            response_headers = (
                'HTTP/1.1 200 OK\n'
                'Content-Type: text/css\n'
                'Cache-Control: public, max-age=3600\n'
                'Connection: close\n\n'
            )
            client_socket.sendall(response_headers.encode('utf-8') + file_data)
        else:
            response_headers = 'HTTP/1.1 404 Not Found\n\n'
            client_socket.sendall(response_headers.encode('utf-8'))
    except Exception as e:
        print(f"[ERROR] Ошибка при отправке статики: {e}")
        response_headers = 'HTTP/1.1 500 Internal Server Error\n\n'
        client_socket.sendall(response_headers.encode('utf-8'))

def start_server():
    """Запуск сервера"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"[INFO] Сервер запущен на http://{HOST}:{PORT}")

        while True:
            client_socket, _ = server_socket.accept()
            threading.Thread(target=handle_client, args=(client_socket,)).start()

if __name__ == "__main__":
    start_server()
