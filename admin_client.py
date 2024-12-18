import socket
import json
import csv

HOST = '127.0.0.1'
PORT = 8080


def get_admin_actions():
    """Отримання списку дій адміністратора з сервера."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((HOST, PORT))
            request = "GET /admin HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: close\r\n\r\n"
            client_socket.sendall(request.encode('utf-8'))

            response = b""
            while True:
                chunk = client_socket.recv(2048)
                if not chunk:
                    break
                response += chunk

            response = response.decode('utf-8')
            request_parts = response.split("\n\n", 2)
            if len(request_parts) < 2:
                raise ValueError("Невірний формат HTTP запиту")

            body = request_parts[1]
            actions = json.loads(body)
            return actions
    except Exception as e:
        print(f"Помилка при отриманні дій адміністратора: {e}")
        return []


def send_admin_request(action, book_data=None, book_ids=None):
    """
    Відправка POST-запиту для виконання дії адміністратора.

    :param action: Одна з дій, переданих сервером.
    :param book_data: Список словників, які містять дані про книги. Мають містити один або кілька з ключів:
    'title', 'author', 'isbn', 'description', 'publish_year'.
    :param book_ids: Список індексів книг.
    :return: Відповідь сервера.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((HOST, PORT))
            request_data = {"action": action}
            if book_data:
                request_data["book_data"] = book_data
            if book_ids:
                request_data["book_ids"] = book_ids

            request_json = json.dumps(request_data)
            request = (
                "POST /admin HTTP/1.1\r\n"
                f"Host: {HOST}:{PORT}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(request_json)}\r\n"
                "Connection: close\r\n\r\n"
                f"{request_json}"
            )
            client_socket.sendall(request.encode('utf-8'))

            response = b""
            while True:
                chunk = client_socket.recv(2048)
                if not chunk:
                    break
                response += chunk

            response = response.decode('utf-8')
            request_parts = response.split("\n\n", 2)
            if len(request_parts) < 2:
                raise ValueError("Невірний формат HTTP запиту")

            body = request_parts[1]
            content = json.loads(body)
            return content
    except Exception as e:
        raise e


def save_books_to_csv(books):
    """
    Збереження списку книг у CSV файл.

    :param books: Список книг, отриманий з сервера, для збереження у result.csv
    """
    if not books:
        print("Немає даних для збереження в CSV.")
        return

    try:
        with open("result.csv", mode="w", newline='', encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["title", "author", "publish_year", "isbn", "description"])
            writer.writeheader()
            writer.writerows(books)
        print("Дані успішно збережено в файл result.csv.")
    except Exception as e:
        print(f"Помилка при збереженні в CSV: {e}")


def get_book_input():
    """Запит даних про книгу."""
    title = input("Введіть назву книги: ")
    author = input("Введіть автора: ")
    year = input("Введіть рік публікації: ")
    isbn = input("Введіть ISBN: ")
    description = input("Введіть опис: ")

    data = {}
    if title:
        data["title"] = title
    if author:
        data["author"] = author
    if year:
        data["publish_year"] = year
    if isbn:
        data["isbn"] = isbn
    if description:
        data["description"] = description

    return data


def main():
    """Основна функція для роботи адміністратора з сервером."""
    actions = get_admin_actions()
    if not actions:
        print("Не вдалося отримати дії від сервера.")
        return

    while True:
        print("\nДоступні дії:")
        for i, action in enumerate(actions, start=1):
            print(f"{i}. {action}")

        choice = input("Введіть номер дії (або 'q' для виходу): ")

        if choice.lower() == 'q':
            print("Вихід з адміністративного клієнта.")
            break

        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(actions):
                selected_action = actions[choice_index]

                if selected_action == "view_books":
                    books = send_admin_request("view_books")
                    save_books_to_csv(books)

                elif selected_action == "add_book":
                    num_books = int(input("Скільки книг ви хочете додати? "))
                    books_data = []
                    for _ in range(num_books):
                        book_data = get_book_input()
                        books_data.append(book_data)
                        print()
                    send_admin_request("add_book", book_data=books_data)

                elif selected_action == "edit_book":
                    book_ids = input("Введіть ID книг для редагування (через кому): ").split(",")
                    book_ids = [book_id.strip() for book_id in book_ids]
                    books_data = []
                    for id in book_ids:
                        print(f"Редагування книги #{id}")
                        book_data = get_book_input()
                        books_data.append(book_data)
                    send_admin_request("edit_book", book_data=books_data, book_ids=book_ids)

                elif selected_action == "delete_book":
                    book_ids = input("Введіть ID книг для видалення (через кому): ").split(",")
                    book_ids = [book_id.strip() for book_id in book_ids]
                    send_admin_request("delete_book", book_ids=book_ids)

                else:
                    print("Вибрана невідома дія.")
            else:
                print("Невірний вибір. Будь ласка, виберіть правильний номер.")
        except ValueError:
            print("Невірний ввід. Будь ласка, введіть число.")


if __name__ == "__main__":
    main()
