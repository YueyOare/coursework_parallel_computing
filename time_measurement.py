import logging
import os
import time
import matplotlib.pyplot as plt

from colorama import init, Fore
from inverted_index import BookEmbeddings, InvertedIndex

if __name__ == "__main__":
    init(autoreset=True)

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    def log_info(message: str) -> None:
        logging.info(Fore.GREEN + message)

    RESULTS_FILE: str = 'time_measures.txt'

    embedding_generator: BookEmbeddings = BookEmbeddings()

    for num_threads in [1, 2, 4, 8, 16]:
        start_time = time.time()
        index: InvertedIndex = InvertedIndex(embedding_generator,
                                             books_file="books_database_with_descriptions.csv",
                                             index_file="test_index.csv",
                                             num_threads=num_threads)
        end_time = time.time()

        elapsed_time = end_time - start_time

        with open(RESULTS_FILE, 'a') as f:
            f.write(f"{num_threads}, {elapsed_time}\n")

        log_info(f"Час створення індекса: {elapsed_time} с, кількість потоків: {num_threads}.")

        if os.path.exists('test_index.csv'):
            os.remove('test_index.csv')

    results = {}
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            num_threads, elapsed_time = map(float, line.strip().split(', '))
            if num_threads not in results:
                results[num_threads] = []
            results[num_threads].append(elapsed_time)

    average_times = {num_threads: sum(times) / len(times) for num_threads, times in results.items()}
    sorted_num_threads = sorted(average_times.keys())
    sorted_average_times = [average_times[num_threads] for num_threads in sorted_num_threads]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_num_threads, sorted_average_times, marker='o', linestyle='-', color='b')
    plt.xlabel('Кількість потоків')
    plt.ylabel('Середній час (с)')
    plt.title('Середній час створення індексу з різною кількістю потоків')
    plt.grid(True)
    plt.savefig('images/time_vs_threads.png')
    plt.show()
