import string
from collections import Counter
import re
from collections import defaultdict

class InvertedIndex:
    def __init__(self, stop_words=None):
        self.index = defaultdict(lambda: defaultdict(int))  # Термін → Документ → Частота
        self.stop_words = stop_words or {"the", "is", "a", "by", "and", "of"}

    def add_document(self, doc_id, text):
        tokens = self.tokenize(text)
        for token in tokens:
            self.index[token][doc_id] += 1

    def search(self, query):
        tokens = self.tokenize(query)
        return self._get_documents(tokens)

    def search_with_ranking(self, query):
        tokens = self.tokenize(query)
        return self._rank_documents(tokens)

    def _get_documents(self, tokens):
        result_sets = [set(self.index[token].keys()) for token in tokens if token in self.index]
        return set.intersection(*result_sets) if result_sets else set()

    def _rank_documents(self, tokens):
        doc_scores = Counter()
        for token in tokens:
            for doc_id, freq in self.index.get(token, {}).items():
                doc_scores[doc_id] += freq
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)  # Витягуємо тільки слова
        return [token for token in tokens if token not in self.stop_words]