import multiprocessing
import spacy
import string

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import is_content_word
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs

class LexicalDiversityIndices:
    def __init__(self, nlp, language: str='en') -> None:
        self.language = language
        self._nlp = nlp

    def get_type_token_ratio_between_all_words(self, text: str, workers=-1) -> float:
        paragraphs = split_text_into_paragraphs(text)
        threads = 1
        tokens = []
        disable_pipeline = []

        tokens = [token.text.lower()
                for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads)
                for token in doc
                if is_word(token)]

        return 0 if len(tokens) == 0 else len(set(tokens)) / len(tokens)

    def get_type_token_ratio_of_content_words(self, text: str, workers=-1) -> float:
        paragraphs = split_text_into_paragraphs(text)
        threads = 1
        tokens = []
        disable_pipeline = [] 
        tokens = [token.text.lower()
                for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads)
                for token in doc
                if is_content_word(token)]
        return 0 if len(tokens) == 0 else len(set(tokens)) / len(tokens)
