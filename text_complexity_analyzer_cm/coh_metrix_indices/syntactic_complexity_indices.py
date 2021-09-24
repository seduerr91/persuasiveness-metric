import multiprocessing
from typing import Tuple

import spacy
import statistics

from spacy.tokens import Span
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs
from text_complexity_analyzer_cm.utils.utils import split_doc_into_sentences


class SyntacticComplexityIndices:
    def __init__(self, nlp, language: str='en') -> None:
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')
        
        self.language = language
        self._nlp = nlp

    def get_mean_number_of_modifiers_per_noun_phrase(self, text: str, workers: int=-1) -> float:
        paragraphs = split_text_into_paragraphs(text)
        threads = 1
        modifiers_per_noun_phrase = []
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['parser', 'tagger', 'noun phrase tagger', 'feature counter']]
        modifiers_counter = lambda doc: [sum(1 for token in nph if token.pos_ == 'ADJ')
                                        for nph in doc._.noun_phrases]
        self._nlp.get_pipe('feature counter').counter_function = modifiers_counter
        modifiers_per_noun_phrase = []

        for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads):
            modifiers_per_noun_phrase.extend(doc._.feature_count)
        
        return statistics.mean(modifiers_per_noun_phrase)

    def get_mean_number_of_words_before_main_verb(self, text: str, workers: int=-1) -> float:
        paragraphs = split_text_into_paragraphs(text)
        threads = 1
        words_before_main_verb = []
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['feature counter',  'sentencizer']]
        words_before_main_verb_counter = lambda doc: [amount_of_words_before_main_verb(s) for s in split_doc_into_sentences(doc)]
        self._nlp.get_pipe('feature counter').counter_function = words_before_main_verb_counter
        for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads):
            words_before_main_verb.extend(doc._.feature_count)
        return statistics.mean(words_before_main_verb)

def amount_of_words_before_main_verb(sentence: Span) -> int:
    left_words = []
    for token in sentence:
        if token.pos_ in ['VERB', 'AUX'] and token.dep_ == 'ROOT':
            break
        else:
            if is_word(token):
                left_words.append(token.text)
    return len(left_words)