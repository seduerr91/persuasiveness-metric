import multiprocessing
import pyphen
import spacy
import string

from typing import Callable
from typing import List
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs
from text_complexity_analyzer_cm.utils.utils import split_text_into_sentences

class ConnectiveIndices:
    def __init__(self, nlp, language: str='en', descriptive_indices: DescriptiveIndices=None) -> None:
        self.language = language
        self._nlp = nlp
        self._incidence = 1
        if descriptive_indices is None: 
            self._di = DescriptiveIndices(language)
        else:
            self._di = descriptive_indices

    def _get_connectives_incidence(self, text: str, disable_pipeline: List, count_connectives_function: Callable, word_count: int=None, workers: int=-1) -> float:
        paragraphs = split_text_into_paragraphs(text) 
        pc = len(paragraphs)
        threads = 1
        wc = word_count if word_count is not None else self._di.get_word_count_from_text(text)
        self._nlp.get_pipe('feature counter').counter_function = count_connectives_function
        connectives = sum(doc._.feature_count for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads))
        return connectives

    def get_causal_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['causal connective tagger', 'feature counter']]
        causal_connectives_counter = lambda doc: len(doc._.causal_connectives)
        result = self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=causal_connectives_counter, workers=workers)
        return result

    def get_temporal_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['temporal connective tagger', 'feature counter']]
        temporal_connectives_counter = lambda doc: len(doc._.temporal_connectives)
        result = self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=temporal_connectives_counter, workers=workers)
        return result

    def get_exemplifications_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['exemplifications tagger', 'tagger', 'feature counter']]
        exemplifications_counter = lambda doc: len(doc._.exemplifications)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=exemplifications_counter, workers=workers)

    def get_emphatics_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['emphatics tagger', 'tagger', 'feature counter']]
        emphatics_counter = lambda doc: len(doc._.emphatics)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=emphatics_counter, workers=workers)

    def get_asks_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['asks tagger', 'tagger', 'feature counter']]
        asks_counter = lambda doc: len(doc._.asks)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=asks_counter, workers=workers)

    def get_polites_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['polites tagger', 'tagger', 'feature counter']]
        polites_counter = lambda doc: len(doc._.polites)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=polites_counter, workers=workers)

    def get_logical_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['logical connective tagger', 'tagger', 'feature counter']]
        logical_connectives_counter = lambda doc: len(doc._.logical_connectives)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=logical_connectives_counter, workers=workers)

    def get_adversative_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['adversative connective tagger', 'tagger', 'feature counter']]
        adversative_connectives_counter = lambda doc: len(doc._.adversative_connectives)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=adversative_connectives_counter, workers=workers)

    def get_additive_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['additive connective tagger', 'tagger', 'feature counter']]
        additive_connectives_counter = lambda doc: len(doc._.additive_connectives)
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=additive_connectives_counter, workers=workers)