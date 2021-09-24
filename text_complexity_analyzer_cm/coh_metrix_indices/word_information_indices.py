import multiprocessing
import spacy

from typing import Callable
from typing import List
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs

class WordInformationIndices:
    def __init__(self, nlp, language: str='en', descriptive_indices: DescriptiveIndices=None) -> None:
        self.language = language
        self._nlp = nlp
        self._incidence = 1000
        if descriptive_indices is None:
            self._di = DescriptiveIndices(language=language, nlp=nlp)
        else:
            self._di = descriptive_indices

    def _get_word_type_incidence(self, text: str, disable_pipeline :List, counter_function: Callable, word_count: int=None, workers: int=-1) -> float:
        paragraphs = split_text_into_paragraphs(text)
        wc = word_count if word_count is not None else self._di.get_word_count_from_text(text)
        self._nlp.get_pipe('feature counter').counter_function = counter_function
        words = sum(doc._.feature_count for doc in self._nlp.pipe(paragraphs, batch_size=1, disable=disable_pipeline, n_process=1))
        result = words #(words / wc)
        return result

    def get_noun_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        noun_counter = lambda doc: sum(1 for token in doc if is_word(token) and token.pos_ in ['NOUN', 'PROPN'])
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        result = self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=noun_counter, workers=workers)
        return result

    def get_verb_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        verb_counter = lambda doc: sum(1 for token in doc if is_word(token) and token.pos_ == 'VERB')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=verb_counter, workers=workers)

    def get_adjective_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        adjective_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'ADJ')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=adjective_counter, workers=workers)

    def get_adverb_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        adverb_counter = lambda doc: sum(1
                                        for token in doc
                                        if is_word(token) and token.pos_ == 'ADV')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=adverb_counter, workers=workers)

    def get_personal_pronoun_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1
                                        for token in doc
                                        if is_word(token) and token.pos_ == 'PRON')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_first_person_singular_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'PRON' and 'Number=Sing' in token.morph and 'Person=1' in token.morph)
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_first_person_plural_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'PRON' and 'Number=Plur' in token.morph and 'Person=1' in token.morph)
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_second_person_singular_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'PRON' and 'Number=Sing' in token.morph and 'Person=2' in token.morph)
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_second_person_plural_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'PRON' and 'Number=Plur' in token.morph and 'Person=2' in token.morph)
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_third_person_singular_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'PRON' and 'Number=Sing' in token.morph and 'Person=3' in token.morph)        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_third_person_plural_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        pronoun_counter = lambda doc: sum(1 for token in doc if is_word(token) and token.pos_ == 'PRON' and 'Number=Plur' in token.morph and 'Person=3' in token.morph)
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'feature counter']]
        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)
