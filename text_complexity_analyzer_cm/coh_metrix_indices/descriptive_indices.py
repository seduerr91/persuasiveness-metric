import multiprocessing
import spacy
import statistics
import string

from typing import Callable
from typing import List
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES, LANGUAGES_DICTIONARY_PYPHEN
from text_complexity_analyzer_cm.pipes.syllable_splitter import SyllableSplitter
from text_complexity_analyzer_cm.utils.statistics_results import StatisticsResults
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs
from text_complexity_analyzer_cm.utils.utils import split_text_into_sentences
from text_complexity_analyzer_cm.utils.utils import split_doc_into_sentences


class DescriptiveIndices:
    def __init__(self, nlp, language: str='en') -> None:
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')
        
        self.language = language
        self._nlp = nlp

    def get_paragraph_count_from_text(self, text: str) -> int:
        if len(text) == 0:
            raise ValueError('The text is empty.')
        
        return len(split_text_into_paragraphs(text))

    def get_sentence_count_from_text(self, text: str, workers: int=-1) -> int:
        if len(text) == 0:
            raise ValueError('The text is empty.')
        elif workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0')
        else:
            paragraphs = split_text_into_paragraphs(text) # Obtain paragraphs
            threads = 1#multiprocessing.cpu_count() if workers == -1 else workers  
            disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['sentencizer', 'feature counter']]

            sentence_counter = lambda doc: sum(1 for _ in doc.sents)
            self._nlp.get_pipe('feature counter').counter_function = sentence_counter
            
            sentences = sum(doc._.feature_count
                            for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads))
            return sentences

    def get_word_count_from_text(self, text: str, workers: int=-1) -> int:
        if len(text) == 0:
            raise ValueError('The text is empty.')
        elif workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0')
        else:
            paragraphs = split_text_into_paragraphs(text) # Obtain paragraphs
            threads = 1#multiprocessing.cpu_count() if workers == -1 else workers
            word_counter = lambda doc: sum(1 for token in doc if is_word(token))
            disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe != 'feature counter']
            self._nlp.get_pipe('feature counter').counter_function = word_counter

            total_words = sum(doc._.feature_count for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads))


            return total_words

    def _get_mean_std_of_metric(self, text: str, disable_pipeline: List, counter_function: Callable, statistic_type: str='all', workers=-1) -> StatisticsResults:
        paragraphs = split_text_into_paragraphs(text)
        threads = 1
        self._nlp.get_pipe('feature counter').counter_function = counter_function
        counter = []

        for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads):
            current_result = doc._.feature_count # Find the values to add to the counter
            if not isinstance(current_result, list): # Add any numbers
                counter.append(current_result)
            else:
                if len(current_result) > 0: # Only add values if its not an empty array
                    counter.extend(current_result)

        stat_results = StatisticsResults()
        if statistic_type in ['std', 'all']:
            stat_results.std = statistics.pstdev(counter)
        
        if statistic_type in ['mean', 'all']:
            stat_results.mean = statistics.mean(counter)

        return stat_results

    def get_length_of_paragraphs(self, text: str, workers: int=-1) -> StatisticsResults:
        count_length_of_paragraphs = lambda doc: sum(1 for _ in split_doc_into_sentences(doc))
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['sentencizer', 'feature counter']]
        return self._get_mean_std_of_metric(text, disable_pipeline=disable_pipeline, counter_function=count_length_of_paragraphs, statistic_type='all', workers=workers)

    def get_length_of_sentences(self, text: str, workers: int=-1) -> StatisticsResults:
        count_length_of_sentences = lambda doc: [len([1 for token in sentence
                                                    if is_word(token)])
                                                    for sentence in doc.sents]

        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['sentencizer', 'feature counter']]

        return self._get_mean_std_of_metric(text, disable_pipeline=disable_pipeline, counter_function=count_length_of_sentences, statistic_type='all', workers=workers)

    def get_length_of_words(self, text: str, workers: int=-1) -> StatisticsResults:

        count_letters_per_word = lambda doc: [len(token)
                                            for token in doc
                                            if is_word(token)]
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe != 'feature counter']
        
        result = self._get_mean_std_of_metric(text, disable_pipeline=disable_pipeline, counter_function=count_letters_per_word, statistic_type='all', workers=workers)
        
        return result

    def get_syllables_per_word(self, text: str, workers=-1) -> StatisticsResults:
        count_syllables_per_word = lambda doc: [len(token._.syllables)
                                                for token in doc
                                                if is_word(token) and token._.syllables is not None]
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['syllables', 'feature counter']]
        
        result = self._get_mean_std_of_metric(text, disable_pipeline=disable_pipeline, counter_function=count_syllables_per_word, statistic_type='all', workers=workers)
    
        return result 
