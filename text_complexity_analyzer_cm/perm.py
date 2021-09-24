import spacy
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.coh_metrix_indices.connective_indices import ConnectiveIndices
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.coh_metrix_indices.lexical_diversity_indices import LexicalDiversityIndices
from text_complexity_analyzer_cm.coh_metrix_indices.syntactic_complexity_indices import SyntacticComplexityIndices
from text_complexity_analyzer_cm.coh_metrix_indices.word_information_indices import WordInformationIndices
from text_complexity_analyzer_cm.pipes.syllable_splitter import SyllableSplitter
from text_complexity_analyzer_cm.pipes.causal_connectives_tagger import CausalConnectivesTagger
from text_complexity_analyzer_cm.pipes.emphatics_tagger import EmphaticsTagger
from text_complexity_analyzer_cm.pipes.asks_tagger import AsksTagger
from text_complexity_analyzer_cm.pipes.polites_tagger import PolitesTagger
from text_complexity_analyzer_cm.pipes.logical_connectives_tagger import LogicalConnectivesTagger
from text_complexity_analyzer_cm.pipes.adversative_connectives_tagger import AdversativeConnectivesTagger
from text_complexity_analyzer_cm.pipes.temporal_connectives_tagger import TemporalConnectivesTagger
from text_complexity_analyzer_cm.pipes.additive_connectives_tagger import AdditiveConnectivesTagger
from text_complexity_analyzer_cm.pipes.feature_counter import FeatureCounter
from typing import Dict

weights = {
    'Ben_Outcome': 0.333333,
    'Ben_Scarcity': 0.2,
    'Lingu_LexDiversityContentWords': 1-0.2,
    'Lingu_SynCompWordsBeforeMainVerb': 1-0.16,
    'Logic_Additives': 0.214286,
    'Logic_Adversatives': 0.206085,
    'Logic_Operators': 0.4,
    'Trust_Asks': 0.272727,
    'Trust_Emphatics': 0.785714,
    'Trust_Polites': 0.2}

class PERM:
    def __init__(self, language:str = 'en') -> None:
        self.language = language
        self._nlp = spacy.load(ACCEPTED_LANGUAGES[language], disable=['ner'])
        self._nlp.max_length = 3000000
        self._nlp.add_pipe('sentencizer')
        self._nlp.add_pipe('syllables', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('causal connective tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('temporal connective tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('emphatics tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('asks tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('polites tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('logical connective tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('adversative connective tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('additive connective tagger', config={"language": 'en'}, after='tagger')
        self._nlp.add_pipe('feature counter', config={"language": 'en'}, last=True)
        self._di = DescriptiveIndices(language=language, nlp=self._nlp)
        self._ci = ConnectiveIndices(language=language, nlp=self._nlp, descriptive_indices=self._di)
        self._ldi = LexicalDiversityIndices(language=language, nlp=self._nlp)
        self._sci = SyntacticComplexityIndices(language=language, nlp=self._nlp)
        self._wii = WordInformationIndices(language=language, nlp=self._nlp, descriptive_indices=self._di)

    def calculate_descriptive_indices_for_one_text(self, text: str, workers: int=-1) -> Dict:
        indices = {}

        indices['Paragraph Count'] = self._di.get_paragraph_count_from_text(text=text)
        indices['Sentence Count'] = self._di.get_sentence_count_from_text(text=text, workers=workers)
        indices['Word Count'] = self._di.get_word_count_from_text(text=text, workers=workers)
        length_of_paragraph = self._di.get_length_of_paragraphs(text=text, workers=workers)
        indices['Mean Length of Paragraphs'] = length_of_paragraph.mean
        length_of_sentences = self._di.get_length_of_sentences(text=text, workers=workers)
        indices['Mean Length of Sentences'] = length_of_sentences.mean
        length_of_words = self._di.get_length_of_words(text=text, workers=workers)
        indices['Mean Length of Words'] = length_of_words.mean
        syllables_per_word = self._di.get_syllables_per_word(text=text, workers=workers)
        indices['Mean Syllables of Words'] = syllables_per_word.mean

        return indices

    def calculate_word_information_indices_for_one_text(self, text: str, workers: int=-1, word_count: int=None) -> Dict:
        indices = {}
        
        indices['WRDNOUN'] = self._wii.get_noun_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDVERB'] = self._wii.get_verb_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDADJ'] = self._wii.get_adjective_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDADV'] = self._wii.get_adverb_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRO'] = self._wii.get_personal_pronoun_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP1s'] = self._wii.get_personal_pronoun_first_person_singular_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP1p'] = self._wii.get_personal_pronoun_first_person_plural_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP2s'] = self._wii.get_personal_pronoun_second_person_singular_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP2p'] = self._wii.get_personal_pronoun_second_person_plural_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP3s'] = self._wii.get_personal_pronoun_third_person_singular_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP3p'] = self._wii.get_personal_pronoun_third_person_plural_form_incidence(text=text, workers=workers, word_count=word_count)
        
        return indices

    def calculate_persuasiveness_categorical(self, text: str, workers: int=-1, word_count: int=None, ):
        
        word_count = self._di.get_word_count_from_text(text=text, workers=workers)
        sentence_count = self._di.get_sentence_count_from_text(text=text, workers=workers)
        
        perm = {}
        
        perm['Ben_Outcome'] = self._ci.get_causal_connectives_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Ben_Outcome']
        perm['Ben_Scarcity'] = self._ci.get_temporal_connectives_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Ben_Scarcity']
        
        perm['Lingu_SynCompWordsBeforeMainVerb'] = round(self._sci.get_mean_number_of_words_before_main_verb(text=text, workers=workers), 2) / weights['Lingu_SynCompWordsBeforeMainVerb'] / sentence_count / weights['Lingu_SynCompWordsBeforeMainVerb']
        perm['Lingu_LexDiversityContentWords'] = round(self._ldi.get_type_token_ratio_of_content_words(text=text, workers=workers), 2) / weights['Lingu_LexDiversityContentWords'] / sentence_count / weights['Lingu_LexDiversityContentWords']
        
        perm['Logic_Adversatives'] = self._ci.get_adversative_connectives_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Logic_Adversatives']
        perm['Logic_Additives'] = self._ci.get_additive_connectives_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Logic_Additives']
        perm['Logic_Operators'] = self._ci.get_logical_connectives_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Logic_Operators']

        perm['Trust_Emphatics'] = self._ci.get_emphatics_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Trust_Emphatics']
        perm['Trust_Asks'] = self._ci.get_asks_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Trust_Asks']
        perm['Trust_Polites'] = self._ci.get_polites_incidence(text=text, workers=workers, word_count=word_count) / sentence_count / weights['Trust_Polites']
        
        return perm
