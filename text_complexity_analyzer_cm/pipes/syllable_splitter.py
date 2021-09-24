import pyphen

from spacy.tokens import Doc
from spacy.tokens import Token
from spacy.language import Language

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES, LANGUAGES_DICTIONARY_PYPHEN
from text_complexity_analyzer_cm.utils.utils import is_word

Token.set_extension('syllables', default=None, force=True)

@Language.factory('syllables')
class SyllableSplitter:
    
    def __init__(self, nlp, name, language) -> None:
        self._language = language
        self._dic = pyphen.Pyphen(lang=LANGUAGES_DICTIONARY_PYPHEN[language])
    
    def __call__(self, doc: Doc) -> Doc:
        for token in doc:
            if is_word(token):
                token._.syllables = self._dic.inserted(token.text).split('-')
        return doc