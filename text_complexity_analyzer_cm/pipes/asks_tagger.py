from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

asks_getter = lambda doc: [doc[span['start']:span['end']] for span in doc._.asks_span_indices]

Doc.set_extension('asks_span_indices', force=False, default=[])
Doc.set_extension('asks', force=False, getter=asks_getter)

@Language.factory('asks tagger')
class AsksTagger:
    def __init__(self, name, nlp, language: str='en') -> None:
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language
        self._matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        self._connectives = []
        if language == 'en': # question words and questionmark
            self._connectives = ['?', 'what', 'how', 'who', 'when']
        else: # Support for future languages
            pass

        for con in self._connectives:
            self._matcher.add(con, None, nlp(con))
        

    def __call__(self, doc: Doc) -> Doc:
        matches = self._matcher(doc)
        asks_spans = [doc[start:end] for _, start, end in matches]
        doc._.asks_span_indices = [{'start': span.start, 'end': span.end, 'label': span.label} for span in filter_spans(asks_spans)] # Save the asks connectives found
        
        return doc