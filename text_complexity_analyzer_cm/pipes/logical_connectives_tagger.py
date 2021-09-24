from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

logical_connectives_getter = lambda doc: [doc[span['start']:span['end']]
                                          for span in doc._.logical_connectives_span_indices]

Doc.set_extension('logical_connectives_span_indices', force=False, default=[])
Doc.set_extension('logical_connectives', force=False, getter=logical_connectives_getter)

@Language.factory('logical connective tagger')
class LogicalConnectivesTagger:
    def __init__(self, name, nlp, language: str='en') -> None:
        '''
        This constructor will initialize the object that tags logical connectives.

        Parameters:
        nlp: The Spacy model to use this tagger with.
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language
        self._matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        self._connectives = []
        if language == 'en': # Logical connectives for spanish
            self._connectives = ['and', 'or']
        else: # Support for future languages
            # self._connectives = ['y', 'o']
            pass

        for con in self._connectives:
            self._matcher.add(con, None, nlp(con))
        

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find all logical connectives and store them in an iterable.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        matches = self._matcher(doc)
        logical_connectives_spans = [doc[start:end] for _, start, end in matches]

        doc._.logical_connectives_span_indices = [{'start': span.start,
                                                    'end': span.end,
                                                    'label': span.label}
                                                  for span in filter_spans(logical_connectives_spans)] # Save the logical connectives found
        
        return doc