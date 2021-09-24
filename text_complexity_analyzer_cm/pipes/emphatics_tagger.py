from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

emphatics_getter = lambda doc: [doc[span['start']:span['end']]
                                        for span in doc._.emphatics_span_indices]

Doc.set_extension('emphatics_span_indices', force=False, default=[])
Doc.set_extension('emphatics', force=False, getter=emphatics_getter)

@Language.factory('emphatics tagger')
class EmphaticsTagger:
    def __init__(self, name, nlp, language) -> None:
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language
        self._matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        self._connectives = []
        if language == 'en': # emphatics connectives for spanish
            self._connectives = ['him', 'there', 'their', 'it', 'he', 'she', 'we', 'who', 'them', 'they', 'you', 'himself', 'her', 'whom', 'itself', 'somebody', 'something', 'us', 'anybody', 'herself', 'anyone', 'everybody', 'nobody', 'everyone', 'themselves', 'yourself', 'someone', 'his', 'yours']
        else: # Support for future languages
            pass

        for con in self._connectives:
            self._matcher.add(con, None, nlp(con))
        

    def __call__(self, doc: Doc) -> Doc:
        matches = self._matcher(doc)
        emphatics_spans = [doc[start:end] for _, start, end in matches]

        doc._.emphatics_span_indices = [{'start': span.start,
                                                    'end': span.end,
                                                    'label': span.label}
                                                for span in filter_spans(emphatics_spans)] # Save the emphatics connectives found
        
        return doc