from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

temporal_connectives_getter = lambda doc: [doc[span['start']:span['end']] for span in doc._.temporal_connectives_span_indices]

Doc.set_extension('temporal_connectives_span_indices', force=False, default=[])
Doc.set_extension('temporal_connectives', force=False, getter=temporal_connectives_getter)

@Language.factory('temporal connective tagger')
class TemporalConnectivesTagger:
    def __init__(self, name, nlp, language) -> None:
        self._language = language
        self._matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        self.temporal_connectives = []
        if language == 'en':
            self.temporal_connectives = ['deadline', 'finish', 'then', 'finally', 'after', 'a minute later', 'ultimately', 'subsequently', 'meanwhile', 'next', 'soon', 'in the meantime', 'once upon a time', 'later', 'finally', 'already', 'afterward', 'presently', 'thereafter', 'initially', 'now', 'last', 'at length', 'previously', 'currently', 'at last', 'the next week', 'lastly', 'a long time ago', 'at the same time', 'during', 'before', 'when', 'the next month', 'the next day', 'on the following day', 'time before', 'while', 'later', 'simultaneously', 'after a short time']
        else: 
            pass
        for con in self.temporal_connectives:
            self._matcher.add(con, None, nlp(con))

    def __call__(self, doc: Doc) -> Doc:
        matches = self._matcher(doc)
        temporal_connectives_spans = [doc[start:end] for _, start, end in matches]
        doc._.temporal_connectives_span_indices = [{'start': span.start,
                                                    'end': span.end,
                                                    'label': span.label}
                                                    for span in filter_spans(temporal_connectives_spans)] # Save the temporal connectives found
        return doc