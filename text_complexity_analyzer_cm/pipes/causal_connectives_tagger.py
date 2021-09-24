from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.language import Language

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

causal_connectives_getter = lambda doc: [doc[span['start']:span['end']] for span in doc._.causal_connectives_span_indices]

Doc.set_extension('causal_connectives_span_indices', force=False, default=[])
Doc.set_extension('causal_connectives', force=False, getter=causal_connectives_getter)

@Language.factory('causal connective tagger')
class CausalConnectivesTagger:
    def __init__(self, name, nlp, language) -> None:
        self._language = language
        self._matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        self.causal_connectives = []
        if language == 'en':
            self.causal_connectives = ['to repeat, briefly', 'finally', 'therefore', 'with this in mind', 'in conclusion', 'because of this', 'because of', 'as a consequence', 'to this end', 'on the score of', 'then', 'because', 'so', 'later', 'hence', 'in short', 'for this reason', 'thus', 'so much that', 'accordingly', 'for', 'so then', 'as I have said', 'therefore', 'in summary', 'on the whole', 'consequently', 'for this purpose', 'since', 'as a result', 'to sum up', 'so that', 'as you can see']
        else: 
            pass
        for con in self.causal_connectives:
            self._matcher.add(con, None, nlp(con))
        
    def __call__(self, doc: Doc) -> Doc:
        matches = self._matcher(doc)
        causal_connectives_spans = [doc[start:end] for _, start, end in matches]       
        doc._.causal_connectives_span_indices = [{'start': span.start,
                                                'end': span.end,
                                                'label': span.label}
                                                for span in filter_spans(causal_connectives_spans)]
        return doc