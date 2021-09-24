import spacy

from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token
from typing import List

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES


def split_text_into_paragraphs(text: str) -> List[str]:
    text_aux = text.strip()
    paragraphs = text_aux.split('\n\n') # Strip any leading whitespaces

    for p in paragraphs:
        p = p.strip()

    return [p.strip() for p in paragraphs if len(p) > 0] # Don't count empty paragraphs


def split_text_into_sentences(text: str) -> List[str]:
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
    nlp.add_pipe('sentencizer')
    text_spacy = nlp(text)
    return [str(sentence) for sentence in text_spacy.sents]


def is_content_word(token: Token) -> bool:
    result = token.is_alpha and token.pos_ in ['PROPN', 'NOUN', 'VERB', 'ADJ', 'ADV']
    return result 


def is_word(token: Token) -> bool:
    return token.is_alpha

def split_doc_into_sentences(doc: Doc) -> List[Span]:
    return [s for s in doc.sents if len(s.text.strip()) > 0]