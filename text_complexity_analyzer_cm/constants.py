'''
This function contains constants that will be used across the entire library.
'''

import os

language = {
    'es': 'es_core_news_lg',
    'en': 'en_core_web_sm'
}

ACCEPTED_LANGUAGES = {
    'es': 'es_core_news_lg',
    'en': 'en_core_web_sm',
}

LANGUAGES_DICTIONARY_PYPHEN = {
    'es': 'es',
    'en': 'en'
}

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
