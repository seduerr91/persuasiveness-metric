from text_complexity_analyzer_cm.perm import PERM
import pprint

text='''
We are going to analyze this text for persuasiveness, word information, and descriptive information.
'''

pp = pprint.PrettyPrinter(indent=4)
perm = PERM('en')

print('\n<<< Textual Input>>>\n')
print(text)

print('\n<<< Persuasiveness of Text >>>\n')
pp.pprint(perm.calculate_persuasiveness_categorical(text, workers=-1))

print('\n<<< Word Information >>>\n')
pp.pprint(perm.calculate_word_information_indices_for_one_text(text, workers=-1))

print('\n<<< Descriptive Information >>>\n')
pp.pprint(perm.calculate_descriptive_indices_for_one_text(text, workers=-1))
