
from utils.dictionary import load_dictionary, get_category
d = load_dictionary('data/raw/derogatory_dictionary.csv')
print(get_category('blacks', d))
print(get_category('hello', d))
print(get_category('chink', d))
print(d.get('chink'))