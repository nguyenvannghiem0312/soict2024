import pandas as pd
import re
import string
from pyvi.ViTokenizer import tokenize

class TextPreprocessor:
    def __init__(self, stopwords_file = None):
        if stopwords_file != None:
            data = pd.read_csv(stopwords_file, sep="\t", encoding='utf-8')
            self.stopwords = set(data['stopwords'])
        else:
            self.stopwords = None
        self.punctuation = string.punctuation.replace('_', '')

    def clean_text(self, text):
        text = re.sub('<.*?>', '', text).strip()
        return re.sub('(\s)+', r'\1', text)

    def normalize_text(self, text):
        for punc in self.punctuation:
            text = text.replace(punc, ' ')
        return text.lower()

    def remove_stopwords(self, text):
        if self.stopwords != None:
            words = [word for word in text.split() if word not in self.stopwords]
            return ' '.join(words)
        else:
            return text

    def word_segment(self, text):
        return tokenize(text.encode('utf-8').decode('utf-8'))

    def preprocess(self, text):
        return self.remove_stopwords(self.normalize_text(self.word_segment(self.clean_text(text))))