from nltk.corpus import stopwords
import unicodedata
import nltk
import re
class Preprocessor:

    def clean_tokens(words):
        word = re.sub('([\.\, \\/ \\" \\^\\\'\_\´\`\’\@\#\$\%\?\!\:\;\*-\=\+\{\}]+)', '', words)
        word = re.sub('(etc|[0-9]{1,4}|decada|seculo|a{2,200}|a{2,200}h{2,200}|a{2,200}rgh|-{1,200})+', '', word)
        return word

    def corpus(data, feature, target):
        corpus_sentences = []
        corpus_words = []
        remove_stopwords = stopwords.words('portuguese')
        # cogroo = Cogroo.Instance()
        for idx, text_row in data.iterrows():
            sentences = re.sub('(\\u200b)+', '', text_row[feature])
            sentences = re.sub('(\.{3})', '.', sentences)
            sentences = re.sub('([0-9]{1,2}\/[0-9]{1,2})', '', sentences)
            sentences = unicodedata.normalize('NFD', sentences).encode('ascii', 'ignore').decode('utf-8')
            sentences = nltk.sent_tokenize(sentences.lower())
            for sent in sentences:
                corpus_sentences.append((sent, text_row[target]))
                corpus_words.append((nltk.word_tokenize(sent), text_row[target]))
        corpus_tokens = [([Preprocessor.clean_tokens(word) for word in words[0] if len(word) > 1 and word not in remove_stopwords], words[1]) for words in corpus_words if len(words) > 0]
        return corpus_tokens
