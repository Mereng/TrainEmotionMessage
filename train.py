from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer
from collections import Counter
from tflearn.data_utils import to_categorical
from sklearn.model_selection import train_test_split
import re, numpy, tensorflow, tflearn


class TweetTrain:
    def __init__(self, vocabulary_size=5000, debug=False):
        self.stemmer = RussianStemmer()
        self.stem_count = Counter()
        self.validator_regex = re.compile(r'[^А-яЁё]')
        self.cache_stems = {}
        self.vocabulary = None
        self.vocabulary_size = vocabulary_size
        self.debug = debug
        self.positive_tweets = None
        self.negative_tweets = None
        self.tweets_vectors = None
        self.labels = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def get_stem(self, token):
        stem = self.cache_stems.get(token, None)
        if stem:
            return stem
        token = self.validator_regex.sub('', token).lower()
        stem = self.stemmer.stem(token)
        self.cache_stems[token] = stem
        return stem

    def count_unique_stem(self, tweets):
        tokenizer = TweetTokenizer()
        for tweet in tweets:
            tokens = tokenizer.tokenize(tweet)
            for token in tokens:
                stem = self.get_stem(token)
                self.stem_count[stem] += 1

    def create_vocabulary(self):
        vocabulary_sort = sorted(self.stem_count, key=self.stem_count.get, reverse=True)[:self.vocabulary_size]
        self.vocabulary = {vocabulary_sort[i] : i for i in range(self.vocabulary_size)}

    def tweet_to_vector(self, tweet):
        vector = numpy.zeros(self.vocabulary_size, dtype=numpy.byte)
        tokenizer = TweetTokenizer()
        for token in tokenizer.tokenize(tweet):
            stem = self.get_stem(token)
            idx = self.vocabulary.get(stem, None)
            if idx is not None:
                vector[idx] = 1
            elif self.debug:
                print("Не известное слово", token)
        return vector

    def prepare_data(self, positive_file, negative_file):
        self.positive_tweets = open(positive_file).read().split('\n')
        self.negative_tweets = open(negative_file).read().split('\n')

        print('Считаем количество уникальных слов...')
        self.count_unique_stem(self.positive_tweets)
        self.count_unique_stem(self.negative_tweets)
        print('Количетство уникальных слов:', len(self.stem_count))

        print('Создаем словарь из', self.vocabulary_size, 'слов...')
        self.create_vocabulary()
        print('Словарь создан')

        self.tweets_vectors = numpy.zeros((len(self.negative_tweets) + len(self.positive_tweets), self.vocabulary_size),
                                     dtype=numpy.byte)
        print('Создаем вектор из негативных постов')
        for i, tweet in enumerate(self.negative_tweets):
            self.tweets_vectors[i] = self.tweet_to_vector(tweet)

        print('Создаем вектор из позитивных постов')
        for i, tweet in enumerate(self.positive_tweets):
            self.tweets_vectors[i + len(self.negative_tweets)] = self.tweet_to_vector(tweet)
        self.labels = numpy.append(numpy.zeros(len(self.negative_tweets), dtype=numpy.byte),
                                   numpy.ones(len(self.positive_tweets), dtype=numpy.byte))
        print('Подготовка завешена')

    def build_model(self, learning_rate=0.1):
        tensorflow.reset_default_graph()
        net = tflearn.input_data([None, self.vocabulary_size])
        net = tflearn.fully_connected(net, 125, activation='RelU')
        net = tflearn.fully_connected(net, 25, activation='RelU')
        net = tflearn.fully_connected(net, 2, activation='softmax')

        tflearn.regression(net, optimizer='sgd', learning_rate=learning_rate)
        return tflearn.DNN(net)

    def create_train_test_data(self):
        x = self.tweets_vectors
        y = to_categorical(self.labels, 2)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3)

    def train(self):
        self.model = self.build_model(0.75)
        self.model.fit(self.x_train, self.y_train, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=30)
        self.model.save('model.tflearn')

    def load_model(self):
        self.model = self.build_model(0.75)
        self.model.load('model.tflearn')

    def test_model(self):
        if self.model is None:
            self.load_model()
        predictions = (numpy.array(self.model.predict(self.x_test))[:, 0] >= 0.5).astype(numpy.byte)
        accuracy = numpy.mean(predictions == self.y_test[:, 0], axis=0)
        print("Точность:", accuracy)

if __name__ == '__main__':
    foo = TweetTrain()
    foo.prepare_data('data/positive_clear.txt', 'data/negative_clear.txt')