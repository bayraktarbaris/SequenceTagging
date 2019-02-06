# -*- coding: utf-8 -*-

import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tagger import Tagger


class TaggerLR(Tagger):
    '''
    logistic regression taggers
    all methods excep tag (single sentence) is here
    tag(sent) is feature (data_class fields) specific
    '''

    def __init__(self, params=None):
        '''
        '''
        super().__init__(params)
        # DataReaderLR class for features
        self.data_class = None
        self.tagger = LogisticRegression()
        self.vectorizer = DictVectorizer()

    def train(self, file_name):
        '''
        '''
        data = self.data_class(file_name)

        X = self.vectorizer.fit_transform(data.features)

        self.logger.debug(self.vectorizer.get_feature_names())
        self.logger.info(
            str(len(self.vectorizer.feature_names_)) +
            ' feature:value tuples are vectorized')

        self.tagger.fit(X, data.y)
        self.logger.info(
            'model is trained with ' +
            str(len(self.tagger.classes_)) + ' classes')
        self.logger.debug(self.tagger.classes_)

    def test(self, file_name, labels_to_remove=[]):
        '''
        test the sentences while tagging them
        '''
        data = self.data_class(file_name)

        tagged_sents = self.tag_sents(data.sentences)
        y_pred = [fields[-1] for tagged_sent in tagged_sents
                  for fields in tagged_sent]

        self.logger.debug(str(len(data.y)) + ' ' + str(len(y_pred)))
        self.logger.debug(str(data.y[:5]))
        self.logger.debug(str(y_pred[:5]))

        # mainly for removing O tag in NER, can also be used for other tags
        labels = list(self.tagger.classes_)
        if labels_to_remove:
            for l in labels_to_remove:
                labels.remove(l)

        self.logger.debug(labels)

        precision = precision_score(data.y, y_pred,
                                    average='micro', labels=labels)
        recall = recall_score(data.y, y_pred,
                              average='micro', labels=labels)
        f1 = f1_score(data.y, y_pred,
                      average='micro', labels=labels)
        accuracy = accuracy_score(data.y, y_pred)
        confusion = confusion_matrix(data.y, y_pred)

        self.logger.debug(str(precision) + ' ' + str(recall) + ' ' +
                          str(f1) + ' ' + str(accuracy))

        return [precision, recall, f1, accuracy, confusion, tagged_sents]

    def _to_empty_sent(self, sentence):
        '''
        convert token to [token, None, None]
        '''
        raise NotImplementedError("_to_empty_sent not implemented")

    def token_to_features(self, sentence, index):
        '''
        feature set for a token
        '''
        raise NotImplementedError("token_to_features not implemented")

    def tag(self, sentence):
        '''
        pos tagger for single sentence
        data_class contains 3 fields
        token_to_features extracts features for a single token(word)
        '''
        tagged_sent = self._to_empty_sent(sentence)
        for i in range(len(tagged_sent)):
            # make a list
            features = [self.token_to_features(tagged_sent, i)]
            feat_vector = self.vectorizer.transform(features)
            tag_pred = self.tagger.predict(feat_vector)[0]
            tagged_sent[i][-1] = tag_pred

        return tagged_sent

    def tag_sents(self, sentences):
        '''
        tag a list of tokenized sentences
        '''
        return [self.tag(sent) for sent in sentences]

    def save(self, file_name):
        '''
        save models for tagger
        '''
        f = open(file_name, 'wb')
        pickle.dump((self.tagger, self.vectorizer), f)
        f.close()
        self.logger.info('saved model to ' + file_name)

    def load(self, file_name):
        '''
        load models for tagger
        '''
        f = open(file_name, 'rb')
        self.tagger, self.vectorizer = pickle.load(f)
        f.close()
        self.logger.info('loaded model from ' + file_name)
