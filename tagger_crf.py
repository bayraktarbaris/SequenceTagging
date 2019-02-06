# -*- coding: utf-8 -*-

import pickle
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn_crfsuite import CRF
from sklearn_crfsuite.utils import flatten
from tagger import Tagger


class TaggerCRF(Tagger):
    '''
    crf taggers
    all methods except tag_sents (list of sentences) is here
    tag_sents(sents) is feature (data_class fields) specific
    '''

    def __init__(self, params=None):
        '''
        '''
        super().__init__(params)
        # DataReaderCRF class for features
        self.data_class = None
        self.tagger = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def train(self, file_name):
        '''
        '''
        data = self.data_class(file_name)
        self.tagger.fit(data.features, data.labels)
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
        y_pred = [[fields[-1] for fields in tagged_sent]
                  for tagged_sent in tagged_sents]

        self.logger.debug(str(len(data.y)) + ' ' + str(len(y_pred)))
        self.logger.debug(str(data.y[:5]))
        self.logger.debug(str(y_pred[:5]))

        # flatten list of lists
        # from itertools import chain
        # list(chain.from_iterable(y))
        y_true_flat = flatten(data.y)
        y_pred_flat = flatten(y_pred)

        # mainly for removing O tag in NER, can also be used for other tags
        labels = list(self.tagger.classes_)
        if labels_to_remove:
            for l in labels_to_remove:
                labels.remove(l)

        precision = precision_score(y_true_flat, y_pred_flat,
                                    average='micro', labels=labels)
        recall = recall_score(y_true_flat, y_pred_flat,
                              average='micro', labels=labels)
        f1 = f1_score(y_true_flat, y_pred_flat,
                      average='micro', labels=labels)
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        confusion = confusion_matrix(y_true_flat, y_pred_flat)

        return [precision, recall, f1, accuracy, confusion, tagged_sents]

    def tag(self, sent):
        '''
        tag a single tokenized sentence
        '''
        return self.tag_sents([sent])[0]

    def _to_empty_sent(self, sentence):
        '''
        convert token to [token, None, None(, None)]
        '''
        raise NotImplementedError("_to_empty_sent not implemented")

    def sent_to_features(self, sentence):
        '''
        feature set for a sentence
        '''
        raise NotImplementedError("sent_to_features not implemented")

    def tag_sents(self, sentences):
        '''
        tag a list of tokenized sentences
        '''
        tagged_sents = [self._to_empty_sent(sent) for sent in sentences]
        features = [self.sent_to_features(sent) for sent in tagged_sents]
        tags = self.tagger.predict(features)

        for i, sent in enumerate(tagged_sents):
            for j, _ in enumerate(sent):
                tagged_sents[i][j][-1] = tags[i][j]

        self.logger.debug(tagged_sents[0])
        return tagged_sents

    def save(self, file_name):
        '''
        save models for tagger
        '''
        f = open(file_name, 'wb')
        pickle.dump(self.tagger, f)
        f.close()
        self.logger.info('saved model to ' + file_name)

    def load(self, file_name):
        '''
        load models for tagger
        '''
        f = open(file_name, 'rb')
        self.tagger = pickle.load(f)
        f.close()
        self.logger.info('loaded model from ' + file_name)
