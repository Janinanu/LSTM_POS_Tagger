#! /usr/bin/python3

import torch.nn as nn
from torch import autograd, LongTensor  # only needed for unit test


class TaggerModel(nn.Module):
    """
    Class that defines the layers in the model according to hyperparameters
    """
    def __init__(self, numWords, numTags, embSize, rnnSize, dropoutRate):

        super(TaggerModel, self).__init__()
        # one additional row in embedding matrix to represent ID 0 for unknown words
        self.embedding = nn.Embedding(numWords+1, embSize)
        self.lstm = nn.LSTM(embSize, rnnSize, batch_first = True, bidirectional = True)
        self.dropout = nn.Dropout(dropoutRate)
        self.linear = nn.Linear(rnnSize*2, numTags) #linear layer compresses rnnSize*2 to numTags

    def forward(self, inputs):
        """
        Implements forward pass of data through the network
        :param inputs: a LongTensor of IDs representing the words in a sentence
        :return: tensor containing the scores across all classes for each word in the sentence
        """
        embeddings = self.dropout(self.embedding(inputs))  # embedding output: seqlen x embSize
        # input to lstm has to have a batch dimension, therefore unsqueze to 1 x seqlen x embSize
        lstm_output, _ = self.lstm(embeddings.unsqueeze(0))  # output of lstm: 1 x seqlen x rnnSize*2
        # input to linear: 1 x seqlen x rnnSize*2,
        tagScores = self.linear(self.dropout(lstm_output)).squeeze(0) # output of linear: 1 x seqlen x numTags
        # squeeze tagScores to seqlen x numTags as input for crossentropy loss
        return tagScores

def run_test():
    """
    Creates tagger model object and tests its functionality
    for some dummy word IDs and blindly chosen hyperparameter values
    """
    test_ids = autograd.Variable(LongTensor([65, 7, 1, 14]))
    tagger = TaggerModel(numWords=100, numTags=100, embSize=100, rnnSize=100, dropoutRate=0.5, use_gpu=False)
    tagScores = tagger(test_ids)

    if tagScores.size() == (len(test_ids), 100): #model output should be of dimension seqlen x numTags
        print("TaggerModel module passed test.")

if __name__ == "__main__":
    run_test()
