#! /usr/bin/python3

from TaggerModel import TaggerModel
from Data import Data
import torch
from datetime import datetime
import random
import argparse

def log(string):
    """
    Print timestamp and description of action taken
    :param string: the description of what is happening
    """
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), string)

def print_results(epoch, num_epochs, train_loss, train_accuracy, dev_loss, dev_accuracy):
    """
    Print training and validation statistics
    :param epoch: current epoch
    :param num_epochs: total number of epochs
    :param train_loss: current training loss
    :param train_accuracy: current training accuracy
    :param dev_loss: current development loss
    :param dev_accuracy: current development accuracy
    """
    log("Epoch: %d/%d" % (epoch, num_epochs))
    log("TrainLoss: %.3f " % train_loss +
        "TrainAccuracy: %.3f " % train_accuracy +
        "DevLoss: %.3f " % dev_loss +
        "DevAccuracy: %.3f " % dev_accuracy)

def train(sentences):
    """
    Loop over the training sentences to do forward and backward propagation
    and to calculate train loss and train accuracy
    :param sentences: the list of training sentences
    :return: the average loss over all examples per epoch, the accuracy over all examples per epoch
    """
    model.train()  # set to training mode to enable dropout
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_func = torch.nn.CrossEntropyLoss(size_average=False)
    sum_loss = 0
    sum_correct = 0
    total_tags = 0
    random.shuffle(sentences)
    for words, tags in sentences:
        tagIDs = data.tags2IDs(tags) #returns Variable of dimension: seqlen
        tagScores = model(data.words2IDs(words)) #returns Variable of dimension: seqlen x numTags
        cur_loss = loss_func(tagScores, tagIDs)
        # convert loss to integer, use .cpu() in case calculations were done on gpu (does nothing otherwise)
        sum_loss += cur_loss.data.cpu().numpy()
        cur_loss.backward()  # backpropagate to minimize the loss function w.r.t. to parameters
        optimizer.step()  # update parameters
        optimizer.zero_grad()  # set parameter gradients to zero for next iteration
        _, predictions = tagScores.max(dim=-1) #returns Variable of dimension: seqlen
        sum_correct += torch.sum((data.to_array(tagIDs) == data.to_array(predictions))) #torch.sum() needs tensors as input
        total_tags += len(tags)
    train_loss = sum_loss/len(sentences)
    train_accuracy = sum_correct/total_tags
    return train_loss, train_accuracy

def validate(sentences):
    """
    Loop over the development sentences to calculate development loss and development accuracy
    :param sentences: the list of development sentences
    :return: the average loss over all examples per epoch, the accuracy over all examples per epoch
    """
    model.eval() # set to evaluation mode to disable dropout
    loss_func = torch.nn.CrossEntropyLoss(size_average=False)
    sum_loss = 0
    sum_correct = 0
    total_tags = 0
    for words, tags in sentences:
        tagIDs = data.tags2IDs(tags)  # returns Variable of dimension: seqlen
        tagScores = model(data.words2IDs(words))  # returns Variable of dimension: seqlen x numTags
        cur_loss = loss_func(tagScores, tagIDs)
        # convert loss to integer, use .cpu() in case calculations were done on gpu (does nothing otherwise)
        sum_loss += cur_loss.data.cpu().numpy()
        # to compute epoch accuracy later:
        _, predictions = tagScores.max(dim=-1)  # returns Variable of dimension: seqlen
        sum_correct += torch.sum((data.to_array(tagIDs) == data.to_array(predictions)))  # torch.sum() requires tensors as input
        total_tags += len(tags)
    dev_loss = sum_loss / len(sentences)
    dev_accuracy = sum_correct / total_tags
    return dev_loss, dev_accuracy

def train_lstm(num_epochs, model_file):
    """
    Train model for a given number of epochs, print statistics
    and store model with best development accuracy
    :param num_epochs: number of training epochs
    :param model_file: file to store the best model
    """
    best_dev_accuracy = -1
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(data.trainSentences)
        dev_loss, dev_accuracy = validate(data.devSentences)
        # epoch counter starts at 0 --> add 1
        print_results(epoch+1, num_epochs, train_loss, train_accuracy, dev_loss, dev_accuracy)
        # save model if dev_accuracy was improved:
        if dev_accuracy > best_dev_accuracy:
            torch.save(model, model_file)
            best_dev_accuracy = dev_accuracy

arg_parser = argparse.ArgumentParser(
    description='LSTM-Tagger training script')
arg_parser.add_argument('traindata', type=str, help='a file containing the training data')
arg_parser.add_argument('devdata', type=str, help='a file containing the evaluation data')
arg_parser.add_argument('model_file', type=str,
                        help='filename prefix for storing model (model will be stored with suffix .rnn, ids for words and tags with .io)')
arg_parser.add_argument('--num_words', type=int, default=10000,
                        help='number of most frequent words to take into account')
arg_parser.add_argument('--emb_size', type=int, default=200,
                        help='length of embedding vectors')
arg_parser.add_argument('--rnn_size', type=int, default=200,
                        help='number of hidden units per time step')
arg_parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
arg_parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate')
arg_parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
arg_parser.add_argument('--gpu', action='store_true',
                        help='if this parameter is present, computation will be performed on gpu, otherwise on cpu')
args = arg_parser.parse_args()

log("Preparing data...")
data = Data(args.traindata, args.devdata, args.num_words, args.gpu)
log("Storing word2id and taglist in " + args.model_file + ".io")
data.store_parameters(args.model_file + ".io")

model = TaggerModel(args.num_words, data.numTags, args.emb_size, args.rnn_size, args.dropout_rate)
if args.gpu:  # move model to gpu
    model.cuda()

log("Starting tagger training")
train_lstm(args.num_epochs, args.model_file + ".rnn")
