import sys
sys.path.append('../')

import perc
import optparse, os
from bilstmcrf.util import *
from bilstmcrf.BiLSTM_CRF import BiLSTM_CRF
from bilstmcrf import bilstmcrf_config as config
from datetime import datetime
import os.path as osp

import torch
import torch.optim as optim


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("../data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("../data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("../data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("models", "default.model"), help="weights for all features stored on disk")
    optparser.add_option('-v', '--valfile', dest='valfile', default=os.path.join("../data", "dev.txt"), help='validation data')
    optparser.add_option("--vf", dest="valfeatfile", default=os.path.join("../data", "dev.feats"), help='validation feature')
    optparser.add_option('--ckpt', dest='ckpt', default='ckpt', help='check point dir')
    (opts, _) = optparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not osp.exists(opts.ckpt):
        os.mkdir(opts.ckpt)
    if not osp.exists('models'):
        os.mkdir('models')

    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile, verbose=False)
    test_data = perc.read_labeled_data(opts.valfile, opts.valfeatfile, verbose=False)
    print("done.", file=sys.stderr)

    word_idx, speech_tag_idx = build_vocab(train_data)
    tag2idx, idx2tag = build_tag_index(tagset)

    if config.prototyping_mode:
        train_data = train_data[1:32]

    print("preparing training data...", file=sys.stderr)
    training_tuples = prepare_training_data(train_data, speech_tag_idx, tag2idx)
    print('preparing testing data...')
    # test_tuples = prepare_test_data(test_data, speech_tag_idx)
    print('Done')
    print("initializing BiLSTM-CRF model... ", file=sys.stderr)
    model = BiLSTM_CRF(len(word_idx), len(speech_tag_idx), len(tag2idx), device)
    print('Done')

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    model.to(device)
    print('Start training')
    train_start_t = datetime.now()

    best_model = None
    best_test_error = None

    for epoch in range(opts.numepochs):
        running_loss = 0.0
        for i, (input_seq, input_tag, target_tag) in enumerate(training_tuples):
            input_seq = input_seq.to(device)
            input_tag = input_tag.to(device)
            target_tag = target_tag.to(device)
            # initialize hidden state and grads before each step.
            model.zero_grad()

            loss = model.NLLloss(input_seq, input_tag, target_tag)
            running_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

            if (i+1) % 100 == 0:
                running_loss /= 100
                print('[Epoch {:3}, iteration {:6}] loss: {}'.format(epoch+1, i+1, running_loss))
                running_loss = 0


        # test_error = test_model(model, test_tuples, idx2tag, cuda)
        # if best_test_error is None or best_test_error > test_error:
        #     best_test_error = test_error
        #     best_model = model.state_dict()
        # print(f"epoch {epoch+1} done. Validation loss = {val_loss}",
        #       file=sys.stderr)
        dump_model(model.state_dict(), word_idx, speech_tag_idx, tag2idx, idx2tag,
                   osp.join(opts.ckpt, 'ckpt_e{}.model'.format(epoch+1)))

    dump_model(model.state_dict(), word_idx, speech_tag_idx, tag2idx, idx2tag, opts.modelfile)
    print('Training completed in {}'.format(datetime.now() - train_start_t))