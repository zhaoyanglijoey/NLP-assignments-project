import sys
import perc
import optparse, os
from bilstmcrf.util import *
from bilstmcrf.BiLSTM_CRF import BiLSTM_CRF, BiLSTM_Enc_Dec_CRF
from datetime import datetime
import os.path as osp

import torch
import torch.optim as optim
import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    argparser.add_argument("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    argparser.add_argument("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    argparser.add_argument("-e", "--numepochs", dest="numepochs", default=int(10), type=int, help="number of epochs of training; in each epoch we iterate over over all the training examples")
    argparser.add_argument("-m", "--modelfile", dest="modelfile", default=os.path.join("models", "default.model"), help="weights for all features stored on disk")
    argparser.add_argument('-v', '--valfile', dest='valfile', default=os.path.join("data", "dev.txt"), help='validation data')
    argparser.add_argument("--vf", dest="valfeatfile", default=os.path.join("data", "dev.feats"), help='validation feature')
    argparser.add_argument('--ckpt', dest='ckpt', default='ckpt', help='check point dir')
    argparser.add_argument('-lr', dest='lr', type=float, default=0.01, help='learning rate')
    argparser.add_argument('-hd', dest='hidden', type=int, default=600, help='hidden dimension')
    argparser.add_argument('-ly', dest='layer', type=int, default=2, help='number of layers')
    argparser.add_argument('--pos-dim', type=int, default=64, help='POS tag embedding dimension')
    argparser.add_argument('-r', '--resume', help='resume training from saved model')
    argparser.add_argument('--prototype', default=False, action='store_true', help='prototyping mode')
    args= argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not osp.exists(args.ckpt):
        os.mkdir(args.ckpt)
    if not osp.exists('models'):
        os.mkdir('models')

    tagset = perc.read_tagset(args.tagsetfile)
    print("reading data ...", file=sys.stderr)
    train_data = perc.read_labeled_data(args.trainfile, args.featfile, verbose=False)
    test_data = perc.read_labeled_data(args.valfile, args.valfeatfile, verbose=False)
    print("done.", file=sys.stderr)

    word_idx, speech_tag_idx = build_vocab(train_data)
    tag2idx, idx2tag = build_tag_index(tagset)
    if args.prototype:
        train_data = train_data[1:8]
        test_data = test_data[1:8]

    print("preparing training data...", file=sys.stderr)
    training_tuples = prepare_training_data(train_data, speech_tag_idx, tag2idx)
    print('preparing testing data...')
    test_tuples = prepare_test_data(test_data, speech_tag_idx)
    print('done.')
    print("initializing BiLSTM-CRF model...", file=sys.stderr)
    model = BiLSTM_Enc_Dec_CRF(len(speech_tag_idx), len(tag2idx), device,
                               args.layer, args.hidden, args.pos_dim)
    print('done.')
    if args.resume:
        print('loading model...', file=sys.stderr)
        model_data = load_model(args.resume)

        word_idx = model_data['word_index']
        speech_tag_idx = model_data['speech_tag_index']
        tag2idx = model_data['tag_index']
        idx2tag = model_data['reverse_tag_index']
        model.load_state_dict(model_data['model'])

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.to(device)
    print('start training...')
    train_start_t = datetime.now()

    best_score = 0
    best_epoch = None
    # save_model_path = osp.join('models', 'h{}layer{}lr{}enc.model'.format(
    #     args.hidden, args.layer, args.lr))
    save_model_path = args.modelfile
    steps_to_print = 500
    ref_file = osp.join('data', 'reference500.txt')
    for epoch in range(args.numepochs):
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

            if (i+1) % steps_to_print == 0:
                running_loss /= steps_to_print
                print('[Epoch {:3}, iteration {:6}] loss: {}'.format(epoch+1, i+1, running_loss))
                running_loss = 0

        predicted_tags = test_model(model, test_tuples, idx2tag, device)
        output = format_prediction(predicted_tags, test_data)
        f1score = compute_score(output, ref_file)
        if f1score > best_score:
            best_score = f1score
            best_epoch = epoch+1
            dump_model(model.state_dict(), word_idx, speech_tag_idx, tag2idx, idx2tag, save_model_path)
            print('best model so far saved at', save_model_path)

        print(f"epoch {epoch+1} done. F1 score = {f1score}",
              file=sys.stderr)
        save_ckpt_path = osp.join(args.ckpt, 'ckpt_e{}.model'.format(epoch+1))
        dump_model(model.state_dict(), word_idx, speech_tag_idx, tag2idx, idx2tag, save_ckpt_path)
        print('check-point model saved at', save_ckpt_path)

    print('Training completed in {}, best F1 score {} obtained after {} epochs. Model saved at {}'.
          format(datetime.now() - train_start_t, best_score, best_epoch, save_model_path))