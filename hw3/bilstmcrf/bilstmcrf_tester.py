import sys
sys.path.append('../')
import os
import perc
import argparse
from bilstmcrf.util import *
from bilstmcrf.BiLSTM_CRF import BiLSTM_CRF, BiLSTM_Enc_Dec_CRF

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("../data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    argparser.add_argument("-i", "--inputfile", dest="inputfile", default=os.path.join("../data", "dev.txt"), help="input data, i.e. the x in \phi(x,y)")
    argparser.add_argument("-f", "--featfile", dest="featfile", default=os.path.join("../data", "dev.feats"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    argparser.add_argument("-m", "--modelfile", dest="modelfile", default=os.path.join("models", "default.model"), help="weights for all features stored on disk")
    argparser.add_argument('-hd', dest='hidden', type=int, default=600, help='hidden dimension')
    argparser.add_argument('-ly', dest='layer', type=int, default=2, help='number of layers')
    argparser.add_argument('--pos-dim', type=int, default=64, help='POS tag embedding dimension')
    argparser.add_argument('-r', '--resume', help='resume training from saved model')
    argparser.add_argument('--prototype', default=False, action='store_true', help='prototyping mode')
    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tagset = perc.read_tagset(args.tagsetfile)
    print("reading data ...", file=sys.stderr)

    test_data = perc.read_labeled_data(args.inputfile, args.featfile, verbose=False)
    print("done.", file=sys.stderr)
    if args.prototype:
        test_data = test_data[0:8]

    print('loading model...', file=sys.stderr)
    model_data = load_model(args.modelfile)

    word_idx = model_data['word_index']
    speech_tag_idx = model_data['speech_tag_index']
    tag2idx = model_data['tag_index']
    idx2tag = model_data['reverse_tag_index']

    model = BiLSTM_Enc_Dec_CRF(len(speech_tag_idx), len(tag2idx), device,
                               args.layer, args.hidden, args.pos_dim)
    model.load_state_dict(model_data['model'])
    model.to(device)
    print('done.', file=sys.stderr)
    print('preparing testing data...', file=sys.stderr)
    test_tuples = prepare_test_data(test_data, speech_tag_idx)
    print('done.', file=sys.stderr)
    predicted_tags = test_model(model, test_tuples, idx2tag, device)

    output = format_prediction(predicted_tags, test_data)
    print(output)
