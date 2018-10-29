import sys
sys.path.append('../')
import os
import perc
import optparse
from bilstmcrf import bilstmcrf_config as config
from bilstmcrf.util import *
from bilstmcrf.BiLSTM_CRF import BiLSTM_CRF, BiLSTM_Enc_Dec_CRF

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("../data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join("../data", "dev.txt"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("../data", "dev.feats"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("models", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)

    test_data = perc.read_labeled_data(opts.inputfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)
    if config.prototyping_mode:
        test_data = test_data[0:32]

    print('Loading model...', file=sys.stderr)
    model_data = load_model(opts.modelfile)

    word_idx = model_data['word_index']
    speech_tag_idx = model_data['speech_tag_index']
    tag2idx = model_data['tag_index']
    idx2tag = model_data['reverse_tag_index']

    model = BiLSTM_Enc_Dec_CRF(len(word_idx), len(speech_tag_idx), len(tag2idx), device)
    model.load_state_dict(model_data['model'])
    model.to(device)
    print('Done', file=sys.stderr)
    print('Preparing testing data...', file=sys.stderr)
    test_tuples = prepare_test_data(test_data, speech_tag_idx)
    print('Done', file=sys.stderr)
    predicted_tags = test_model(model, test_tuples, idx2tag, device)

    output = format_prediction(predicted_tags, test_data)
    print('F1 score:{:.5}'.format(compute_score(output)))
