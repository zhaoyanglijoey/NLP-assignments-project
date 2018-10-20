import os
import perc
import optparse
import sys

import neural_model

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join("data", "dev.txt"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "dev.feats"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    feat_vec = {}


    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)

    test_data = perc.read_labeled_data(opts.inputfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)

    model_with_data = neural_model.load_model(opts.modelfile)
    predicted_tags = neural_model.test_all(model_with_data, test_data, tagset)
    for idx, _ in enumerate(predicted_tags):
        print("\n".join(perc.conll_format(predicted_tags[idx], test_data[idx][0])))
        print()
