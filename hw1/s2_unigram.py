import re
import argparse

initialWeight = 1
stepSize = 0

def main(vocab_file, s2_file):
    fileIn = open(vocab_file, "r")
    fileOut = open(s2_file, "w+")

    rules = fileIn.readlines()
    cats = dict()
    for rule in rules:
      if rule.strip() == '':
        continue
      cat = re.split('\s+', rule)[1]
      if cat in cats:
        cats[cat] += stepSize
      else:
        cats[cat] = initialWeight

    fileOut.write("1     S2      _Word\n")
    fileOut.write("1     _Word   Word _Word\n")
    fileOut.write("1     _Word   Word\n")
    for cat in cats:
      fileOut.write("%-5d Word    %s\n"%(cats[cat], cat))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vocab", dest="vocab_file", required=True, help="input vocab gr file")
    ap.add_argument("-s2", "--s2", dest="s2_file", required=True, help="output s2 gr file")
    args = ap.parse_args()
    main(args.vocab_file, args.s2_file)
