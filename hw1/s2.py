import re

fileIn = open("dev_vocab.gr", "r")
fileOut = open("dev_s2.gr", "w+")

rules = fileIn.readlines()
cats = dict()
for rule in rules:
  cat = re.split('\s+', rule)[1]
  if cat in cats:
    cats[cat] += 1
  else:
    cats[cat] = 1

fileOut.write("1     S2      _Word\n")
fileOut.write("1     _Word   _Word _Word\n")
fileOut.write("1     _Word   Word\n")
for cat in cats:
  fileOut.write("%-5d Word    %s\n"%(cats[cat], cat))
