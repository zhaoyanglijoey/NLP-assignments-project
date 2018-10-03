import re
import argparse
import collections

initialWeight = 1
stepSize = 1

def startWithTag(tag):
  return "%-15s %s"%("S2", "_" + tag)

def endWithTag(tag):
  return "%-15s %s"%("_" + tag, tag)

def tagAFollowedByTagB(tagA, tagB):
  return "%-15s %-15s %s"%("_" + tagA, tagA, "_" + tagB)

def sanitize(nont):
  nont = nont.strip()
  if nont == '.':
    nont = 'PERIOD'
  if nont == ':':
    nont = 'COLON'
  if nont == ',':
    nont = 'COMMA'
  if nont == "''":
    nont = 'TWOSINGLEQUOTES'
  if nont == "``":
    nont = 'TWOGRAVES'
  if nont == '(':
    nont = '-LRB-'
  if nont == ')':
    nont = '-RRB-'
  return nont

def main(vocab_file, tree_file, s2_file):
    fileIn = open(vocab_file, "r")

    rules = fileIn.readlines()
    fileIn.close()
    tags = []
    for rule in rules:
      if rule.strip() == '':
        continue
      tag = re.split('\s+', rule)[1]
      if tag not in tags:
        tags.append(tag)

    # Initialization

    weights = collections.OrderedDict()
    for tag in tags:
      weights[startWithTag(tag)] = initialWeight
    for tagA in tags:
      weights[endWithTag(tagA)] = initialWeight
      for tagB in tags:
        weights[tagAFollowedByTagB(tagA, tagB)] = initialWeight

    # Count the weight from the tree file

    fileIn = open(tree_file, "r")
    count = 0
    lastTermTag = None
    for line in fileIn.readlines():
      i = 0
      l = len(line) - 1 # The last character is the new line
      while i < l:
        if line[i] == '(':
          # 1. We get to a terminal like (COLON ;) or
          # 2. We get to a non-terminal line (S
          count += 1
          curTag = ""
          i += 1
          if line[i] in {'(', ')'}:
            # Terminals (( () or () ))
            curTag = sanitize(line[i])
            if lastTermTag:
              weights[tagAFollowedByTagB(lastTermTag, curTag)] += stepSize
            else:
              weights[startWithTag(curTag)] += stepSize
            # print(str(lastTermTag) + " " + curTag, end=',')
            lastTermTag = curTag
            i += 3
            count -= 1
          else:
            while i < l and line[i] != ' ':
              curTag = curTag + line[i]
              i += 1
            if i < l:
              i += 1
              if line[i] != '(':
                # Terminals
                curTag = sanitize(curTag)
                if lastTermTag:
                  weights[tagAFollowedByTagB(lastTermTag, curTag)] += stepSize
                else:
                  weights[startWithTag(curTag)] += stepSize
                # print(str(lastTermTag) + " " + curTag, end=',')
                lastTermTag = curTag
                while i < l and line[i] != ')':
                  # We care only about the tags
                  i += 1
                count -= 1
              else:
                i -= 1
        elif line[i] == ')':
          count -= 1
        if count == 0 and lastTermTag:
          weights[endWithTag(lastTermTag)] += stepSize
          lastTermTag = None
          # print()
        i += 1

    fileIn.close()

    fileOut = open(s2_file, "w+")
    for rule in weights:
      fileOut.write("%-10d %s\n"%(weights[rule], rule))
    fileOut.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vocab", dest="vocab_file", required=True, help="input vocab gr file")
    ap.add_argument("-tree", "--tree", dest="tree_file", required=True, help="input tree file")
    ap.add_argument("-s2", "--s2", dest="s2_file", required=True, help="output s2 gr file")
    args = ap.parse_args()
    main(args.vocab_file, args.tree_file, args.s2_file)
