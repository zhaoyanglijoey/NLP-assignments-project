# Get more sentences and output them to 

import re
import nltk

# run the following line only at the first time to download nltk.books
# nltk.download()

from nltk.book import *

# Extract allowed words from the file.

allowedWordsFile = open("allowed_words.txt", "r")
allowedWords = set([allowedWord.rstrip('\n').lower() for allowedWord in allowedWordsFile.readlines()])


def prohibitedWords(words):
  """
  Checks if words are allowed.
  """
  wordSet = set([word.lower() for word in words])
  return wordSet - allowedWords


def wordsToSentences(words):
  """
  Takes a list of words and returns a list of sentences (lists of words).
  """

  # First format the punctuation marks.
  # For example, split "?!" into "?" and "!"
  puncPattern = re.compile(r'[?!,.()"\'\[\]:;-]+')
  oldWords = words
  words = []
  for word in oldWords:
    if puncPattern.match(word):
      curWord = ""
      for char in word:
        if curWord:
          if char == curWord[0]:
            curWord += char
          else:
            words.append(curWord)
            curWord = ""
            curWord = curWord + char
        else:
          curWord = curWord + char
      words.append(curWord)
    else:
      words.append(word)

  # Then take out all the plots.
  # Exclude the following:
  # a. "SCENE 1 :"
  # b. "[ thud ]"
  # c. "SOLDIER # 1 :" or "HISTORIAN ' S WIFE" or "CART - MASTER :":
  plots = []
  curPlot = []
  potentialTitle = []
  isDescription = False
  isTitle = False
  for word in words:
    if word == "[":
      isDescription = True
      if curPlot:
        plots.append(curPlot)
      curPlot = []
      continue
    if word == "]":
      isDescription = False
      continue
    if isTitle:
      if word == ":":
        isTitle = False
        potentialTitle = []
        # The last plot terminates before the this title.
        if curPlot:
          plots.append(curPlot)
          curPlot = []
        continue
      elif word.isupper() or word.isdigit() or word == "#" or word == "," or word == "-":
        # still potentially inside a title (a character name or a scene title)
        potentialTitle.append(word)
        continue
      else:
        # actually inside a sentence
        isTitle = False
        curPlot = curPlot + potentialTitle
        potentialTitle = []
    elif word.isupper():
      isTitle = True
      potentialTitle = [word]
    if not isDescription and not isTitle:
      curPlot.append(word)

  if curPlot:
    plots.append(curPlot)

  # Finally, separate plots into sentences.
  sentenceTerminals = {"?", ".", "!"}
  sentences = []
  for plot in plots:
    l = len(plot)
    i = 0
    j = 1
    while j < l:
      if plot[j - 1] in sentenceTerminals and plot[j][0].isupper():
        # Sentences are not separated simply by punctuations in case there are consecutive puncs.
        # For example, "This is a sentence ! ? Yes, it is ... . Ha"
        sentences.append(plot[i : j])
        i = j
      j += 1
    sentences.append(plot[i : j])

  return sentences


def formattedSentence(words):
  return ' '.join(words)


f = open("moreSentences.txt", "w+")
for sentence in wordsToSentences(text6):
  # With only allowed words, we don't get many extra sentences than the devset...
  # if(prohibitedWords(sentence)):
  f.write(formattedSentence(sentence))
  f.write("\n")
f.close()