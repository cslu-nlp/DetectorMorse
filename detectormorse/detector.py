# Copyright (c) 2014-2016 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Sentence boundary detection object."""


import logging

from re import finditer
from re import match
from re import search
from collections import namedtuple

from nlup.confusion import Accuracy
from nlup.confusion import BinaryConfusion
from nlup.decorators import listify
from nlup.util import case_feature
from nlup.util import isnumberlike

from perceptronix import SparseBinomialClassifier

from .ptbtokenizer import word_tokenize


NOCASE = False  # Disable case-based features?
EPOCHS = 20     # Number of epochs (iterations for classifier training).
BUFSIZE = 32    # For reading in left and right contexts...see below.
CLIP = 8        # Clip numerical count feature values.
NFEATS = 2048   # Guestimate for number of features.

VOWELS = frozenset("AEIOUY")

QUOTE_TOKEN = "*QUOTE*"
NUMBER_TOKEN = "*NUMBER*"

PUNCT = r"((\.+)|([!?]))"
TARGET = PUNCT + r"(['`\")}\]]*)(\s+)"

LTOKEN = r"(\S+)\s*$"
RTOKEN = r"^\s*(\S+)"
NEWLINE = r"^\s*[\r\n]+\s*$"

QUOTE = r"^['`\"]+$"

Observation = namedtuple("Observation", ["L", "P", "R", "B", "end"])


def slurp(filename):
  """Slurps in a whole file."""
  with open(filename, "r") as source:
    return source.read()


class Detector(object):
  """Core sentence splitter class."""

  @classmethod
  def read(cls, filename, nocase=NOCASE):
    """Reads serialized model from disk.

    Args:
      filename: The filename of a serialized model.

    Returns:
      A Detector instance.
    """
    retval = cls.__new__(cls)
    retval.nocase = nocase
    retval.classifier = SparseBinomialClassifier.read(filename)
    return retval

  def __init__(self, *args, nocase=NOCASE, nfeats=NFEATS,
               classifier=SparseBinomialClassifier, **kwargs):
    self.nocase = nocase
    self.classifier = classifier(nfeats, *args, **kwargs)

  def __repr__(self):
    return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

  @staticmethod
  def candidates(text):
     """Identifies candidates and their contexts for feature extraction.

     Args:
       text: Input text string.

     Yields: Observation objects.
     """
     for Pmatch in finditer(TARGET, text):
       # The punctuation mark itself.
       P = Pmatch.group(1)
       # Is it a boundary?
       B = bool(match(NEWLINE, Pmatch.group(5)))
       # L & R.
       start = Pmatch.start()
       end = Pmatch.end()
       Lmatch = search(LTOKEN, text[max(0, start - BUFSIZE):start])
       if not Lmatch:  # When a line begins with '.'
         continue
       L = word_tokenize(" " + Lmatch.group(1))[-1]
       Rmatch = search(RTOKEN, text[end:end + BUFSIZE])
       if not Rmatch:  # When at the end of the file, usually
         continue
       R = word_tokenize(Rmatch.group(1) + " ")[0]
       # Complete observation.
       yield Observation(L, P, R, B, end)

  @listify
  def extract_one(self, L, P, R):
    """Extracts features from a context.

    Args:
      L: Left context string
      P: Punctuation string
      R: Right context string.

    Returns:
      A list of feature strings.
    """
    yield "*bias*"
    # L feature(s)
    if match(QUOTE, L):
      L = QUOTE_TOKEN
    elif isnumberlike(L):
      L = NUMBER_TOKEN
    else:
      yield "len(L)={}".format(min(len(L), CLIP))
      if "." in L:
        yield "L:*period*"
      if not self.nocase:
        cf = case_feature(R)
        if cf:
          yield "L:{}'".format(cf)
      L = L.upper()
      if not any(char in VOWELS for char in L):
        yield "L:*no-vowel*"
    L_feat = "L='{}'".format(L)
    yield L_feat
    # P feature.
    yield "P='{}'".format(P)
    # R feature(s).
    if match(QUOTE, R):
      R = QUOTE_TOKEN
    elif isnumberlike(R):
      R = NUMBER_TOKEN
    else:
      if not self.nocase:
        cf = case_feature(R)
        if cf:
          yield "R:{}'".format(cf)
      R = R.upper()
    R_feat = "R='{}'".format(R)
    yield R_feat
    # The L,R bigram.
    yield "{},{}".format(L_feat, R_feat)

  def fit(self, text, epochs=EPOCHS):
    """Trains the classifier for a fixed number of iterations.

    Args:
      text: Input text string.
      epochs: Numbef of training epochs (default: 20).
    """
    logging.info("Performing feature extraction")
    data = []
    for (L, P, R, B, _) in Detector.candidates(text):
      data.append((self.extract_one(L, P, R), B))
    logging.info("Performing training")
    # Actual training.
    for epoch in range(1, 1 + epochs):
      logging.debug("Starting epoch %d", epoch)
      cx = Accuracy()
      for (feature, outcome) in data:
        cx.outcome(self.classifier.train(feature, outcome))
      logging.debug("Epoch %d resubstitution accuracy: %.4f", epoch,
                    cx.accuracy)

  def predict(self, L, P, R):
    """Predicts the outcome for a given context.

    Args:
      L: Left context string
      P: Punctuation string
      R: Right context string.

    Returns:
      Whether or not the context is predicted to be a sentence boundary.
    """
    return self.classifier.predict(self.extract_one(L, P, R))

  def segments(self, text):
    """Segments a string of text given the current model.

    Args:
      text: Input text string.

    Yields:
      Segmented strings of text (i.e., join them on newline).
    """
    start = 0
    for (L, P, R, B, end) in Detector.candidates(text):
      if B:  # Passes when a newline is already present.
        continue
      if self.predict(L, P, R):
        yield text[start:end].rstrip()
        start = end
    yield text[start:].rstrip()

  def evaluate(self, text):
    """Constructs confusion matrix for applying current model to some data.

    Args:
      text: Input text string.

    Returns:
      A BinaryConfusion confusion matrix object.
    """
    cx = BinaryConfusion()
    for (L, P, R, gold, _) in Detector.candidates(text):
      guess = self.predict(L, P, R)
      cx.update(gold, guess)
      if not gold and guess:
        logging.debug("False pos.: L='%s', R='%s'", L, R)
      elif gold and not guess:
        logging.debug("False neg.: L='%s', R='%s'". L, R)
    return cx

  # Delegates all remaining attributes to underlying classifier. Some useful
  # attributes include:
  #
  # * averaged
  # * average
  # * write

  def __getattr__(self, name):
    return getattr(self.classifier, name)
