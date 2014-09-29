#!/usr/bin/env python -O
#
# Copyright (c) 2013-2014 Kyle Gorman
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
#
# confusion.py: confusion matrices and summary statistics


from math import sqrt
from functools import partial
from collections import defaultdict


NAN = float("nan")
INF = float("inf")


class Accuracy(object):

    """
    Accuracy measure for classification task
    """

    def __init__(self, correct=0, incorrect=0):
        self.correct = correct
        self.incorrect = incorrect

    def __repr__(self):
        return "{}(correct={}, incorrect={})".format(
            self.__class__.__name__, self.correct, self.incorrect)

    def __len__(self):
        return self.correct + self.incorrect

    def update(self, truth, guess):
        if truth == guess:
            self.correct += 1
        else:
            self.incorrect += 1

    def batch_update(self, truths, guesses):
        for (truth, guess) in zip(truths, guesses):
            self.update(truth, guess)

    def __add__(self, right):
        """
        Combine two accuracy objects
        """
        if type(right) is not Accuracy:
            raise TypeError("Unsupported operand type(s) for '+':" +
                            "{} and {}.".format(type(self), type(right)))
        return Accuracy(self.correct + right.correct,
                        self.incorrect + right.incorrect)

    @property
    def accuracy(self):
        return self.correct / len(self)


class BinaryConfusion(object):

    """
    Binary confusion matrix, including summary statistics
    """

    def __init__(self, hit=True, tp=0, fp=0, fn=0, tn=0):
        self.hit = hit
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def __repr__(self):
        return self.__class__.__name__ + \
            "(tp={}, fp={}, fn={}, tn={})".format(self.tp, self.fp,
                                                  self.fn, self.tn)

    def pprint(self):
        """
        >>> cm = BinaryConfusion()
        >>> cm.tp = 5809125
        >>> cm.tn = 2235458
        >>> cm.fp = cm.fn = 1
        >>> cm.pprint()
        Truth | Guess
        ---------------------------------------
              |       Hit         Miss
         Hit  | 5,809,125            1
         Miss |         1    2,235,458
        """
        print("""Truth | Guess
---------------------------------------
      |       Hit         Miss
 Hit  | {:>9,}    {:>9,}
 Miss | {:>9,}    {:>9,}""".format(self.tp, self.fn, self.fp, self.tn))

    def __len__(self):
        return self.tp + self.fp + self.fn + self.tn

    def update(self, truth, guess):
        if truth == self.hit:
            if guess == self.hit:
                self.tp += 1
            else:
                self.fn += 1
        elif guess == self.hit:
            self.fp += 1
        else:
            self.tn += 1

    def __add__(self, right):
        """
        Combine two binary confusion matrices
        """
        if type(right) is not BinaryConfusion:
            raise TypeError("Unsupported operand type(s) for '+':" +
                            "{} and {}.".format(type(self), type(right)))
        if self.hit != right.hit:
            raise ValueError("Operands do not have matching 'hit' label.")
        return BinaryConfusion(hit=self.hit,
                               tp=self.tp + right.tp,
                               fp=self.fp + right.fp,
                               fn=self.fn + right.fn,
                               tn=self.tn + right.tn)

    # aggregate scores

    @property
    def accuracy(self):
        try:
            return (self.tp + self.tn) / len(self)
        except ZeroDivisionError:
            return NAN

    @property
    def Kappa(self):
        """
        Cohen's Kappa, a popular interannotator agreement statistic
        """
        # probability the two sources say yes
        if len(self) == 0:
            return NAN
        Px = (self.tp + self.fp) / len(self)
        Py = (self.tp + self.fn) / len(self)
        # probability of chance agreement
        Pe = (Px * Py) + ((1. - Px) * (1. - Py))
        return (self.accuracy - Pe) / (1. - Pe)

    def Fscore(self, ratio=1.):
        """
        F-score, by default F_1; ratio is the importance of recall vs.
        precision
        """
        assert ratio > 0.
        r_square = ratio * ratio
        P = self.precision
        R = self.recall
        return ((1. + r_square) * P * R) / (r_square * P + R)

    @property
    def F1(self):
        return self.Fscore()

    def Sscore(self, ns_ratio=1.):
        """
        Same idea as F-score, but defined in terms of specificity and
        sensitivity; ratio is the importance of specificity vs. sensitivity
        """
        assert ratio > 0.
        r_square = ratio * ratio
        Sp = self.specificity
        Se = self.sensitivity
        return ((ns_ratio + r_square) * Sp * Se) / (r_square * Sp * Se)

    @property
    def S1(self):
        return self.Sscore()

    @property
    def MCC(self):
        try:
            N = len(self)
            S = (self.tp + self.fn) / N
            P = (self.tp + self.fp) / N
            PS = P * S
            denom = sqrt(PS * (1. - S) * (1. - P))
            return ((self.tp / N) - PS) / denom
        except ZeroDivisionError:
            return NAN

    # precision

    @property
    def precision(self):
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return INF

    @property
    def PPV(self):
        return self.precision

    # recall

    @property
    def recall(self):
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return INF

    @property
    def sensitivity(self):
        return self.recall

    @property
    def TPR(self):
        return self.recall

    # specificity

    @property
    def specificity(self):
        try:
            return self.tp / (self.fp + self.tn)
        except ZeroDivisionError:
            return INF

    @property
    def TNR(self):
        return self.specificity

    # others, rarely used

    @property
    def FPR(self):
        try:
            return self.fp / (self.fp + self.tn)
        except ZeroDivisionError:
            return INF

    @property
    def NPV(self):
        try:
            return self.tn / (self.tn + self.fn)
        except ZeroDivisionError:
            return INF

    @property
    def FDR(self):
        return 1. - self.PPV


class Confusion(object):

    """
    Generic confusion matrix
    """

    def __init__(self):
        self.matrix = defaultdict(partial(defaultdict, int))

    def __len__(self):
        return sum(len(guesses) for guesses in self.matrix.values())

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def pprint(self):
        print("Confusion matrix:")
        for (truth, guess_count) in self.matrix.items():
            print("{}:".format(truth))
            for (guess, count) in guess_count.items():
                print("\t{}: {:,}".format(guess, count))

    def update(self, truth, guess, k=1):
        self.matrix[truth][guess] += k

    def __add__(self, right):
        """
        Combine two Confusion instances
        """
        if type(right) is not Confusion:
            raise TypeError("Unsupported operand type(s) for '+':" +
                            "{} and {}.".format(type(self), type(right)))
        # construct new Confusion instance
        retval = Confusion()
        retval.matrix = self.matrix.copy()
        for (truth, guess_count) in right.matrix.items():
            truth_ptr = retval.matrix[truth]
            for (guess, count) in guess_count.items():
                truth_ptr[guess] += count
        return retval

    @property
    def accuracy(self):
        length = correct = 0
        for (truth, guess_count) in self.matrix.items():
            for (guess, count) in guess_count.items():
                if truth == guess:
                    correct += count
                length += count
        return correct / length

if __name__ == "__main__":
    import doctest
    doctest.testmod()
