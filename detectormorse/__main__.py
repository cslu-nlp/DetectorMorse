# Copyright (c) 2014 Kyle Gorman <gormanky@ohsu.edu>
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


import logging

from argparse import ArgumentParser


from .detector import Detector, slurp, EPOCHS, default_model

# Sentinel for reading in the default model
_READ_DEFAULT = object()


LOGGING_FMT = "%(message)s"


argparser = ArgumentParser(prog="python -m detectormorse",
                           description="Detector Morse")
vrb_group = argparser.add_mutually_exclusive_group()
vrb_group.add_argument("-v", "--verbose", action="store_true",
                       help="enable verbose output")
vrb_group.add_argument("-V", "--really-verbose", action="store_true",
              help="enable even more verbose output")
inp_group = argparser.add_mutually_exclusive_group(required=True)
inp_group.add_argument("-t", "--train", help="training data")
inp_group.add_argument("-r", "--read", nargs='?', const=_READ_DEFAULT,
              help="read in a serialized model from a path or read "
                       "the default model if no path is specified")
out_group = argparser.add_mutually_exclusive_group(required=True)
out_group.add_argument("-s", "--segment", help="segment sentences")
out_group.add_argument("-w", "--write",
              help="write out serialized model")
out_group.add_argument("-e", "--evaluate",
              help="evaluate on segmented data")
argparser.add_argument("-E", "--epochs", type=int, default=EPOCHS,
              help="# of epochs (default: {})".format(EPOCHS))
argparser.add_argument("-C", "--nocase", action="store_true",
              help="disable case features")
argparser.add_argument("--preserve-whitespace", action="store_true",
                       help="preserve whitespace when segmenting")
args = argparser.parse_args()
# verbosity block
if args.really_verbose:
    logging.basicConfig(format=LOGGING_FMT, level="DEBUG")
elif args.verbose:
    logging.basicConfig(format=LOGGING_FMT, level="INFO")
else:
    logging.basicConfig(format=LOGGING_FMT)
# input block
detector = None
if args.train:
    logging.info("Training model on '{}'.".format(args.train))
    detector = Detector(slurp(args.train), epochs=args.epochs,
                        nocase=args.nocase)
elif args.read:
    if args.read is _READ_DEFAULT:
        # If sentinel specified, load default model
        logging.info("Reading default model.")
        detector = default_model()
    else:
        # Otherwise load normally
        logging.info("Reading pretrained model '{}'.".format(args.read))
        detector = Detector.load(args.read)
# output block
if args.segment:
    logging.info("Segmenting '{}'.".format(args.segment))
    print("\n".join(detector.segments(slurp(args.segment),
                                      strip=not args.preserve_whitespace)))
if args.write:
    logging.info("Writing model to '{}'.".format(args.write))
    detector.dump(args.write)
elif args.evaluate:
    logging.info("Evaluating model on '{}'.".format(args.evaluate))
    cx = detector.evaluate(slurp(args.evaluate))
    if args.verbose or args.really_verbose:
        cx.pprint()
    print(cx.summary)
