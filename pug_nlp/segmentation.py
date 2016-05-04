import os

from .detector_morse import Detector
from .detector_morse import slurp
from .penn_treebank_tokenizer import word_tokenize
words = word_tokenize

import nlup

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def generate_sentences(text='', train_path=None, case_sensitive=True, epochs=20, classifier=nlup.BinaryAveragedPerceptron, **kwargs):
    """Generate sentences from a sequence of characters (text)

    Thin wrapper for Kyle Gorman's "DetectorMorse" module

    Arguments:
      case_sensitive (int): whether to consider case to make decisions about sentence boundaries
      epochs (int): number of epochs (iterations for classifier training)

    """
    if train_path:
        generate_sentences.detector = Detector(slurp(train_path), epochs=epochs, nocase=not case_sensitive)
    # generate_sentences.detector = SentenceDetector(text=text, nocase=not case_sensitive, epochs=epochs, classifier=classifier)
    return iter(generate_sentences.detector.segments(text))
generate_sentences.detector = nlup.decorators.IO(Detector.load)(os.path.join(DATA_PATH, 'wsj_detector_morse_model.json.gz'))


def generate_words(text='', train_path=None, case_sensitive=True, epochs=20, classifier=nlup.BinaryAveragedPerceptron, **kwargs):
    """Generate sentences from a sequence of characters (text)

    Thin wrapper for Kyle Gorman's "DetectorMorse" module

    Arguments:
      case_sensitive (int): whether to consider case to make decisions about sentence boundaries
      epochs (int): number of epochs (iterations for classifier training)

    """
    if train_path:
        generate_sentences.detector = Detector(slurp(train_path), epochs=epochs, nocase=not case_sensitive)
    # generate_sentences.detector = SentenceDetector(text=text, nocase=not case_sensitive, epochs=epochs, classifier=classifier)
    return iter(generate_sentences.detector.segments(text))
generate_sentences.detector = nlup.decorators.IO(Detector.load)(os.path.join(DATA_PATH, 'wsj_detector_morse_model.json.gz'))
