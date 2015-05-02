from .detector_morse import Detector as SentenceDetector
from .penn_treebank_tokenizer import word_tokenize
words = word_tokenize

import nlup


def generate_sentences(text='', case_sensitive=True, epochs=20, classifier=nlup.BinaryAveragedPerceptron, **kwargs):
    """Generate sentences from a sequence of characters (text)

    Thin wrapper for Kyle Gorman's "DetectorMorse" module

    Arguments:
      case_sensitive (int): whether to consider case to make decisions about sentence boundaries
      epochs (int): number of epochs (iterations for classifier training)

    """
    detector = SentenceDetector(text=text, nocase=not case_sensitive, epochs=epochs, classifier=classifier)
    return iter(detector.segments(text))
