from detector_morse import Detector as SentenceDetector
from penn_treebank_tokenizer import word_tokenize as words


def generate_sentences(text=None, nocase=NOCASE, epochs=EPOCHS, classifier=BinaryAveragedPerceptron, **kwargs):
    detector = SentenceDetector(text=text, nocase=nocase, epochs=epochs, classifier=classifier)