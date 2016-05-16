import os
import re
import string
from itertools import chain

from .detector_morse import Detector
from .detector_morse import slurp
# from .penn_treebank_tokenizer import word_tokenize
import nlup

from pug.nlp.constant import DATA_PATH
from pug.nlp.util import generate_files
from pug.nlp.regex_patterns import CRE_TOKEN, RE_NONWORD


def list_ngrams(token_list, n=1, join=' '):
    """Return a list of n-tuples, one for each possible sequence of n items in the token_list

    Arguments:
      join (bool or str): if str, then join ngrom tuples on it before returning
         True is equivalent to join=' '
         default = True

    See: http://stackoverflow.com/a/30609050/623735

    >>> list_ngrams('goodbye cruel world'.split(), join=False)
    [('goodbye',), ('cruel',), ('world',)]
    >>> list_ngrams('goodbye cruel world'.split(), 2, join=False)
    [('goodbye', 'cruel'), ('cruel', 'world')]
    """
    join = ' ' if join is True else join
    if isinstance(join, basestring):
        return [join.join(ng) for ng in list_ngrams(token_list, n=n, join=False)]
    return zip(*[token_list[i:] for i in range(n)])


def list_ngram_range(token_list, *args, **kwargs):
    """Return a list of n-tuples, one for each possible sequence of n items in the token_list

    Arguments:
      join (bool or str): if str, then join ngrom tuples on it before returning
         True is equivalent to join=' '
         default = True

    >>> list_ngram_range('goodbye cruel world'.split(), 0, 2, join=False)
    [('goodbye',), ('cruel',), ('world',), ('goodbye', 'cruel'), ('cruel', 'world')]
    >>> list_ngram_range('goodbye cruel world'.split(), 2, join=False)
    [('goodbye',), ('cruel',), ('world',), ('goodbye', 'cruel'), ('cruel', 'world')]
    >>> list_ngram_range('goodbye cruel world'.split(), 0, 2, join='|')
    ['goodbye', 'cruel', 'world', 'goodbye|cruel', 'cruel|world']
    >>> list_ngram_range('goodbye cruel world'.split(), 0, 2, join=True)
    ['goodbye', 'cruel', 'world', 'goodbye cruel', 'cruel world']
    """
    m, n = (args if len(args) > 1 else ((0, args[0]) if args else (0, 1)))
    join = args[2] if len(args) > 2 else kwargs.pop('join', True)
    return list(chain(*(list_ngrams(token_list, i + 1, join=join) for i in range(0, n))))


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


def str_strip(s, strip_chars=string.punctuation + ' \t\n\r'):
    return s.strip(strip_chars)


def str_lower(s):
    return s.lower()


def to_ascii(s, filler='-'):
    if not s:
        return ''
    if not isinstance(s, basestring):  # e.g. np.nan
        return to_ascii(repr(s))
    try:
        return s.encode('utf8')
    except:
        return ''.join(c if c < chr(128) else filler for c in s if c)


def passthrough(s):
    return s


class Tokenizer(object):
    """Callable and iterable class that yields substrings split on spaces or other configurable delimitters.

    For both __init__ and __call__, doc is the first arg.
    TODO: All args and functionality of __init__() and __call__() should be the same.

    FIXME: Implement the `nltk.tokenize.TokenizerI` interface
           Is it at all pythonic to make a class callable and iterable?
           Is it pythonic to have to instantiate a TokenizerI instance and then call that instance's `tokenize` method?

    >>> abc = (chr(ord('a') + (i % 26)) for i in xrange(1000))
    >>> tokenize = Tokenizer(ngrams=5)
    >>> ans = list(tokenize(' '.join(abc)))
    >>> ans[:7]
    ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    >>> ans[1000:1005]
    ['a b', 'b c', 'c d', 'd e', 'e f']
    >>> ans[1999:2004]
    ['a b c', 'b c d', 'c d e', 'd e f', 'e f g']
    >>> tokenize = Tokenizer(stem='Porter')
    >>> doc = "Here're some stemmable words provided to you for your stemming pleasure."
    >>> sorted(set(tokenize(doc)) - set(Tokenizer(doc, stem='Lancaster')))
    [u"Here'r", u'pleasur', u'some', u'stemmabl', u'your']
    >>> sorted(set(Tokenizer(doc, stem='WordNet')) - set(Tokenizer(doc, stem='Lancaster')))
    ["Here're", 'pleasure', 'provided', 'some', 'stemmable', 'stemming', 'your']
    """
    def __init__(self, doc=None, regex=CRE_TOKEN, strip=True, nonwords=False, nonwords_set=None, nonwords_regex=RE_NONWORD,
                 lower=None, stem=None, ngrams=1):
        # specific set of characters to strip
        self.strip_chars = None
        if isinstance(strip, basestring):
            self.strip_chars = strip
            # strip_chars takes care of the stripping config, so no need for strip function anymore
            self.strip = None
        elif strip is True:
            self.strip_chars = '-_*`()"' + '"'
        strip = strip or None
        # strip whitespace, overrides strip() method
        self.strip = strip if callable(strip) else (str_strip if strip else None)
        self.doc = to_ascii(doc)
        self.regex = regex
        if isinstance(self.regex, basestring):
            self.regex = re.compile(self.regex)
        self.nonwords = nonwords  # whether to use the default REGEX for nonwords
        self.nonwords_set = nonwords_set or set()
        self.nonwords_regex = nonwords_regex
        self.lower = lower if callable(lower) else (str_lower if lower else None)
        self.stemmer_name, self.stem = 'passthrough', passthrough  # stem can be a callable Stemmer instance or just a function
        self.ngrams = ngrams or 1  # ngram degree, numger of ngrams per token
        if isinstance(self.nonwords_regex, basestring):
            self.nonwords_regex = re.compile(self.nonwords_regex)
        elif self.nonwords:
            try:
                self.nonwords_set = set(self.nonwords)
            except TypeError:
                self.nonwords_set = set(['None', 'none', 'and', 'but'])
                # if a set of nonwords has been provided dont use the internal nonwords REGEX?
                self.nonwords = not bool(self.nonwords)

    def __call__(self, doc):
        """Lazily tokenize a new document (tokens aren't generated until the class instance is iterated)

        >>> list(Tokenizer()('new string to parse'))
        ['new', 'string', 'to', 'parse']
        """
        # tokenization doesn't happen until you try to iterate through the Tokenizer instance or class
        self.doc = to_ascii(doc)
        # need to return self so that this will work: Tokenizer()('doc (str) to parse even though default doc is None')
        return self
    # to conform to this part of the nltk.tokenize.TokenizerI interface
    tokenize = __call__

    def __reduce__(self):
        """Unpickling constructor and args so that pickling can be done efficiently without any bound methods, etc"""
        return (Tokenizer, (None, self.regex, self.strip, self.nonwords, self.nonwords_set, self.nonwords_regex,
                self.lower, self.stemmer_name, self.ngrams))

    def span_tokenize(self, s):
        """Identify the tokens using integer offsets `(start_i, end_i)` rather than copying them to a new sequence

        The sequence of tokens (strings) can be generated with

            `s[start_i:end_i] for start_i, end_i in span_tokenize(s)`

        Returns:
          generator of 2-tuples of ints, like ((int, int) for token in s)
        """
        return
        # raise NotImplementedError("span_tokenizer interface not yet implemented, so just suck it up and use RAM to tokenize() ;)")

    def tokenize_sents(self, strings):
        """NTLK.
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:
            return [self.tokenize(s) for s in strings]
        :rtype: list(list(str))
        """
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(self, strings):
        """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:
            return iter((self.span_tokenize(s) for s in strings))
        :rtype: iter(list(tuple(int, int)))
        """
        for s in strings:
            yield list(self.span_tokenize(s))

    def __iter__(self, ngrams=None):
        r"""Generate a sequence of words or tokens, using a re.match iteratively through the str

        TODO:
          - need two different self.lower and lemmatize transforms, 1 before and 1 after nonword detection
          - each of 3 nonword filters on a separate line, setting w=None when nonword "hits"
          - refactor `nonwords` arg/attr to `ignore_stopwords` to be more explicit

        >>> doc = "John D. Rock\n\nObjective: \n\tSeeking a position as Software --Architect-- / _Project Lead_ that can utilize my expertise and"
        >>> doc += " experiences in business application development and proven records in delivering 90's software. "
        >>> doc += "\n\nSummary: \n\tSoftware Architect"
        >>> doc += " who has gone through several full product-delivery life cycles from requirements gathering to deployment / production, and"
        >>> doc += " skilled in all areas of software development from client-side JavaScript to database modeling. With strong experiences in:"
        >>> doc += " \n\tRequirements gathering and analysis."

        The python splitter will produce 2 tokens that are only punctuation ("/")
        >>> len([s for s in doc.split() if s])
        72

        The built-in nonword REGEX ignores all-punctuation words, so there are 2 less here:
        >>> len(list(Tokenizer(doc, strip=False, nonwords=False)))
        70

        In addition, punctuation at the end of tokens is stripped so "D. Rock" doesn't tokenize to "D." but rather "D"
        >>> run_together_tokens = ''.join(list(Tokenizer(doc, strip=False, nonwords=False)))
        >>> '/' in run_together_tokens or ':' in ''.join(run_together_tokens)
        False

        But you can turn off stripping when instantiating the object.
        >>> all(t in Tokenizer(doc, strip=False, nonwords=True) for t in ('D', '_Project', 'Lead_', "90's", "product-delivery"))
        True
        """
        ngrams = ngrams or self.ngrams
        # FIXME: Improve memory efficiency by making this ngram tokenizer an actual generator
        if ngrams > 1:
            original_tokens = list(self.__iter__(ngrams=1))
            for tok in original_tokens:
                yield tok
            for i in range(2, ngrams + 1):
                for tok in list_ngrams(original_tokens, n=i, join=' '):
                    yield tok
        else:
            for w in self.regex.finditer(self.doc):
                if w:
                    w = w.group()
                    w = w if not self.strip_chars else str_strip(w, self.strip_chars)
                    w = w if not self.strip else self.strip(w)
                    w = w if not self.stem else self.stem(w)
                    w = w if not self.lemmatize else self.lemmatize(w)
                    w = w if not self.lower else self.lower(w)
                    # FIXME: nonword check before and after preprossing? (lower, lemmatize, strip, stem)
                    # 1. check if the default nonwords REGEX filter is requested, if so, use it.
                    # 2. check if a customized nonwords REGES filter is provided, if so, use it.
                    # 3. make sure the word isn't in the provided (or empty) set of nonwords
                    if w and (not self.nonwords or not re.match(r'^' + RE_NONWORD + '$', w)) and (
                            not self.nonwords_regex or not self.nonwords_regex.match(w)) and (
                            w not in self.nonwords_set):
                        yield w

    # can these all just be left to default assignments in __init__ or as class methods assigned to global `passthrough()`
    def strip(self, s):
        """Strip punctuation surrounding a token"""
        return s

    def stem(self, s):
        """Find the lexial root of a word, e.g. convert 'running' to 'run'"""
        return s

    def lemmatize(self, s):
        """Find the semantic root of a word, e.g. convert 'was' to 'be'"""
        return s

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class PassageIter(object):

    """Passage (document, sentence, line, phrase) generator for files at indicated path

    Walks all the text files it finds in the indicated path,
    segmenting sentences and yielding them one at a time

    References:
      Radim's [word2vec tutorial](http://radimrehurek.com/2014/02/word2vec-tutorial/)
    """

    def __init__(self, path='', ext='', level=None, dirs=False, files=True,
                 sentence_segmenter=generate_sentences, word_segmenter=string.split, verbosity=0):
        self.file_generator = generate_files(path=path, ext='', level=None, dirs=False, files=True, verbosity=0)

    def __iter__(self):
        for fname in os.listdir(self.file_generator):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
