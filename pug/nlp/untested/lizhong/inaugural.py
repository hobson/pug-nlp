# -*- coding: utf-8 -*-
""" By LiZhong Zheng, et. al"""
from __future__ import division

import os
from math import log, e
from collections import Mapping, Counter
import json

import nltk
import numpy as np

import character_subset as ascii
from util import is_ignorable_str, get_words


# trial for sharing code

IGNORE_FILES = ('readme',)

# john is editing from his github account

DOCUMENT_FOLDER_SUFFIX = 'pug/nlp/data'
for depth in range(7):
    if depth:
        DATA_FOLDER = os.path.join(os.path.sep.join(['..'] * depth), DOCUMENT_FOLDER_SUFFIX)
    else:
        DATA_FOLDER = DOCUMENT_FOLDER_SUFFIX
    DOCUMENT_FOLDER = os.path.join(DATA_FOLDER, 'inaugural_speeches')
    # print DOCUMENT_FOLDER
    try:
        DEFAULT_FILE_LIST = os.listdir(DOCUMENT_FOLDER)
        break
    except:
        DEFAULT_FILE_LIST = []
# print DATA_FOLDER


def get_adjacency_matrix(filenames=DEFAULT_FILE_LIST, entropy_threshold=0.90, folder=DOCUMENT_FOLDER, normalized=False, verbosity=1):
    """Calculate relevance score for words in a set of files

    score_words_in_files(filenames=DEFAULT_FILE_LIST, entropy_threshold=0.9):
        return scores, occurrence_matrix, filenames, most_relevant_words

    by LiZhong Zeng
    """

    stem = nltk.stem.PorterStemmer().stem
    filtered_filenames = []

    Text = []     # list of segmented texts
    Total = []    # all texts put together 
    Speech_Length = []
    
    if verbosity > 0:
        print 'Reading %s files' % len(filenames)
    i = 0
    for fn in filenames:
        if is_ignorable_str(fn, ignorable_strings=IGNORE_FILES, lower=True, filename=True, startswith=True):
            continue
        filtered_filenames += [fn]
        i += 1
        f = open(os.path.join(folder, fn))
        if verbosity > 1:
            print 'Reading ' + f.name
        raw = f.read()
        tokens = [stem(w) for w in nltk.word_tokenize(raw)]
        # delete short words and make everything lowercase
        tokens=[w.lower() for w in tokens if len(w) > 2]
        Speech_Length += [len(tokens)]
        Text.append(tokens)        
        Total=Total+tokens
    if verbosity > 0:
        print '%s files were indexed (%s were ignored)' % (i, len(filenames)-i)
            
    Empirical_Total=nltk.FreqDist(Total)    
    Vocabulary=Empirical_Total.keys()   # the entire set of words
    Size=len(Vocabulary)
    #numDoc=len(Text)

    Dist=range(Size)
    Vectors=[]          # Record a list of empirical distributions
    for i in range(len(filtered_filenames)):
        fdist=nltk.FreqDist(Text[i])

        for j in range(Size):
            Dist[j]=fdist[Vocabulary[j]]
            
        Vectors.append(Dist[:]) # Dist[:] makes a copy of the list to append
    
    Word_Relevance=range(Size) # store a relevance score for each word
    for wordIndex in range(Size):
        Word_Dist= nltk.FreqDist([Vectors[i][wordIndex] for i in range(len(filtered_filenames))])

        Word_Relevance[wordIndex] = 0

        # # check if the number of files that do not have the word is close to half
        # if (abs(Word_Dist[0] - len(filenames)/2) <= 3):
        #    Word_Relevance[wordIndex]=1            

        # information entropy normalized by the support (number of events)
        H_normalized = renyi_entropy(Word_Dist, alpha=.6, base=e, normalized=True)

        if (H_normalized > entropy_threshold):
            Word_Relevance[wordIndex] = 1

    Key_words = [Vocabulary[i] for i in range(Size) if Word_Relevance[i]] 

    if verbosity > 0:
        print 'Computed a relevance score for %s words and reduced it to %s words above %s%% relevance.' % (Size, len(Key_words), entropy_threshold * 100.)

    Reduced_Vectors=[]    
    for i in range(len(filtered_filenames)):
        total_count = sum(Vectors[i])
        if normalized: 
            Reduced_Vectors.append([Vectors[i][j] / total_count for j in range(Size) if Word_Relevance[j]])
        else:
            Reduced_Vectors.append([Vectors[i][j] for j in range(Size) if Word_Relevance[j]])
    return Reduced_Vectors, filtered_filenames, Key_words

def svd_scores(vectors, labels=None, verbosity=1):
    labels = labels or [str(i) for i in range(len(vectors))]

    if len(labels) != len(vectors):
        raise RuntimeWarning("The number of labels provided doesn't matc the number of vectors.")

    U, s, V = np.linalg.svd(vectors)

    scores = []

    if verbosity > 0:
        print 'SCORES', ':', 'LABELS'    
    for i, vector in enumerate(vectors):
        scores += [np.inner(V[0], vector) / sum(vector)]
        if verbosity > 0:
            print scores[-1], ':', labels[-1]
    return scores


def get_occurrence_matrix(strings, row_labels=None, tokenizer=get_words, stemmer=None):
    if stemmer is None:
        stemmer = get_occurrence_matrix.stemmer
    stemmer = stemmer or unicode
    print stemmer
    O_sparse = [Counter(stemmer(w) for w in tokenizer(s)) for s in strings]
    return matrix_from_counters(O_sparse, row_labels=row_labels)
get_occurrence_matrix.stemmer = nltk.stem.porter.PorterStemmer().stem


def sum_counters(counters):
    total = Counter()
    for counter in counters:
        total += counter
    return total


def matrix_from_counters(counters, row_labels=None):
    if not row_labels and isinstance(counters, Mapping) and all(isinstance(counter, Counter) for row_label, counter in counters.iteritems()):
        # TODO: def list_pair_from_dict(dict):
        row_labels = []
        new_counters = []
        for row_label, counter in counters.iteritems():
            row_labels += [row_label]
            new_counters += [counter]
        counters = new_counters
    if not row_labels or len(row_labels) != len(counters):
        row_labels = [i for i, c in enumerate(counters)]


    total = sum_counters(counters)

    col_labels, O = list(total), []
    for counts in counters:
        O += [[0] * len(col_labels)]
        for label, count in counts.iteritems():
            j = col_labels.index(label)
            O[-1][j] += count
    return O, row_labels, col_labels

def score_words_in_files(filenames=DEFAULT_FILE_LIST, entropy_threshold=0.90, folder=DOCUMENT_FOLDER, verbosity=1):
    adjacency_matrix, files, words = get_adjacency_matrix(filenames=DEFAULT_FILE_LIST, entropy_threshold=0.90, folder=DOCUMENT_FOLDER)
    return svd_scores(adjacency_matrix, files, verbosity), adjacency_matrix, files, words 


def reverse_dict(d):
    return dict((v, k) for (k, v) in dict(d).iteritems())


def reverse_dict_of_lists(d):
    ans = {}
    for (k, v) in dict(d).iteritems():
        for new_k in list(v):
            ans[new_k] = k
    return ans


def shannon_entropy(discrete_distribution, base=e, normalized=True):
    """Shannon entropy (information uncertainty) for the logarithm base (units of measure) indicated

    The default logarithm base is 2, which provides entropy in units of bits. Shannon used
    bases of 2, 10, and e (natural log), but scipy and most math packages assume e.

    >>> from scipy.stats import entropy as scipy_entropy
    >>> scipy_entropy([.5, .5]) / log(2) # doctest: +ELLIPSIS
    1.0
    >>> shannon_entropy([.5, .5], base=2)
    0.69314718...
    >>> scipy_entropy([.5, .5])  # doctest: +ELLIPSIS
    0.69314718...
    1.0
    >>> shannon_entropy([.5, .5], base=2, normalized=False)  # doctest: +ELLIPSIS
    0.69314718...
    >>> scipy_entropy([c / 13. for c in [1, 2, 3, 4, 3]])  # doctest: +ELLIPSIS
    1.5247073930...
    >>> shannon_entropy([1, 2, 3, 4, 3], normalized=False)
    1.5247073930...
    """
    if isinstance(discrete_distribution, Mapping):
        discrete_distribution = discrete_distribution.values()
    if base == None:
        base = e

    if not len(discrete_distribution):
        raise RuntimeWarning("Empty discrete_distribution probability distribution, so Renyi entropy (information) is zero.")
        return float('-0.')
    if not any(discrete_distribution):
        raise RuntimeWarning("Invalid value encountered in divison, 0/0 (zero divided by zero). Sum of discrete_distribution (discrete distribution integral) is zero.")
        return float('nan')
    if any(count < 0 for count in discrete_distribution):
        raise RuntimeWarning("Some counts or frequencies (probabilities) in discrete_distribution were negative.")

    total_count = float(sum(discrete_distribution))

    if base == e:
        H = sum(count * log(count / total_count) for count in discrete_distribution) / total_count
    else:
        H = sum(count * log(count / total_count, base) for count in discrete_distribution) / total_count
    if normalized:
        return -1 * H / log(len(discrete_distribution), base)
    return -1 * H


def renyi_entropy(word_counts, alpha=1, base=2, normalized=True):
    """Renyi entropy of order alpha and the logarithm base (units of measure) indicated

    The default logarithm base is 2, which provides entropy in units of bits, Renyi's 
    preferred units. Shannon used bases of 2, 10, and e (natural log).

    >>> from scipy.stats import entropy as scipy_entropy
    >>> scipy_entropy([.4, .6])  # doctest: +ELLIPSIS
    0.97...
    >>> renyi_entropy([.4, .6])  # doctest: +ELLIPSIS
    0.97...
    >>> renyi_entropy([.4, .6], .6)
    0.97...
    >>> renyi_entropy([], .6)
    -0.0
    >>> renyi_entropy([0.] * 10, .6)
    nan
    """
    if isinstance(word_counts, Mapping):
        word_counts = word_counts.values()
    if base == None:
        base = e

    N = len(word_counts)
    if not N:
        raise RuntimeWarning("Empty word_counts probability distribution, so Renyi entropy (information) is zero.")
        return float('-0.')
    if not any(word_counts):
        raise RuntimeWarning("Invalid value encountered in divison, 0/0 (zero divided by zero). Sum of word_counts (discrete distribution integral) is zero.")
        return float('nan')

    if alpha == 1:
        return shannon_entropy(word_counts, base=base, normalized=normalized)

    total_count = float(sum(word_counts))    
    entropy_unif = log(N, base)  # log of the support, don't have to turn to float
    sum_pow = sum(pow(count / total_count, alpha) for count in word_counts)
    
    # log(x) is 75% faster as log(x, math.e), according to timeit on python 2.7.5
    # so split into 2 cases, base e (None), and any other base
    if base == e:
        log_sum_pow = log(sum_pow)
    else:
        log_sum_pow = log(sum_pow, base)

    H = log_sum_pow / (1 - alpha)

    if normalized:
        return H / (entropy_unif or 1.)
    else:
        return H


def zheng_normalized_entropy(Word_Dist, alpha=.6):
    """Renyi entropy of order alpha and the logarithm base (units of measure) indicated

    The default logarithm base is 2, which provides entropy in units of bits, Renyi's 
    preferred units. Shannon used bases of 2, 10, and e (natural log).

    >>> from scipy.stats import entropy as scipy_entropy
    >>> scipy_entropy([.4, .6])  # doctest: +ELLIPSIS
    0.97...
    >>> zheng_normalized_entropy([.4, .6], alpha=1)  # doctest: +ELLIPSIS
    0.97...
    >>> word_freq = [c / 13. for c in [1, 2, 3, 4, 3]]
    >>> zheng_normalized_entropy(dict((k, v) for (k, v) in enumerate(word_freq)))  # doctest: +ELLIPSIS
    0.966162932302...
    """
    keys=Word_Dist.keys()
    Entropy_Unif= log(len(keys))  # log of the support, don't have to turn to float
    totalcount= float(sum([Word_Dist[w] for w in keys]))
    
    Entropy= -sum([ Word_Dist[w]/totalcount * log(Word_Dist[w]/totalcount) for w in keys])
    
    # Renyi entropy of order alpha
    #Entropy = log(sum ([pow(float(Word_Dist[w])/totalcount, alpha) for w in keys]))/(1-alpha)

    return float(Entropy) / Entropy_Unif


def co_adjacency(adjacency_matrix, row_names, col_names=None, bypass_col_names=True, normalized=True, verbosity=1):
    """Reduce a heterogenous adjacency matrix into a homogonous co-adjacency matrix

    coadjacency_matrix, names = co_adjacency(adjacency_matrix, row_names, col_names, bypass_col_names=True)
    """
    bypass_indx = int(not (int(bypass_col_names) % 2))
    names = (row_names, col_names or row_names)[bypass_indx]

    A = np.matrix(adjacency_matrix)
    if normalized:
        size = (float(len(adjacency_matrix[0])), float(len(adjacency_matrix)))
        A = A / size[int(bypass_indx)]
    if not bypass_indx:
        return (A * A.transpose()).tolist(), names
    return (A.transpose() * A).tolist(), names


def len_str_to_group(s, bin_min=None, bin_width=None):
    if bin_min is None:
        bin_min = len_str_to_group.min
    bin_width = bin_width or len_str_to_group.width or 1
    return int(round((len(s) - bin_min + 1) / float(bin_width)))
len_str_to_group.min = 3
len_str_to_group.width = 3


def yr_str_to_group(s, bin_min=None, bin_width=None, digits=None):
    if bin_min is None:
        bin_min = yr_str_to_group.min
    if digits is None:
        digits = yr_str_to_group.digits
    bin_width = bin_width or yr_str_to_group.width or 1
    return int(round((float(s[:digits]) - bin_min) / float(bin_width)))
yr_str_to_group.min = 1770
yr_str_to_group.width = 3
yr_str_to_group.digits = 4


def scale_exp(value, in_min, in_max, out_min=1, out_max=10, exp=1):
    """Rescale a scalar value to fit within out_min and out_max

    >>> scale_exp(150, in_min=100, in_max=200, out_min=0, out_max=1)
    .5
    >>> scale_exp(150, in_min=100, in_max=200, out_min=0, out_max=1, exp=2)
    .25
    """
    return np.power((out_max - out_min) * (value - in_min) / float(in_max - in_min), exp)


def matrix_scale_exp(A, out_min=0, out_max=1, exp=1):
    """Apply scape_exp to all elements of a matrix, auto-calculating in_min and in_max
    """
    A = np.matrix(A)
    return scale_exp(A, np.min(A), np.max(A), out_min, out_max, exp=exp).tolist()


def strip_path_ext_characters(s, strip_characters=ascii.ascii_nonword):
    return str(s).split('/')[-1].split('.')[0].strip(strip_characters)


def d3_graph(adjacency_matrix, row_names, col_names=None, str_to_group=len_str_to_group, str_to_name=strip_path_ext_characters, str_to_value=float, num_groups=7., directional=True):
    """Convert an adjacency matrix to a dict of nodes and links for d3 graph rendering

    row_names = [("name1", group_num), ("name2", group_num), ...]
    col_names = [("name1", group_num), ("name2", group_num), ...]

    Usually row_names and col_names are the same, but not necessarily.
    Group numbers should be an integer between 1 and the number of groupings
    of the nodes that you want to display.

    adjacency_matrix = [
        [edge00_value, edge01_value, edge02_value...],
        [edge10_value, edge11_value, edge12_value...],
        [edge20_value, edge21_value, edge22_value...],
        ...
        ]

    The output is a dictionary of lists of vertexes (called "nodes" in d3)
    and edges (called "links" in d3):

    {
        "nodes": [{"group": 1, "name": "Alpha"}, 
                  {"group": 1, "name": "Beta"}, 
                  {"group": 2, "name": "Gamma"}, ...
                 ],
        "links": [{"source": 1, "target": 0, "value": 1}, 
                  {"source": 2, "target": 0, "value": 8}, 
                  {"source": 3, "target": 0, "value": 10}, 
                 ]
    }
    """
    if col_names is None:
        col_names = row_names

    nodes, links = [], []

    print '-' * 10
    # get the nodes list first, from the row and column labels, even if not square
    for names in (row_names, col_names):
        for i, name_group in enumerate(names):
            if isinstance(name_group, basestring):
                name_group = (str_to_name(name_group), str_to_group(name_group))
            node = {"name": name_group[0], "group": name_group[1] or 1}
            print node
            if node not in nodes:
                nodes += [node]                

    for i, row in enumerate(adjacency_matrix):
        for j, value in enumerate(row):
            links += [{"source": i, "target": j, "value": str_to_value(value)}]
            if directional:
                links += [{"source": j, "target": i, "value": str_to_value(value)}]

    return {'nodes': nodes, 'links': links}


def d3_graph_occurrence(adjacency_matrix, row_names, col_names=None, str_to_group=len_str_to_group, str_to_value=float, verbosity=1, directional=True):
    """Convert an adjacency matrix to a dict of nodes and links for d3 graph rendering

    row_names = [("name1", group_num), ("name2", group_num), ...]
    col_names = [("name1", group_num), ("name2", group_num), ...]

    Usually row_names and col_names are the same, but not necessarily.
    Group numbers should be an integer between 1 and the number of groupings
    of the nodes that you want to display.

    adjacency_matrix = [
        [edge00_value, edge01_value, edge02_value...],
        [edge10_value, edge11_value, edge12_value...],
        [edge20_value, edge21_value, edge22_value...],
        ...
        ]

    The output is a dictionary of lists of vertexes (called "nodes" in d3)
    and edges (called "links" in d3):

    {
        "nodes": [{"group": 1, "name": "Alpha"}, 
                  {"group": 1, "name": "Beta"}, 
                  {"group": 2, "name": "Gamma"}, ...
                 ],
        "links": [{"source": 1, "target": 0, "value": 1}, 
                  {"source": 2, "target": 0, "value": 8}, 
                  {"source": 3, "target": 0, "value": 10}, 
                 ]
    }
    """
    if col_names is None:
        col_names = row_names

    nodes, links = [], []
    N, M = len(row_names), len(col_names)
    j = 0

    if verbosity > 1:
        print '-' * 10
    # get the nodes list first, from the row and column labels, even if not square
    for group_num, names in enumerate([row_names, col_names]):
        for i, name in enumerate(names):
            node = {"name": name, "group": group_num + 1}
            if verbosity > 1:
                print j, node
            nodes += [node]
            j += 1


    # get the edges next
    for i, row in enumerate(adjacency_matrix):
        for j, value in enumerate(row):
            links += [{"source": i, "target": N + j, "value": str_to_value(value)}]
            if directional:
                links += [{"source": j, "target": i, "value": str_to_value(value)}]

    return {'nodes': nodes, 'links': links}

def write_file_twice(js, fn, other_dir='../miner/static'):
    with open(fn, 'w') as f:
        json.dump(js, f, indent=2)
    other_path = os.path.join(other_dir, fn)
    if os.path.isdir(other_dir):
        with open(other_path, 'w') as f:
            json.dump(js, f, indent=2)

def president_party(name):
    surname = strip_path_ext_characters(name.split(' ')[-1], strip_characters=ascii.ascii_nonletter)
    return president_party.surname_party.get(surname, '')
president_party.party = reverse_dict_of_lists(json.load(open(os.path.join(DATA_FOLDER, 'president_political_parties.json'), 'r')))
president_party.surname_party = dict((name.split(' ')[-1], party) for (name, party) in president_party.party.iteritems())


def group_to_party(group):
    return group_to_party.parties[group or 0] or ""
group_to_party.parties = ["", "Whig", "Democratic", "Republican", "Democratic-Republican", "Federalist"]

def party_to_group(s):
    """
    >>> group_to_party(party_to_group("Republican"))
    'Republican'
    >>> group_to_party(party_to_group("Democratic"))
    'Democratic'
    group_to_party(party_to_group(president_party("whatever/2001-Bush")))
    'Republican'
    group_to_party(party_to_group(president_party("whatever/1980-Reagan")))
    'Republican'
    group_to_party(party_to_group(president_party("whatever/2012-Obama")))
    'Democratic'
    group_to_party(party_to_group(president_party("whatever/1776-Washington")) or 0)
    ''
    group_to_party(party_to_group(president_party("whatever/12345-Jefferson")) or 0)
    'Whig'
    """
    return party_to_group.parties.get(s)
party_to_group.parties = dict((p, i) for (i, p) in enumerate(group_to_party.parties))


def generate_word_cooccurrence(adjacency_matrix, files, words, num_groups=7., path=os.path.join('..','miner','static','word_cooccurrence.json'), normalized=False):
    print len(adjacency_matrix), len(adjacency_matrix[0])
    print sum(adjacency_matrix[0]), sum(adjacency_matrix[1])
    O, names = co_adjacency(adjacency_matrix, row_names=files, col_names=words, bypass_col_names=False, normalized=normalized)
    print len(O), len(O[0])
    print sum(O[0]), sum(O[1])
    #O = matrix_scale_exp(O, out_min=1 ** 2, out_max=31 ** 2, exp=.5)
    len_str_to_group.min = min(len(s) for s in names)
    len_str_to_group.width = float(max(len(s) for s in names) - len_str_to_group.min) / num_groups
    graph = d3_graph(O, names, str_to_group=len_str_to_group)

    print len(graph['nodes']), len(graph['links'])/float(len(graph['nodes']))
    print sum(e['value'] for e in graph['links'] if e['source'] == 1), sum(e['value'] for e in graph['links'] if e['source'] == 2)
    print '-'*50
    json.dump(graph, open(path, 'w'), indent=2)
    return graph


def generate_document_cooccurrence(adjacency_matrix, files, words, num_groups=7., path=os.path.join('..','miner','static','doc_cooccurrence.json'), normalized=False):
    O, names = co_adjacency(adjacency_matrix, row_names=files, col_names=words, bypass_col_names=True, normalized=normalized)
    #O = matrix_scale_exp(O, out_min=1 ** .66, out_max=31 ** .66, exp=1.5)
    graph = d3_graph(O, [(strip_path_ext_characters(n), party_to_group(president_party(n)) or 1) for n in names])
    json.dump(graph, open(path, 'w'), indent=2)
    return graph


def generate_occurrence(adjacency_matrix, files, words, path=os.path.join('..','miner','static','occurrence.json')):
    #O = matrix_scale_exp(O, out_min=1 ** .66, out_max=31 ** .66, exp=1.5)
    graph = d3_graph_occurrence(adjacency_matrix, [strip_path_ext_characters(n) for n in files], [strip_path_ext_characters(n) for n in words])
    json.dump(graph, open(path, 'w'), indent=2)
    return graph


def normalize_adjacency_matrix(adjacency_matrix, rowwise=True):
    am_normalized = []
    if rowwise:
        for row in adjacency_matrix:
            row_normalized, norm = [], sum(row)
            for value in row:
                row_normalized += [value / norm]
            am_normalized += [row_normalized]
    return am_normalized


def generate_json():
    adjacency_matrix, files, words = get_adjacency_matrix(sorted(DEFAULT_FILE_LIST), entropy_threshold=0.90, normalized=False, verbosity=2)
    print len(adjacency_matrix), len(adjacency_matrix[0])
    print sum(adjacency_matrix[0]), sum(adjacency_matrix[1])
    normalized_adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)
    print len(normalized_adjacency_matrix), len(normalized_adjacency_matrix[0])
    print sum(normalized_adjacency_matrix[0]), sum(normalized_adjacency_matrix[1])
    generate_document_cooccurrence(normalized_adjacency_matrix, files, words, num_groups=5, normalized=False)
    generate_word_cooccurrence(normalized_adjacency_matrix, files, words, num_groups=5, normalized=False)
    generate_occurrence(normalized_adjacency_matrix, files, words)


def generate_matrix(normalized=True, matrix_only=False, format='json', path=os.path.join('..','miner','static','adjacency_matrix.json')):
    adjacency_matrix, files, words = get_adjacency_matrix(sorted(DEFAULT_FILE_LIST), entropy_threshold=0.92, normalized=False, verbosity=2)
    if normalized:
        adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)
    format = path.split('.')[-1].strip()
    if format == 'json':
        if matrix_only:
            js = adjacency_matrix
        else:
            js = {
                'column_labels': words,
                'row_labels': files,
                'matrix': adjacency_matrix,
                }
    else:
        raise NotImplementedError("I only know how to write json files.")
    json.dump(js, open(path, 'w'))
    return adjacency_matrix, files, words

if __name__ == '__main__':
    """
    # This will produce a 14 x 14 matrix, 
    >>> scores, adjacency_matrix, files, words = score_words_in_files(DEFAULT_FILE_LIST[:14], entropy_threshold=0.99)
    >>> coadjacency, names = co_adjacency(adjacency_matrix, row_names=files, col_names=words, bypass_col_names=True)
    >>> len(coadjacence), len(coadjacency[0], len(names)
    (14, 14, 14)
    """
    #adjacency_matrix, files, words = get_adjacency_matrix(sorted(DEFAULT_FILE_LIST), entropy_threshold=0.92, normalized=False, verbosity=2)
    #scores = svd_scores(adjacency_matrix, labels=files, verbosity=2)
    #graph = d3_graph_occurrence(adjacency_matrix, files, words)
    # print json.dumps(graph, indent=4)
