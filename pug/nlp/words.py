"""Lists of Words (Lexica)"""

thesaurus = {
    'date': ('datetime', 'time', 'date_time'),
    'time': ('datetime', 'date', 'date_time'),
    }


spellings = {
    # one spelling,   punctuation-free, single-word spelling
    'C++': 'C-Plus-Plus',
    'c++': 'C-Plus-Plus',
    'C#':  "C-Sharp",
    '+1': 'plus-one',
    '.Net': "Dot-Net",
    'js': "Javascript",
    'JS': "Javascript",
    'IMHO': "in my humble opinion",
    'IMO': "in my opinion",
    'BRB': 'be right back',
    'OOO': 'out of the office',
    'OTL': 'out to lunch',
    'WTF': 'what the freak',
    'FUBAR': 'freaked out beyond recognition',
    ':)': 'small-smile',
    ':-)': 'smile',
    ';)': 'wink-smile',
    ';-)': 'wink-smile',
    ':D': 'big-smile',
    ':-D': 'big-smile',
    ':P': 'tongue-out-smile',
    ':-P': 'tongue-out-smile',
    '<3': 'heart',
}


ignorable_suffixes = set(
    "'s",
    )
ignorable_suffixes = set(
    "'s",
    )


def synonyms(word):
    return thesaurus.get(word.lower().strip(), [])
