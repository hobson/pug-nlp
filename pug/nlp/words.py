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
  '.Net': "Dot-Net",
  'IMHO': "in my humble opinion",
  'IMO': "in my opinion",
  'BRB': 'be right back',
  'OOO': 'out of the office',
  'OTL': 'out to lunch',
  'WTF': 'what the freak',
  'FUBAR': 'freaked out beyond recognition',
}


ignorable_suffixes = set(
  "'s",
  )


def synonyms(word):
    return thesaurus.get(word.lower().strip(), [])
