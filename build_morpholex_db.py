# !/usr/bin/env python
#  encoding: utf-8

"""
This script takes the modified* English Lexicon Project file as input and outputs a new
CSV file which is the MorphoLex database.

* Modifications are the following:
- Some blatantly erroneous segmentations were fixed
- A new segmentation column was added, that better fits our projects' aims
(These changes are described in more detail in the following article:
[INSERT ARTICLE])

The output database must contain the following data for each entry:
- ELP ItemID
- Word
- POS
- Word length
- Number of morphemes
- PRS signature
- MorphoLexSegm
- Morphological variables for each morpheme:
  * Frequency
  * Family size
  * %MoreFreqInFam
  * P
  * P*
"""

from collections import OrderedDict
from pprint import pprint
import copy
import os
import re
import csv
import nltk

PROJECT_PATH = '/home/hugo/Projects/ELP_morpho_vars/'
# SEGM_DB_PATH = os.path.join(PROJECT_PATH, 'ELP_segmentations_no_flex.csv')
DB_PATH = os.path.join(PROJECT_PATH, '/home/hugo/Projects/ELP_morpho_vars/input/ELP-2016-11-26.csv')
VARS_SAVE_PATH = os.path.join(PROJECT_PATH, 'output/ELP_morphological_variables.csv')
HAPAX_SBTL_FREQ_THRESHOLD = 0.02
HAPAX_HAL_FREQ_THRESHOLD = 1
DB_ITEMID_COL = 0
DB_WORD_COL = 1
DB_POS_COL = 2
DB_HAL_FREQ_COL = 25
DB_SBTL_FREQ_COL = 27
DB_SEGM_COL = 48


def apply_morpho_vars_to_lex_db(db, morpho_vars):
    """
    Build a dict {PRS_signature: [lexical_data]}, where lexical_data is a list
    of rows from the lexical database, each of which contain the data described
    in the top comment of this script.
    """
    res = {}
    for row in db:
        # temp = copy.deepcopy(row)
        segm = row[DB_SEGM_COL]
        prs = get_PRS_signature(row[DB_SEGM_COL])
        if prs not in res.keys():
            res[prs] = []
        morphemes = get_morphemes(segm)
        wordlen = len(row[DB_WORD_COL])
        n_morphemes = len(morphemes)
        prs_string = ','.join([str(x) for x in prs])
        temp = [row[DB_ITEMID_COL], row[DB_WORD_COL], row[DB_POS_COL], wordlen, 
                n_morphemes, prs_string, segm]
        for m in morphemes:
            freq = morpho_vars[m]['hal_freq']
            rel_fam_freq = get_relative_fam_freq(m, db, segm, freq,
                                                 morpho_vars[m]['family'])
            m_vars = [morpho_vars[m]['family_size'],
                      morpho_vars[m]['hal_freq'],
                      rel_fam_freq,
                      morpho_vars[m]['hal_p'],
                      morpho_vars[m]['hal_p*']]
            temp.extend(m_vars)
        res[prs].append(temp)

    return res


def compute_morphological_variables(db, hapax_set):
    """
    For each morpheme in the segmented lexical database, compute the following
    attributes:

    - family size
    - summed token frequency
    - p-measure
    - p*-measure
    - % of family more frequent (PFMF)
    """
    morpho_vars = {}
    counted = set()

    for i, row in enumerate(db):
        print('\r{} / {}'.format(i, len(db)), end='')    
        segm = row[DB_SEGM_COL]
        morphemes = get_morphemes(segm)
        for m in [x for x in morphemes if x not in counted]:
            counted.add(m)
            freq = total_morpheme_freq(m, db)
            family = get_family(m, db)
            morpho_vars[m] = {'hal_freq': freq}
            morpho_vars[m]['family'] = family
            morpho_vars[m]['family_size'] = len(family)
            
            hapax_freq = total_morpheme_freq(m, hapax_set)
            if hapax_freq == 0:
                morpho_vars[m]['hal_p'] = 0
                morpho_vars[m]['hal_p*'] = 0
            else:
                morpho_vars[m]['hal_p'] = hapax_freq / freq
                morpho_vars[m]['hal_p*'] = hapax_freq / len(hapax_set)

    print('\n')
    return morpho_vars


def generate_headers(prs):
    """
    Given a prs signature for a CSV database, returns the appropriate header
    structure (column names).
    """
    p, r, s = int(prs[0]), int(prs[1]), int(prs[2])
    headers = []
    for i in range(1, p+1):
        headers.extend(['PREF%d_FamSize' % i, 
                        'PREF%d_Freq_HAL' % i, 'PREF%d_%%FamMoreFreq_HAL' % i,
                        'PREF%d_P_HAL' % i, 'PREF%d_P*' % i
                        ])
    for i in range(1, r+1):
        headers.extend(['ROOT%d_FamSize' % i, 
                        'ROOT%d_Freq_HAL' % i, 'ROOT%d_%%FamMoreFreq_HAL' % i,
                        'ROOT%d_P_HAL' % i, 'ROOT%d_P*' % i
                        ])
    for i in range(1, s+1):
        headers.extend(['SUFF%d_FamSize' % i, 
                        'SUFF%d_Freq_HAL' % i, 'SUFF%d_%%FamMoreFreq_HAL' % i,
                        'SUFF%d_P_HAL' % i, 'SUFF%d_P*' % i
                       ])
    return headers


def get_family(morpheme, db):
    """
    Returns a dict {WORD: {'hal': HAL_FREQUENCY, 'sbtl': SBTL:FREQUENCY}} of all the words
    in database that contain the morpheme.
    
    Since we deleted non-derivational suffixes from segm, two different words
    (ie 'friend' and 'friends') might have the same segmentation. All frequencies
    from words with the same segm field are counted under the same segm key.
    This is useful in computing the size of the family; we want 'friend' and
    'friends' to count as a single family member, while 'friendship' is a true
    OTHER family member.
    """
    family = {}
    for row in db:
        segm = row[DB_SEGM_COL]
        if morpheme in segm:
            if segm not in family:
                family[segm] = {'hal_freq': 0}
            family[segm]['hal_freq'] += row[DB_HAL_FREQ_COL]

    return family


def get_hapax_set(db):
    """
    Returns the set of words in database that have either:
    a) subtitle word frequency equal or lower to HAPAX_SBTL_FREQ_THRESHOLD 
        OR
    b) HAL frequency equal or lower to HAPAX_HAL_FREQ_THRESHOLD 
    """
    hapax = []
    for row in db:
        if row[DB_HAL_FREQ_COL] <= HAPAX_HAL_FREQ_THRESHOLD:
            hapax.append(row)
        elif row[DB_SBTL_FREQ_COL] <= HAPAX_SBTL_FREQ_THRESHOLD:
            hapax.append(row)
    return hapax


def get_morphemes(segm):
    """
    Parses the ELP morpheme segmentation string and returns a list of morphemes.
    """
    if segm == 'NULL':
        raise Exception("NULL segmentation!")
    return re.findall(r'[<>(][^><)]+?[<>)]', segm)


def get_PRS_signature(segm):
    """
    Returns a signature composed of 3 integers, each digit representing
    # of prefixes, # of roots, and # of suffixes, respectively.
    """
    n_pref = int(segm.count('<') / 2)
    n_root = segm.count('(')
    n_suff = int(segm.count('>') / 2)
    return (n_pref, n_root, n_suff)


def get_relative_fam_freq(morpheme, db, segm, word_freq, family):
    """
    Given a word and one of its morphemes, return the percentage of words in db
    that contain the same morpheme and are more frequent.
    """
    if len(family) == 1:
        return 0
    family.pop(segm, None)
    more_frequent_in_fam_hal = sum([1 for x in family.values()
                                    if x['hal_freq'] > word_freq])
    return (more_frequent_in_fam_hal / len(family)) * 100


def merge_new_data_with_database(prs_data, main_db):
    """
    For each word in prs_data, put together the following data:
    - ELP ItemID
    - Word
    - POS
    - Word length
    - Number of morphemes
    - PRS signature
    - MorphoLexSegm
    - Morphological variables for each morpheme:
      * Frequency
      * Family size
      * %MoreFreqInFam
      * P
      * P*
    """
    for signature in prs_data:
        for i, row in enumerate(prs_data[signature]):
            word = row[0]
            try:
                prs_data[signature][i] = main_db[word] + prs_data[signature][i][3:]
            except KeyError:
                print(word)
                continue

    return prs_data


def preprocess_db(db):
    """
    Prepare csv data for processing by the program.
    """
    # 1. Keep only the subset of the db for which we have a MorphoLexSegm value
    #    Those items that have a zero or NULL frequency everywhere or that have a NULL
    #    POS are a subset of the items without a segmentation.
    valid_db_subset = [x for x in db if x[DB_SEGM_COL] != 'NULL']

    # 2. Ensure that relevant numeric values are not expressed as strings
    #    - Some values are NULL in the SBTLWF column
    #    - All values in Freq_HAL are assumed to be non-null and integer
    for row in valid_db_subset:
        if row[DB_SBTL_FREQ_COL] != 'NULL':
            row[DB_SBTL_FREQ_COL] = float(row[DB_SBTL_FREQ_COL])
        else:
            row[DB_SBTL_FREQ_COL] = 0
        try:
            row[DB_HAL_FREQ_COL] = int(row[DB_HAL_FREQ_COL])
        except:
            print(row)
            raise
    return valid_db_subset


def total_morpheme_freq(morpheme, db):
    """
    Sums the frequencies of words containing the morpheme in the database.
    Returns an int.
    """
    total_freq_hal = 0
    for row in db:
        if morpheme in row[DB_SEGM_COL]:
            total_freq_hal += row[DB_HAL_FREQ_COL]
    return total_freq_hal


def save_morpho_vars_to_file(morpho_vars, filepath):

    ordered_mfreqs = OrderedDict(sorted(morpho_vars.items()))

    with open(filepath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Morpheme', 'Family size', 'Frequency', 'P', 'P*'])
        for morpheme, d in ordered_mfreqs.items():
            csvwriter.writerow([morpheme,
                                d['family_size'],
                                d['freq'],
                                d['p'],
                                d['p*']])


if __name__ == '__main__':
    """
    Load lexical database, get morphological variables, divide lexical items
    by PRS signature and build one database per PRS signature where each
    lexical item is associated with its parts' morphological variables.
    """
    # Load lexical database
    with open(DB_PATH) as database:
        db = [x for x in list(csv.reader(database))[2:] if x]  # skip headers (2 rows)

    print('preprocessing db')
    db = preprocess_db(db)
    print('getting hapax set')
    hapax_set = get_hapax_set(db)
    print('computing morphological variables')
    morpho_vars = compute_morphological_variables(db, hapax_set)
    print('applying morphological variables to database')
    new_data_by_prs = apply_morpho_vars_to_lex_db(db, morpho_vars)

    # with open(DB_PATH) as database:
    #     reader = csv.reader(database)
    #     headers1 = next(reader)
    #     headers2 = next(reader)
    #     main_db = {x[1]:x for x in reader if len(x) > 1}
    #     print(len(main_db))
    # print('merging new data with existing database')
    # merged_data = merge_new_data_with_database(new_data_by_prs, main_db)

    # create output directory if it doesn't exist already
    os.makedirs('output', exist_ok=True)
    headers = ['ELP_ItemID', 'Word', 'POS', 'Length', 'Nmorph', 'PRS_signature', 
               'MorphoLexSegm']
    for prs, data in new_data_by_prs.items():
        prs_str = re.sub(r'[,()]', '', str(prs))
        savepath = os.path.join(PROJECT_PATH, 'output/%s.csv' % prs_str)
        with open(savepath, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(headers + generate_headers(prs))
            csvwriter.writerows(data)


    # save_morpho_vars_to_file(morpho_vars, VARS_SAVE_PATH)
