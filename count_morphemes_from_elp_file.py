# !/usr/bin/env python
#  encoding: utf-8

"""
This script takes the English Lexicon Project file as input and outputs a new
CSV file with two rows: morphemes segmented in the ELP are in the first row,
their summed corpus frequency in the second row.
"""

from __future__ import division
from collections import OrderedDict
from pprint import pprint
import copy
import os
import re
import csv
import nltk

PROJECT_PATH = '/home/hugo/Projects/elp_freq_morpho/'
SEGM_DB_PATH = os.path.join(PROJECT_PATH, 'input/ELP_segmentations.csv')
MAIN_DB_PATH = os.path.join(PROJECT_PATH, 'input/ELP-2016-04-07.csv')
VARS_SAVE_PATH = os.path.join(PROJECT_PATH, 'output/ELP morphological variables 4.csv')
HAPAX_FREQ_THRESHOLD = 0.02
DB_WORD_COL = 0
DB_FREQ_COL = 1
DB_POS_COL = 4
DB_SEGM_COL = 3


def apply_morpho_vars_to_lex_db(db, morpho_vars):
    """
    Build a dict {PRS_signature: [lexical_data]}, where lexical_data is a list
    of rows from the lexical database, each of which are expanded by including
    morphological variables for each segmented morphemes.
    """
    res = {}
    for row in db:
        temp = copy.deepcopy(row)
        segm = row[DB_SEGM_COL]
        word_freq = row[DB_FREQ_COL]
        prs = get_PRS_signature(row[DB_SEGM_COL])
        if prs not in res.keys():
            res[prs] = []
        morphemes = get_morphemes(segm)
        for m in morphemes:
            relative_fam_freq = get_relative_fam_freq(m, db, segm, word_freq)
            m_vars = [morpho_vars[m]['freq'], morpho_vars[m]['family_size'],
                      morpho_vars[m]['p'], morpho_vars[m]['p*'],
                      relative_fam_freq]
            temp.extend(m_vars)
        res[prs].append(temp)

    return res


def compute_morphological_variables(db, hapax_set):
    """
    For each morpheme in the segmented lexical database, compute the following
    attributes:

    - summed token frequency
    - family size
    - p-measure
    - p*-measure
    """
    morpho_vars = {}
    counted = set()

    for row in db:
        morphemes = get_morphemes(row[DB_SEGM_COL])
        pos_list = parse_pos_info(row[DB_POS_COL])  # Possibly more than one POS
        if not pos_list:
            continue
        for m in [x for x in morphemes if x not in counted]:
            freq = total_morpheme_freq(m, db)
            family = get_family(m, db)
            m_freq_in_hapax = total_morpheme_freq(m, hapax_set)
            morpho_vars[m] = {'freq': freq}
            morpho_vars[m]['family_size'] = len(family)
            if m_freq_in_hapax == 0:
                morpho_vars[m]['p'] = 0
                morpho_vars[m]['p*'] = 0
            else:
                # print(m, row[0], m_freq_in_hapax, freq)
                morpho_vars[m]['p'] = m_freq_in_hapax / freq
                morpho_vars[m]['p*'] = m_freq_in_hapax / len(hapax_set)
            counted.add(m)

    return morpho_vars


def generate_headers(prs):
    """
    Given a prs signature for a CSV database, returns the appropriate header
    structure (column names).
    """
    p, r, s = int(prs[0]), int(prs[1]), int(prs[2])
    headers = ['Word', 'SUBTLWF', 'LgSUBTLWF', 'MorphSp', 'POS', '']
    for i in range(1, p+1):
        headers.extend(['PREF%d_SumTokFreq' % i, 'PREF%d_FamSize' % i,
                        'PREF%d_P' % i, 'PREF%d_P*' % i,
                        'PREF%d_%%FamMoreFreq' % i])
    for i in range(1, r+1):
        headers.extend(['ROOT%d_SumTokFreq' % i, 'ROOT%d_FamSize' % i,
                        'ROOT%d_P' % i, 'ROOT%d_P*' % i,
                        'ROOT%d_%%FamMoreFreq' % i])
    for i in range(1, s+1):
        headers.extend(['SUFF%d_SumTokFreq' % i, 'SUFF%d_FamSize' % i,
                        'SUFF%d_P' % i, 'SUFF%d_P*' % i,
                        'SUFF%d_%%FamMoreFreq' % i])
    return headers


def get_family(morpheme, db):
    """
    Returns a dict {WORD: FREQUENCY} of all the words in database that contain
    the morpheme.
    """
    family = {}
    for row in db:
        if morpheme in row[DB_SEGM_COL]:
            if row[DB_FREQ_COL] != 'NULL':
                family[row[DB_SEGM_COL]] = row[DB_FREQ_COL]
            else:
                family[row[DB_SEGM_COL]] = 0

    return family


def get_hapax_set(db):
    """
    Returns the set of words in database that have frequency equal or lower to
    HAPAX_FREQ_THRESHOLD
    """
    hapax = []
    for row in db:
        if row[DB_FREQ_COL] != 'NULL':
            if row[DB_FREQ_COL] <= HAPAX_FREQ_THRESHOLD:
                hapax.append(row)
    return hapax


def get_morphemes(segm):
    """
    Parses the ELP morpheme segmentation string and returns a list of morphemes.
    """
    if segm == 'NULL':
        raise Exception("NULL segmentation OMG!")
    morphemes = re.findall(r'[<>^][^><^]+?[<>^]', segm)
    # rest = segm
    # for m in morphemes:
    #     rest = rest.replace(m, '')
    # if len(rest) > 0:
    #     print(word + '\t\t' + rest)
    # print(segm)
    # print(morphemes)
    # for m in morphemes:
    #     segm = segm.replace(m, '')
    # morphemes += [x[1] for x in re.findall(r'({|-)([^{}-]+?)(}|-)', segm)]
    return morphemes


def get_PRS_signature(segm):
    """
    Returns a signature composed of 3 integers, each digit representing
    # of prefixes, # of roots, and # of suffixes, respectively.
    """
    n_pref = segm.count('<') / 2
    n_root = segm.count('^') / 2
    n_suff = segm.count('>') / 2
    return (n_pref, n_root, n_suff)


def get_relative_fam_freq(morpheme, db, segm, word_freq):
    """
    Given a word and one of its morphemes, return the percentage of words in db
    that contain the same morpheme and are more frequent.
    """
    family = get_family(morpheme, db)
    if len(family) == 1:
        return 0
    family.pop(segm, None)
    more_frequent_in_fam = sum([1 for x in family.values() if x > word_freq])
    return (more_frequent_in_fam / len(family)) * 100


def merge_new_data_with_database(prs_data, main_db):
    for signature in prs_data:
        for word in prs_data[signature]:
            try:
                prs_data[signature][word] = main_db[word] + prs_data[signature][word]
            except KeyError:
                print(word)
                continue

    return prs_data


def p_measures(morpheme, morpheme_freq, hapax_set):
    """
    Computes the P and P* measures given a morpheme a corpus and the set of
    hapax legomena in the corpus. Returns a tuple containing (P, P*).

    hapax_set is the set of all words with smallest frequency in ELP, that is,
    with SUBTLWF = 0.02.

    For details, see:
    http://www.sfs.uni-tuebingen.de/~hbaayen/publications/BaayenLCP1994.pdf
    """
    # In case the frequency was NULL in the database
    if morpheme_freq == 0:
        return (0, 0)

    morph_type_count_in_hapax_set = 0

    p = morph_type_count_in_hapax_set / morpheme_freq
    p_star = morph_type_count_in_hapax_set / len(hapax_set)

    return (p, p_star)


def parse_pos_info(pos_info):
    """
    Parses a string that may contain multiple POS codes separated by pipelines.
    Some parsed POS are not acceptable and are discarded
    """
    return [x for x in re.split(r'\|', pos_info) if x not in {'encl', 'minor'}]


def preprocess_db(db):
    """
    Prepare csb data for processing by the program.
    """
    for row in db:
        if row[DB_FREQ_COL] != 'NULL':
            row[DB_FREQ_COL] = float(row[DB_FREQ_COL])
        else:
            row[DB_FREQ_COL] = 0
        row[DB_SEGM_COL] = re.sub(r'--?', '', row[DB_SEGM_COL])
        row[DB_SEGM_COL] = re.sub(r'{([^<>]+?)}', r'^\1^', row[DB_SEGM_COL])
        row[DB_SEGM_COL] = re.sub(r'(^|<|{)([^^{}<>]+?)(}|$|>)', r'\1^\2^\3',
                                  row[DB_SEGM_COL])
    return db


def total_morpheme_freq(morpheme, db):
    """
    Sums the frequencies of words containing the morpheme in the database.
    Returns an int.
    """
    total_freq = 0
    for row in db:
        if morpheme in row[DB_SEGM_COL]:
            count = 0 if row[DB_FREQ_COL] == 'NULL' else row[DB_FREQ_COL]
            total_freq += count

    return total_freq


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
    with open(SEGM_DB_PATH) as database:
        segm_db = [x for x in list(csv.reader(database))[1:] if x]  # skip headers

    print('preprocessing db')
    segm_db = preprocess_db(segm_db)
    print('getting hapax set')
    hapax_set = get_hapax_set(segm_db)
    print('computing morphological variables')
    morpho_vars = compute_morphological_variables(segm_db, hapax_set)
    print('applying morphological variables to database')
    new_data_by_prs = apply_morpho_vars_to_lex_db(segm_db, morpho_vars)

    with open(MAIN_DB_PATH) as database:
        main_db = [x for x in list(csv.reader(database))[2:] if x]  # skip headers
    print('merging new data with existing database')
    merged_data = merge_new_data_with_database(new_data_by_prs, main_db)

    # create output directory if it doesn't exist already
    os.makedirs('output', exist_ok=True)   
    for prs, data in new_data_by_prs.items():
        prs_str = re.sub(r'[,()]', '', str(prs))
        savepath = os.path.join(PROJECT_PATH, 'output/%s.csv' % prs_str)
        with open(savepath, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(generate_headers(prs))
            csvwriter.writerows(data)


    # save_morpho_vars_to_file(morpho_vars, VARS_SAVE_PATH)
