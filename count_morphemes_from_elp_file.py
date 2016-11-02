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
SEGM_DB_PATH = os.path.join(PROJECT_PATH, 'input/ELP_segmentations_3.csv')
MAIN_DB_PATH = os.path.join(PROJECT_PATH, 'input/ELP-2016-04-07.csv')
VARS_SAVE_PATH = os.path.join(PROJECT_PATH, 'output/ELP_morphological_variables.csv')
HAPAX_SBTL_FREQ_THRESHOLD = 0.02
HAPAX_HAL_FREQ_THRESHOLD = 2
DB_WORD_COL = 0
DB_SBTL_FREQ_COL = 1
DB_HAL_FREQ_COL = 2
DB_SEGM_COL = 3
DB_POS_COL = 4


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
            m_vars = [morpho_vars[m]['family_size'],
                      morpho_vars[m]['sbtl_freq'],
                      morpho_vars[m]['sbtl_rel_fam_freq'],
                      morpho_vars[m]['sbtl_p'],
                      morpho_vars[m]['sbtl_p*'],
                      morpho_vars[m]['hal_freq'],
                      morpho_vars[m]['hal_rel_fam_freq'],
                      morpho_vars[m]['hal_p'],
                      morpho_vars[m]['hal_p*']]
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
        segm = row[DB_SEGM_COL]
        morphemes = get_morphemes(segm)
        for m in [x for x in morphemes if x not in counted]:
            freq = total_morpheme_freq(m, db)
            family = get_family(m, db)
            relative_fam_freq = get_relative_fam_freq(m, db, segm, freq)
            morpho_vars[m] = {'sbtl_freq': freq['sbtl'], 'hal_freq': freq['hal']}
            morpho_vars[m]['family_size'] = len(family)
            morpho_vars[m]['sbtl_rel_fam_freq'] = relative_fam_freq['sbtl']
            morpho_vars[m]['hal_rel_fam_freq'] = relative_fam_freq['hal']
            hapax_freq = total_morpheme_freq(m, hapax_set)
            if hapax_freq == 0:
                morpho_vars[m]['sbtl_p'] = 0
                morpho_vars[m]['sbtl_p*'] = 0
                morpho_vars[m]['hal_p'] = 0
                morpho_vars[m]['hal_p*'] = 0
            else:
                # print(m, row[0], m_freq_in_hapax, freq)
                morpho_vars[m]['sbtl_p'] = hapax_freq['sbtl'] / freq['sbtl']
                morpho_vars[m]['sbtl_p*'] = hapax_freq['sbtl'] / len(hapax_set)
                morpho_vars[m]['hal_p'] = hapax_freq['hal'] / freq['hal']
                morpho_vars[m]['hal_p*'] = hapax_freq['hal'] / len(hapax_set)
            counted.add(m)

    return morpho_vars


def generate_headers(prs):
    """
    Given a prs signature for a CSV database, returns the appropriate header
    structure (column names).
    """
    p, r, s = int(prs[0]), int(prs[1]), int(prs[2])
    headers = ['Word', 'SUBTLWF', 'Freq_HAL', 'MorphSp', 'POS', '']
    for i in range(1, p+1):
        headers.extend(['PREF%d_FamSize' % i, 

                        'PREF%d_Freq_SBTL' % i, 'PREF%d_%%FamMoreFreq_SBTL' % i,
                        'PREF%d_P_SBTL' % i, 'PREF%d_P*_SBTL' % i,

                        'PREF%d_Freq_HAL' % i, 'PREF%d_%%FamMoreFreq_HAL' % i,
                        'PREF%d_P_HAL' % i, 'PREF%d_P*' % i
                        ])
    for i in range(1, r+1):
        headers.extend(['ROOT%d_FamSize' % i, 

                        'ROOT%d_Freq_SBTL' % i, 'ROOT%d_%%FamMoreFreq_SBTL' % i,
                        'ROOT%d_P_SBTL' % i, 'ROOT%d_P*_SBTL' % i,

                        'ROOT%d_Freq_HAL' % i, 'ROOT%d_%%FamMoreFreq_HAL' % i,
                        'ROOT%d_P_HAL' % i, 'ROOT%d_P*' % i
                        ])
    for i in range(1, s+1):
        headers.extend(['SUFF%d_FamSize' % i, 

                        'SUFF%d_Freq_SBTL' % i, 'SUFF%d_%%FamMoreFreq_SBTL' % i,
                        'SUFF%d_P_SBTL' % i, 'SUFF%d_P*_SBTL' % i,

                        'SUFF%d_Freq_HAL' % i, 'SUFF%d_%%FamMoreFreq_HAL' % i,
                        'SUFF%d_P_HAL' % i, 'SUFF%d_P*' % i
                       ])
    return headers


def get_family(morpheme, db):
    """
    Returns a dict {WORD: {'hal': HAL_FREQUENCY, 'sbtl': SBTL:FREQUENCY}} of all the words
    in database that contain the morpheme.
    """
    family = {}
    for row in db:
        if morpheme in row[DB_SEGM_COL]:
            if row[DB_SBTL_FREQ_COL] != 'NULL':
                family[row[DB_SEGM_COL]] = {'sbtl': row[DB_SBTL_FREQ_COL]}
            else:
                family[row[DB_SEGM_COL]] = {'sbtl': 0}
            family[row[DB_SEGM_COL]]['hal'] = row[DB_HAL_FREQ_COL]

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
        if row[DB_HAL_FREQ_COL] < HAPAX_HAL_FREQ_THRESHOLD:
            hapax.append(row)
        elif row[DB_SBTL_FREQ_COL] != 'NULL':
            if row[DB_SBTL_FREQ_COL] <= HAPAX_FREQ_THRESHOLD:
                hapax.append(row)
    return hapax


def get_morphemes(segm):
    """
    Parses the ELP morpheme segmentation string and returns a list of morphemes.
    """
    if segm == 'NULL':
        raise Exception("NULL segmentation OMG!")
    morphemes = re.findall(r'[<>{][^><}]+?[<>}]', segm)
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
    n_root = segm.count('{')
    n_suff = segm.count('>') / 2
    return (n_pref, n_root, n_suff)


def get_relative_fam_freq(morpheme, db, segm, word_freq, family):
    """
    Given a word and one of its morphemes, return the percentage of words in db
    that contain the same morpheme and are more frequent.
    """
    if len(family) == 1:
        return {'sbtl': 0, 'hal': 0}
    family.pop(segm, None)
    more_frequent_in_fam_sbtl = sum([1 for x in family.values() if x['sbtl'] > word_freq['sbtl']])
    more_frequent_in_fam_hal = sum([1 for x in family.values() if x['hal'] > word_freq['hal']])
    return {'sbtl': (more_frequent_in_fam_sbtl / len(family)) * 100,
            'hal': (more_frequent_in_fam_hal / len(family)) * 100}


def merge_new_data_with_database(prs_data, main_db):
    for signature in prs_data:
        for word in prs_data[signature]:
            try:
                prs_data[signature][word] = main_db[word] + prs_data[signature][word]
            except KeyError:
                print(word)
                continue

    return prs_data


def preprocess_db(db):
    """
    Prepare csv data for processing by the program.
    """
    for row in db:
        if row[DB_FREQ_COL] != 'NULL':
            row[DB_FREQ_COL] = float(row[DB_FREQ_COL])
        else:
            row[DB_FREQ_COL] = 0
    return db


def total_morpheme_freq(morpheme, db):
    """
    Sums the frequencies of words containing the morpheme in the database.
    Returns an int.
    """
    total_freq_sbtl = 0
    total_freq_hal = 0
    for row in db:
        if morpheme in row[DB_SEGM_COL]:
            count = 0 if row[DB_SBTL_FREQ_COL] == 'NULL' else row[DB_SBTL_FREQ_COL]
            total_freq_sbtl += count
            count = 0 if row[DB_HAL_FREQ_COL] == 'NULL' else row[DB_HAL_FREQ_COL]
            total_freq_hal += count

    return {'sbtl': total_freq_sbtl, 'hal': total_freq_hal}


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
