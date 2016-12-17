# !/usr/bin/env python
# encoding: utf-8

"""
Script used to fix the MorphSp column of the ELP database, and add the result as a
MSW_MorphSp column.

1. Remove non-derivational suffixes: ed, ing, s, 'll, 'd, etc.
2. Move the --o at the beginning of some classical greco-latin morphemes to the end of
   their immediate left neighbor 
3. Assign prefix/root/suffix status to underannotated sequences between curly brackets
   following these rules:
   - any morpheme that is ALL of the following:
        - between curly brackets
        - that occurs by itself in the database
        - of length > 3
        - that doesn't start with an uppercase character
          (to filter proper nouns, e.g., <e<{vince} isn't good, but caused by {Vince})

          OR that is:
        - between curly brackets
        - a greco-latin morpheme
     is a root.

   - everything in curly brackets to the left of a root is a prefix
   - everything in curly brackets to the right of a root is a suffix
 
4. Make all classical greco-latin morphemes roots
6. Eliminate double consonants at beginning of roots
7. Normalize morpheme spelling


Two strategies to resegmentation:
Hyper-segmentation:
    Every time we see an affix, we mark it as such. Anything else is a root. There is
    no merging of morphemes previously segmented in ELP.
Hypo-segmentation: 
    Every time we see a root (free or classical), we mark it as such. Anything between
    curly brackets that is not marked as such is a suffix if there is a root on the left,
    and a prefix if there is a root on the right. Anything between curly crackets not
    marked at the end of this process is merged with its neighboring morphemes
    and forms a root.

------------------------------------------

"""

import csv
import re
import os
import json


project_fp = '/home/hugo/Projects/ELP_morpho_vars'
elp_fp = os.path.join(project_fp, 'input/ELP-2016-12-10.csv')
output_fp = os.path.join(project_fp, 'input/ELP-2016-12-15.csv')
roots_fp = os.path.join(project_fp, 'linguistic_data/roots.txt')
non_roots_fp = os.path.join(project_fp, 'linguistic_data/non_roots.txt')
allo_prefs_fp = os.path.join(project_fp, 'linguistic_data/rev_allomorphs_prefixes.json')
allo_roots_fp = os.path.join(project_fp, 'linguistic_data/rev_allomorphs_roots.json')
allo_suffs_fp = os.path.join(project_fp, 'linguistic_data/rev_allomorphs_suffixes.json')

def rreplace(s, old, new, count):
    """ Replaces only the rightmost occurrence of old in string s, count times."""
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]


with open(elp_fp) as f:
    reader = csv.reader(f)
    h1 = next(reader)  # Keep headers 1 and 2 in separate variables
    h2 = next(reader)
    elp = list(reader)   # This is the data, without headers

# 46 is MorphSp
# 47 is MorphSp_revised
# 48 is MorphoLexSegm
# 49 is MorphoLexSegmMerged
# Copy MorphSp_revised (segmentation) column in new list, we work on that list
# allo_segm will hold the segmentation where morphs are replaced by their canonical forms
new_segm = [x[47] for x in elp]
old_segm = [x[47] for x in elp]

################################################################
### Apply miscellaneous fixes to erroneous ELP segmentations ###
################################################################

# 1. Remove non-derivational suffixes: ed/d, ing, s, and contractions like 'll, 's, etc.
new_segm = [re.sub(r">(ed|d|ing|s|\w*'\w*)>$", '', segm) for segm in new_segm]

free_roots = [re.sub(r'[{}]', '', x) for i,x in enumerate(new_segm)
              if not re.search(r'([<>-]|\}\{)', x)  # Must be only morpheme in x
                                                    # ("}{" would mean two roots)
                 and len(x) > 5                     # Must have more than 3 chars (+2 for {})
                 and not elp[i][1][0].isupper()]    # Word can't begin with uppercase letter

with open(roots_fp) as f:
    roots = set(f.read().split('\n'))
with open(non_roots_fp) as f:
    non_roots = set(f.read().split('\n'))

with open(allo_prefs_fp) as f:
    allo_prefs = json.load(f)
with open(allo_roots_fp) as f:
    allo_roots = json.load(f)
with open(allo_suffs_fp) as f:
    allo_suffs = json.load(f)

roots = roots.union(set(free_roots)) - set(non_roots)

# Annotate roots
for i, segm in enumerate(new_segm):
    rts = [x for x in re.findall(r'[<>{}-](.+?)[<>{}-]', segm) if x in roots]
    for r in rts:
        new_segm[i] = new_segm[i].replace(r, '('+r+')')

# Remove affix notation where we changed to root notation
new_segm = [re.sub(r'[><](\(\w+?\))[><]', r'\1', x) for x in new_segm]

# Annotate suffixes between curly brackets
for i, segm in enumerate(new_segm):
    suffs_sequence = ''.join([x[0] for x in re.findall(r'\)((--\w+)*--\w+)', segm)])
    # Output for {re--anti--pre--(hyster)--ec--tom--y--(other)--stuff}
    # '--ec--tom--y--stuff'
    suffs = re.findall(r'--\w+', suffs_sequence)
    # ['--ec', '--tom', '--y', '--stuff']
    for suff in suffs:
        # Take care to replace only last occurrence of suff
        # (There could be a prefix with the same spelling)
        new_segm[i] = rreplace(new_segm[i], suff, '>'+suff+'>', 1)

# Annotate prefixes between curly brackets
for i, segm in enumerate(new_segm):
    prefs = re.findall(r'(\w+--)(?=[^\)\}]*\()', segm) 
    # Output for {re--anti--pre--(hyster)--ec--tom--y--(other)--stuff}
    # ['re--', 'anti--', 'pre--']
    for pref in prefs:
        # Take care to replace only first occurrence of pref
        # (There could be a suffix with the same spelling)
        new_segm[i] = new_segm[i].replace(pref, '<'+pref+'<', 1)

# Remove dashes
new_segm = [x.replace('-', '') for x in new_segm]

# Any uninterrupted alphabetic sequence between curly brackets not marked as root
# must be marked as root
new_segm = [re.sub(r'\{(\w+)\}', r'{(\1)}', x) for x in new_segm]

# Build allomorph-merged segmentations from new_segm
allo_segm = new_segm[:]

# Replace each morph by its canonical form, if any
for i, segm in enumerate(new_segm):
    prefs = re.findall(r'<(\w+)<', segm)
    rts = re.findall(r'\((\w+)\)', segm)
    suffs = re.findall(r'>(\w+)>', segm)
    for p in prefs:
        if p in allo_prefs:
            allo_segm[i] = allo_segm[i].replace('<'+p+'<', '<'+allo_prefs[p]+'<')        
    for r in rts:
        if r in allo_roots:
            allo_segm[i] = allo_segm[i].replace('('+r+')', '('+allo_roots[r]+')')
    for s in suffs:
        if s in allo_suffs:
            allo_segm[i] = rreplace(allo_segm[i], '>'+s+'>', '>'+allo_suffs[s]+'>', 1)

# Save elements of new_segm as the 48th column of the ELP database
for i, segm in enumerate(new_segm):
    elp[i][48] = segm
# Save elements of allo_segm as the 49th column of the ELP database
for i, segm in enumerate(allo_segm):
    elp[i][49] = segm

with open(output_fp, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(h1)
    writer.writerow(h2)
    writer.writerows(elp)

with open('segm_to_fix.txt', 'w') as f:
    for i, _ in enumerate(new_segm):
        f.write(old_segm[i]+','+new_segm[i]+'\n')