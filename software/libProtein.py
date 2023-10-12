#!/usr/bin/env python

THREE_TO_ONE = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',\
                'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',\
                'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',\
                'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

ONE_TO_THREE = dict(zip(THREE_TO_ONE.values(), THREE_TO_ONE.keys()))

def return_one_letter_aa(line):
    return THREE_TO_ONE[line[17:20]]

def three_aa_to_one_aa(aa_name):
    if aa_name not in THREE_TO_ONE.keys():
        return 'X'
    else:
        return THREE_TO_ONE[aa_name]

    
