# Import `Python` modules
import os
from os import listdir
import glob
import subprocess
import time
import sys
import pandas as pd
from sys import argv
from pyrosetta import *
from scipy.spatial import distance
pyrosetta.init()


def count_hbonds_protein_dna(pose, tag, energy_cutoff) :

    ## Motif in ss_H3?
    rif_resis = []
    rec_helix = []
    motif = []
    info = pose.pdb_info()
    for res in range(1,pose.total_residue()+1):
        reslabel = info.get_reslabels(res)
        if 'RIFRES' in reslabel:
            rif_resis.append(res)
        if 'ss_RH' in reslabel:
            rec_helix.append(res)
        if 'MOTIF' in reslabel:
            motif.append(res)

    # I did this because the rec helix in 1L3L and 1PER topologies are annotated ss_H3
    if len(rec_helix) == 0:
        for res in range(1,pose.total_residue()+1):
            reslabel = info.get_reslabels(res)
            if 'ss_H3' in reslabel:
                rec_helix.append(res)

    rifres_in_rec_helix = False
    for rif_resi in rif_resis:
        if rif_resi in rec_helix:
            rifres_in_rec_helix = True

    motif_in_rec_helix = False
    if len(rec_helix) > 0 and len(motif) > 0:
        if (min(rec_helix) + 1 == min(motif)) or (max(rec_helix) -1 == max(motif)):
            motif_in_rec_helix = True


    pose_hb = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose)
    pose_hb = pose.get_hbonds()
    hbond_id = []
    hbond_dna = []
    hbonds_don_hatm = []
    hbonds_acc_atm = []
    hbonds_don_res = []
    hbonds_acc_res = []
    hbonds_energy = []
    involves_dna = []

    for hbond in range (1,pose_hb.nhbonds()+1):
        hbond_id.append(hbond)
        hbond = pose_hb.hbond(hbond)
        donor_hatm = hbond.don_hatm()
        acceptor_atm = hbond.acc_atm()
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        hbond_energy = hbond.energy()
        hbonds_don_hatm.append(donor_hatm)
        hbonds_acc_atm.append(acceptor_atm)
        hbonds_don_res.append(donor_res)
        hbonds_acc_res.append(acceptor_res)
        hbonds_energy.append(hbond_energy)


    ### define base-pairs
    strand1 = []
    strand2 = []
    bp_don_dict = {}
    bp_acc_dict = {}
    for residue in range(len(hbonds_acc_res)) :
        acc_idx = hbonds_acc_res[residue]
        don_idx = hbonds_don_res[residue]
        acc_id = pose.residue(acc_idx).name().split(':')[0]
        don_id = pose.residue(don_idx).name().split(':')[0]
        if acc_id == 'GUA' and don_id in ['CYT','RCY'] :
            if acc_idx < don_idx:
                strand1.append(acc_idx)
                strand2.append(don_idx)
            else:
                strand1.append(don_idx)
                strand2.append(acc_idx)
            bp_don_dict[hbonds_don_res[residue]] = hbonds_acc_res[residue]
            bp_acc_dict[hbonds_acc_res[residue]] = hbonds_don_res[residue]
        if acc_id == 'ADE' and don_id == 'THY' :
            if acc_idx < don_idx:
                strand1.append(acc_idx)
                strand2.append(don_idx)
            else:
                strand1.append(don_idx)
                strand2.append(acc_idx)
            bp_don_dict[hbonds_don_res[residue]] = hbonds_acc_res[residue]
            bp_acc_dict[hbonds_acc_res[residue]] = hbonds_don_res[residue]

    ## define aas, hbond_types, and initialize aa_hbond_dict
    aas = ['ARG','LYS','ASP','GLU','ASN','GLN','SER','THR','TYR','CYS','HIS','ALA','GLY','ILE','MET','PRO','PHE','VAL','HIS_D','TRP','CYV','LEU']

    hbond_types = ['g_hbonds','c_hbonds','a_hbonds','t_hbonds',
                   'g_bidentates','a_bidentates','g_c_cross_bidentates','a_t_cross_bidentates',
                   'g_c_complex_hbonds','a_t_complex_hbonds','g/g_stacked_bidentates','g/c_stacked_bidentates',
                   'g/a_stacked_bidentates','g/t_stacked_bidentates','a/a_stacked_bidentates','a/t_stacked_bidentates',
                   'triple_base_hbonds','phosphate_hbonds','phosphate_bidentates','phosphate_base_bidentates',
                   'base_hbonds','g_c_bidentates','a_t_bidentates','base_step_bidentates','base_step_complex_hbonds',
                   'stacked_bidentates','base_bidentates','hbonds','base_bidentates_w_phosphates','bidentates','cross_step_bidentates']

    aa_hbond_dict = dict.fromkeys(aas)
    for key in aa_hbond_dict:
        aa_hbond_dict[key] = [0]*len(hbond_types)

    prot_phosphate_hbonds = {}
    prot_base_hbonds = {}

    strand1_hbonds = 0
    strand2_hbonds = 0

    # this for loop counts all hbonds in phosphates and base steps
    for residue in range(len(hbonds_acc_res)) :
        acc_id = pose.residue(hbonds_acc_res[residue]).name().split(':')[0]
        don_id = pose.residue(hbonds_don_res[residue]).name().split(':')[0]

        if hbonds_energy[residue] > float(energy_cutoff):
            continue

        if acc_id in ['GUA','CYT','ADE','THY','RCY'] and don_id in aas:
            acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")
            aa_idx = hbonds_don_res[residue]
            aa = don_id
            if acc_atom_type in ['OP1','OP2']:
                aa_hbond_dict[aa][17] += 1 # add aa_phosphate single hbond
                try:
                    prot_phosphate_hbonds['{0}'.format(aa_idx)].append(acc_id)
                    aa_hbond_dict[aa][18] += 1 # add aa_phosphate oxygen bidentate
                except:
                    prot_phosphate_hbonds['{0}'.format(aa_idx)] = [acc_id]


        if acc_id == 'GUA' and don_id in aas:
            acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")
            if acc_atom_type in ['O6','N7'] :
                basepair_idx = hbonds_acc_res[residue]
                try:
                    base_idx = bp_acc_dict[basepair_idx]
                except:
                    base_idx = 10000 # do this if base does not exist in a pair
                aa_idx = hbonds_don_res[residue]
                aa = don_id
                # check for strand1 or strand2 contact
                if hbonds_acc_res[residue] in strand1:
                    strand1_hbonds += 1
                else:
                    strand2_hbonds += 1
                aa_hbond_dict[aa][0] += 1 # add _aa_g single hbond
                try: # check for bidentates in a single base-step
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)].append(acc_atom_type)
                    prot_base_hbonds_curr = prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)]
                    # check if this is aa_g_bidentate
                    if sorted(prot_base_hbonds_curr) == sorted(['O6','N7']) or sorted(prot_base_hbonds_curr) == sorted(['O6','N7','N7']) or sorted(prot_base_hbonds_curr) == sorted(['O6','N7','O6']) :
                        aa_hbond_dict[aa][4] += 1 # add aa_g bidentate
                        #aa_hbond_dict[aa][0] += -1 # remove aa_g single hbond
                    # check if this is aa_g_c_bidentate
                    elif ( sorted(prot_base_hbonds_curr) == sorted(['O6','O4']) ) or ( sorted(prot_base_hbonds_curr) == sorted(['N7','O4']) ):
                        aa_hbond_dict[aa][6] += 1 # add aa_g_c bidentate
                        #aa_hbond_dict[aa][1] += -1 # remove aa_c single hbond
                    # check if this is aa_g_c_tridentate
                    elif sorted(prot_base_hbonds_curr) == sorted(['O6','N7','H42']):
                        aa_hbond_dict[aa][8] += 1 # add aa_g_c complex
                        #aa_hbond_dict[aa][6] += -1 # remove aa_g_c bidentate
                except: # this is an aa_g single hbond (so far)
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)] = [acc_atom_type]


        if don_id in ['CYT','RCY'] and acc_id in aas :
            don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
            if don_atom_type in ['H42'] :
                base_idx = hbonds_don_res[residue]
                try:
                    basepair_idx = bp_don_dict[base_idx]
                except:
                    basepair_idx = 10000 # do this if base does not exist in a pair
                aa_idx = hbonds_acc_res[residue]
                aa = acc_id
                # check for strand1 or strand2 contact
                if hbonds_don_res[residue] in strand1:
                    strand1_hbonds += 1
                else:
                    strand2_hbonds += 1
                aa_hbond_dict[aa][1] += 1 # add aa_c single hbond
                try: # check for bidentates in a single base-step
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)].append(don_atom_type)
                    prot_base_hbonds_curr = prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)]
                    # check if this is aa_g_c_bidentate
                    if ( sorted(prot_base_hbonds_curr) == sorted(['O6','H42']) ) or ( sorted(prot_base_hbonds_curr) == sorted(['N7','H42']) ):
                        aa_hbond_dict[aa][6] += 1 # add aa_g_c bidentate
                        #aa_hbond_dict[aa][0] += -1 # remove aa_g single hbond
                    # check if this is aa_g_c_tridentate
                    elif sorted(prot_base_hbonds_curr) == sorted(['O6','N7','H42']):
                        aa_hbond_dict[aa][8] += 1  # add aa_g_c complex
                        #aa_hbond_dict[aa][4] += -1 # remove aa_g bidentate
                except: # this is a aa_c single hbond (so far)
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)] = [don_atom_type]

        if don_id in ['ADE'] and acc_id in aas :
            don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
            if don_atom_type in ['H62'] :
                base_idx = hbonds_don_res[residue]
                try:
                    basepair_idx = bp_acc_dict[base_idx]
                except:
                    basepair_idx = 10000 # do this if base does not exist in a pair
                aa_idx = hbonds_acc_res[residue]
                aa = acc_id
                # check for strand1 or strand2 contact
                if hbonds_don_res[residue] in strand1:
                    strand1_hbonds += 1
                else:
                    strand2_hbonds += 1
                aa_hbond_dict[aa][2] += 1 # add aa_a single hbond
                try: # check for bidentates in a single base-step
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)].append(don_atom_type)
                    prot_base_hbonds_curr = prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)]
                    # check if this is aa_a_bidentate
                    if sorted(prot_base_hbonds_curr) == sorted(['N7','H62']):
                        aa_hbond_dict[aa][5] += 1 # add aa_a tridentate
                        #aa_hbond_dict[aa][2] += -1 # remove aa_a single hbond
                    # check if this is aa_a_t bidentate
                    elif sorted(prot_base_hbonds_curr) == sorted('H62','O4'):
                        aa_hbond_dict[aa][7] += 1  # add aa_a_t bidentate
                        #aa_hbond_dict[aa][3] += -1 # remove aa_t single hbond
                    # check if this is aa_a_t tridentate
                    elif sorted(prot_base_hbonds_curr) == sorted('N7','H62','O4'):
                        aa_hbond_dict[aa][9] += 1 # add aa_a_t complex
                        #aa_hbond_dict[aa][7] += -1 # remove aa_a_t bidentate
                except: # this is aa_a single hbond (so far)
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)] = [don_atom_type]

        if acc_id in ['ADE'] and don_id in aas :
            acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")
            if acc_atom_type in ['N7'] :
                base_idx = hbonds_acc_res[residue]
                try:
                    basepair_idx = bp_acc_dict[base_idx]
                except:
                    basepair_idx = 10000 # do this if base does not exist in a pair
                aa_idx = hbonds_don_res[residue]
                aa = don_id
                # check for strand1 or strand2 contact
                if hbonds_acc_res[residue] in strand1:
                    strand1_hbonds += 1
                else:
                    strand2_hbonds += 1
                aa_hbond_dict[aa][2] += 1 # add aa_a single hbond
                try: # check for bidentates in a single base-step
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)].append(acc_atom_type)
                    prot_base_hbonds_curr = prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)]
                    # check if this is aa_a_bidentate
                    if sorted(prot_base_hbonds_curr) == sorted(['N7','H62']):
                        aa_hbond_dict[aa][5] += 1 # add aa_a bidentate
                        #aa_hbond_dict[aa][2] += -1 # remove aa_a single hbond
                    # check if this is aa_a_t_bidentate
                    elif sorted(prot_base_hbonds_curr) == sorted(['N7','O4']):
                        aa_hbond_dict[aa][7] += 1 # add aa_a_t bidentate
                        #aa_hbond_dict[aa][3] += -1 # remove aa_t single hbond
                    # check if this is aa_a_t_tridentate
                    elif sorted(prot_base_hbonds_curr) == sorted(['N7','H62','O4']):
                        aa_hbond_dict[aa][9] += 1 # add aa_a_t complex
                        #aa_hbond_dict[aa][7] += -1 # remove aa_a_t bidentate
                except: # this is aa_a single hbond (so far)
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)] = [acc_atom_type]

        if acc_id in ['THY'] and don_id in aas :
            acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")
            if acc_atom_type in ['O4'] :
                basepair_idx = hbonds_acc_res[residue]
                try:
                    base_idx = bp_don_dict[basepair_idx]
                except:
                    base_idx = 10000 # do this if base does not exist in a pair
                aa_idx = hbonds_don_res[residue]
                aa = don_id
                # check for strand1 or strand2 contact
                if hbonds_acc_res[residue] in strand1:
                    strand1_hbonds += 1
                else:
                    strand2_hbonds += 1
                aa_hbond_dict[aa][3] += 1 # add aa_a single hbond
                try: # check for bidentates in a single base-step
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)].append(acc_atom_type)
                    prot_base_hbonds_curr = prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)]
                    # check if this is aa_a_t bidentate
                    if sorted(prot_base_hbonds_curr) == sorted(['N7','O4']) or sorted(prot_base_hbonds_curr) == sorted(['H62','O4']) :
                        aa_hbond_dict[aa][7] += 1 # add aa_a_t bidentate
                        #aa_hbond_dict[aa][2] += -1 # remove aa_a single hbond
                    # check if this is aa_a_t tridentate
                    elif sorted(prot_base_hbonds_curr) == sorted(['N7','H62','O4']):
                        aa_hbond_dict[aa][9] += 1 # add aa_a_t complex
                        #aa_hbond_dict[aa][5] += -1 # remove aa_a bidentate
                except: # this is aa_a single hbond (so far)
                    prot_base_hbonds['{0}-{1}-{2}'.format(aa_idx,base_idx,basepair_idx)] = [acc_atom_type]
    observed_pairs = []
    triple_base_hbonds = {}
    # count bidentates and complex hbonds across stacked and diagonal bases, also count bidentate phosphate-base hbonds
    for i in prot_base_hbonds :
        aa_i_idx, base_i_idx, basepair_i_idx = i.split('-')
        aa_i_idx = int(aa_i_idx)
        base_i_idx = int(base_i_idx)
        basepair_i_idx = int(basepair_i_idx)

        for phosphate_hbond in prot_phosphate_hbonds:
            aa_l_idx = int(phosphate_hbond)
            if aa_i_idx == aa_l_idx:
                aa_hbond_dict[aa][19] += 1

        remainder = prot_base_hbonds.copy()
        del remainder[i]
        for j in remainder:
            aa_j_idx, base_j_idx, basepair_j_idx = j.split('-')
            aa_j_idx = int(aa_j_idx)
            base_j_idx = int(base_j_idx)
            basepair_j_idx = int(basepair_j_idx)

            if i+j in observed_pairs: # Continue loop if pair is already observed
                continue
            observed_pairs.append(j+i)
            if aa_i_idx != aa_j_idx: # Continue loop if same residue is not contacting multiple base steps
                continue
            aa = pose.residue(aa_i_idx).name().split(':')[0]
            if base_i_idx != 10000:
                #try:
                base = pose.residue(base_i_idx).name().split(':')[0]
                #except:
                    #base = 'NAN' # do this if base does not exist in a pair
            else:
                base = 'NAN' # do this if base does not exist in a pair
            if basepair_i_idx != 10000:
                #try:
                basepair = pose.residue(basepair_i_idx).name().split(':')[0]
                #except:
                    #basepair = 'NAN' # do this if base does not exist in a pair
            else:
                basepair = 'NAN' # do this if base does not exist in a pair

            # determine identity of the 2nd base step
            if base_i_idx == base_j_idx + 1 :
                stacked_base = pose.residue(base_j_idx).name().split(':')[0]
                try:
                    diagonal_base = pose.residue(basepair_j_idx).name().split(':')[0]
                except:
                    diagonal_base = 'NAN' # do this if base does not exist in a pair
            elif base_i_idx == basepair_j_idx + 1 :
                stacked_base = pose.residue(basepair_j_idx).name().split(':')[0]
                try:
                    diagonal_base = pose.residue(base_j_idx).name().split(':')[0]
                except:
                    diagonal_base = 'NAN' # do this if base does not exist in a pair
            else: # continue loop if base_i_idx not 1 greater than base_j_idx or basepair_j_idx to avoid double counting
                continue
            # count cross base step bidentates or complex hbonds for g/g, g/c, g/a, g/t, c/c, c/a, c/t, a/a, a/t, t/t
            pair1 = ['GUA','GUA','GUA','GUA','ADE','ADE']
            pair2 = ['GUA','CYT','ADE','THY','ADE','THY']
            position = 10
            for base_step in range(len(pair1)):
                if ( base == pair1[base_step] and stacked_base == pair2[base_step] ) or ( base == pair2[base_step] and stacked_base == pair1[base_step]):
                    aa_hbond_dict[aa][position] += 1
                elif ( basepair == pair1[base_step] and diagonal_base == pair2[base_step] ) or ( basepair == pair2[base_step] and diagonal_base == pair1[base_step]):
                    aa_hbond_dict[aa][position] += 1
                position += 1

            # count up triple base bidentates
            remainder2 = remainder.copy()
            del remainder2[j]
            for k in remainder2:
                aa_k_idx, base_k_idx, basepair_k_idx = k.split('-')
                aa_k_idx = int(aa_k_idx)

                if aa_i_idx == aa_k_idx: # Continue loop if same residue is not contacting multiple base steps
                    try:
                        triple_base_hbonds[aa_i_idx].append(1)
                    except:
                        triple_base_hbonds[aa_i_idx] = [1]
                        aa_hbond_dict[aa][16] += 1

    for aa in aa_hbond_dict:
        aa_hbond_dict[aa][20] = aa_hbond_dict[aa][0] + aa_hbond_dict[aa][1] + aa_hbond_dict[aa][2] + aa_hbond_dict[aa][3] # count all base_contacts made by aa
        aa_hbond_dict[aa][21] = aa_hbond_dict[aa][4] + aa_hbond_dict[aa][6] # count all g_c bidentates made by aa
        aa_hbond_dict[aa][22] = aa_hbond_dict[aa][5] + aa_hbond_dict[aa][7] # count all a_t bidentates made by aa
        aa_hbond_dict[aa][23] = aa_hbond_dict[aa][4] + aa_hbond_dict[aa][5] + aa_hbond_dict[aa][6] + aa_hbond_dict[aa][7] # count all base step bidentates made by aa
        aa_hbond_dict[aa][24] = aa_hbond_dict[aa][8] + aa_hbond_dict[aa][9] # count all single base step complex hbonds made by aa
        aa_hbond_dict[aa][25] = sum(aa_hbond_dict[aa][10:16]) # count all stacked/diagonal bidentates or complex hbonds made by aa
        aa_hbond_dict[aa][26] = aa_hbond_dict[aa][23] + aa_hbond_dict[aa][25] # count all base bidentates by aa (excluding phosphate base bidentates)
        aa_hbond_dict[aa][27] = aa_hbond_dict[aa][20] + aa_hbond_dict[aa][17] # count all contacts (base + phosphate) made by aa
        aa_hbond_dict[aa][28] = aa_hbond_dict[aa][19] + aa_hbond_dict[aa][26] # count all base bidentates by aa (including phosphate base bidentates)
        aa_hbond_dict[aa][29] = aa_hbond_dict[aa][18] + aa_hbond_dict[aa][28] # count all bidentates by aa (phosphate + base)
        aa_hbond_dict[aa][30] = aa_hbond_dict[aa][6] + aa_hbond_dict[aa][7] # count all cross step bidentates by aa

    columns = ['Tag','dominant_strand_percent','rifres_in_rec_helix','motif_in_rec_helix','total_g_hbonds','total_c_hbonds','total_a_hbonds','total_t_hbonds',
                'total_g_bidentates','total_a_bidentates','total_g_c_cross_bidentates','total_a_t_cross_bidentates',
                'total_g_c_complex_hbonds','total_a_t_complex_hbonds','total_g/g_stacked_bidentates','total_g/c_stacked_bidentates',
                'total_g/a_stacked_bidentates','total_g/t_stacked_bidentates','total_a/a_stacked_bidentates','total_a/t_stacked_bidentates',
                'total_triple_base_hbonds','total_phosphate_hbonds','total_phosphate_bidentates','total_phosphate_base_bidentates',
                'total_base_hbonds','total_g_c_bidentates','total_a_t_bidentates','total_base_step_bidentates','total_base_step_complex_hbonds',
                'total_stacked_bidentates','total_base_bidentates','total_hbonds','total_base_bidentates_w_phosphates','total_bidentates','total_cross_step_bidentates']

    result = [0]*len(columns)
    result[columns.index('Tag')] = tag
    try:
        result[columns.index('dominant_strand_percent')] = max(strand1_hbonds/(strand1_hbonds+strand2_hbonds),strand2_hbonds/(strand1_hbonds+strand2_hbonds))*100
    except:
        result[columns.index('dominant_strand_percent')] = 100
    result[columns.index('rifres_in_rec_helix')] = rifres_in_rec_helix
    result[columns.index('motif_in_rec_helix')] = motif_in_rec_helix
    for j in range(len(columns[4:])):
        for aa in aas:
            result[j+4] += aa_hbond_dict[aa][j]

    for aa in aas:
        aa_columns = ['{0}_g_hbonds'.format(aa),'{0}_c_hbonds'.format(aa),'{0}_a_hbonds'.format(aa),'{0}_t_hbonds'.format(aa),
                   '{0}_g_bidentates'.format(aa),'{0}_a_bidentates'.format(aa),'{0}_g_c_cross_bidentates'.format(aa),
                   '{0}_a_t_cross_bidentates'.format(aa),'{0}_g_c_complex_hbonds'.format(aa),'{0}_a_t_complex_hbonds'.format(aa),
                   '{0}_g/g_stacked_bidentates'.format(aa),'{0}_g/c_stacked_bidentates'.format(aa),'{0}_g/a_stacked_bidentates'.format(aa),
                   '{0}_g/t_stacked_bidentates'.format(aa),'{0}_a/a_stacked_bidentates'.format(aa),'{0}_a/t_stacked_bidentates'.format(aa),
                   '{0}_triple_base_hbonds'.format(aa),'{0}_phosphate_hbonds'.format(aa),'{0}_phosphate_bidentates'.format(aa),
                   '{0}_phosphate_base_bidentates'.format(aa),'{0}_base_hbonds'.format(aa),'{0}_g_c_bidentates'.format(aa),'{0}_a_t_bidentates'.format(aa),
                   '{0}_base_step_bidentates'.format(aa),'{0}_base_step_complex_hbonds'.format(aa),'{0}_stacked_bidentates'.format(aa),
                   '{0}_base_bidentates'.format(aa),'{0}_hbonds'.format(aa),'{0}_base_bidentates_w_phosphates'.format(aa),'{0}_bidentates'.format(aa),
                   '{0}_cross_step_bidentates'.format(aa)]
        aa_result = aa_hbond_dict[aa]
        for j in range(len(aa_columns)):
            columns.append(aa_columns[j])
            result.append(aa_result[j])
    return columns, result
