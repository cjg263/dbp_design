#!/usr/bin/env python

import os, sys
import math

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# get the program directory
from sys import argv
prog_dir = os.path.dirname(sys.argv[0]) 
sys.path.append(prog_dir)

# Load external paths
path_file = prog_dir + '/external_paths.txt'
with open(path_file,'r') as f_in:
    paths_dict = {}
    for line in f_in.readlines():
        key = line.split(',')[0]
        path = line.split(',')[1].rstrip()
        paths_dict[key] = path


# Build a PyRosetta with Python3.9
# Will it work?
sys.path.insert( 0, paths_dict['pyrosetta_path'] )

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.io.pdb import create_pdb_contents_from_sfr
from pyrosetta.rosetta.core.io.pose_to_sfr import PoseToStructFileRepConverter


import numpy as np
from collections import defaultdict
from collections import OrderedDict
import time
import argparse
import itertools
import subprocess
import time
import pandas as pd
import glob
import random
from decimal import Decimal

from Bio import AlignIO
import tempfile

import torch

# import function for counting hydrogen bonds
import count_hbond_types, compactness_filter

# import util for loading rosetta movers
sys.path.append(prog_dir)
import xml_loader

sys.path.insert( 0, paths_dict['silent_tools_path'])
import silent_tools 

#sys.path.append( paths_dict['ligand_mpnn_path'])
import run #from run import main, get_argument_parser


init( "-mute all -beta_nov16 -in:file:silent_struct_type binary" +
    " -holes:dalphaball /software/rosetta/DAlphaBall.gcc"  +
    " -use_terminal_residues true -mute basic.io.database core.scoring" +
    f"@{prog_dir}/flags_and_weights/RM8B.flags")


def cmd(command, wait=True):
    the_command = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if (not wait):
        return
    the_stuff = the_command.communicate()
    return str( the_stuff[0]) + str(the_stuff[1] )

def range1( iterable ): return range( 1, iterable + 1 )

#################################
# Argument Validation Functions
#################################

def design_xml_check( input_val ):
    if not os.path.exists( input_val ):
        raise argparse.ArgumentTypeError( 'The argument -design_xml_check must be a path to an xml file, you used: %s'%input_val )
    return input_val

def feature_check( input_val ):
    if input_val not in [ 'full', 'coarse' ]:
        raise argparse.ArgumentTypeError( 'The argument -protein_features must be either full or coarse, you used: %s'%input_val )
    return input_val


#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()
parser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run.' )
parser.add_argument( "-pdb", type=str, default="", help='Single PDB file.' )
parser.add_argument( "-pdb_folder", type=str, default="", help='Folder if pdbs.' )
parser.add_argument( "-silent_tag", type=str, default = '', help="Input silent tag")
parser.add_argument( "-silent_tag_file", type=str, default = '', help="File containing tags to design in silent")
parser.add_argument( "-silent_tag_prefix", type=str, default = '', help="Prefix for silent tag outputs")
parser.add_argument( "-silent_output_path_prefix", type=str, default = 'out.silent', help='path of silent output file')
parser.add_argument( "-seq_per_struct", type=int, default=1, help='Number of MPNN sequences that will be produced' )
parser.add_argument( "-checkpoint_path", type=str, default=f'{paths_dict["mpnn_model_checkpoint"]}')
parser.add_argument( "-path_to_model_weights_sc", type=str, default=f'{paths_dict["sc_checkpoint_path"]}', help="Path to model weights folder;")
parser.add_argument( "-model_name_sc", type=str, default="v_32_005", help="Side Chain LigandMPNN model name: v_32_005")
parser.add_argument( "-mpnn_sidechain_packing", type=int, default=1, help="1 - to pack side chains, 0 - do not")
parser.add_argument( "-pssm_type", type=str, default=".2.20.pssm", help="Type of pssm to use." )
parser.add_argument( "-pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
parser.add_argument( "-pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
parser.add_argument( "-pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
parser.add_argument( "-pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
parser.add_argument( "-temperature", type=float, default=0.1, help='I will tell you what it is not--An a3m file containing the MSA of your target' )
parser.add_argument( "-augment_eps", type=float, default=0.05, help='The variance of random noise to add to the atomic coordinates (default 0.05)' )
parser.add_argument( "-ddg_cutoff", type=float, default=100, help='The threshold that predicted ddg must pass for a structure to be written to disk (default 100)' )
parser.add_argument( "-num_connections", type=int, default=48, help='Number of neighbors each residue is connected to, default 30, maximum 64, higher number leads to better interface design but will cost more to run the model.' )
parser.add_argument( "-freeze_hbond_resis", type=int, default=0, help='By default (0), we allow redesign of base contacting hbond resis. If 1, MPNN will freeze them.' )
parser.add_argument( "-freeze_resis", type=str, default='', help="Residues to freeze in MPNN design.")
parser.add_argument( "-freeze_seed_resis", type=int, default=0, help="Freeze seed resis")
parser.add_argument( "-init_seq", type=str, default='', help="Path to initial sequence for each tag.")
parser.add_argument( "-freeze_hotspot_resis", type=int, default=0, help="Freeze hotspot resis, default 0")
parser.add_argument( "-run_predictor", type=int, default=0, help='By default (0), will go directly to relax. If 1, will send first to predictor' )
parser.add_argument( "-run_mpnn_sc_predictor", type=int, default=0, help='By default (0), will go directly to relax. If 1, will send first to predictor' )
parser.add_argument( "-analysis_only", type=int, default=0, help="Setting to 1 will only analyze the input complex")
parser.add_argument( "-run_relax", type=int, default=0, help='By default (0), will not relax. If 1, will relax sequence.' )
parser.add_argument( "-relax_input", type=int, default=0, help='By default (0), will not relax input. If 1, will relax input sequence.' )
parser.add_argument( "-fast_design", type=int, default=0, help='By default (0), will not run FD. If 1, will generate PSSM from MPNN seqs and run FD.' )
parser.add_argument( "-prefilter_eq_file", type=str, default = None, help="Sigmoid equation for prefiltering")
parser.add_argument( "-prefilter_mle_cut_file", type=str, default = None, help="MLE cutoff for prefiltering")
parser.add_argument( "-hbond_energy_cut", type=float, default=-0.5,help="hbond energy cutoff")
parser.add_argument( "-bb_phos_cutoff", type=int, default=0, help="# of required backbone phosphate contacts to continue to design. Default 0.")
parser.add_argument( "-net_charge_cutoff", type=int, default=1000, help="Maximum net charge to continue to design. Default 0.")
parser.add_argument( "-preempt_bb_on_charge", type=int, default=0, help="Preempt MPNN design of backbone by net charge of trial run. Default 0.")
parser.add_argument( "-compactness_prefilter", type=int, default=0, help="# of required backbone phosphate contacts to continue to design. Default 0.")
parser.add_argument( "-require_motif_in_rec_helix", type=int, default = 0, help="Default 0. If 1, will require motif in recognition helix.")
parser.add_argument( "-require_rifres_in_rec_helix", type=int, default = 0, help="Default 0. If 1, will require motif in recognition helix.")
parser.add_argument( "-n_per_silent", type=int, default=0, help="Number of random tags to design per silent. 0 for all tags.")
parser.add_argument( "-start_num", type=int, default=0, help="Numbering of first design. Default = 0.")
parser.add_argument( "-out_path", type=str, default='out.csv', help='path to write dataframe')
parser.add_argument( "-task_id", type=int, default=1, help='Slurm Array Task ID')
parser.add_argument( "-n_chunks", type=int, default=1, help='Number of chunks to divide up the job')
parser.add_argument( "-ignore_checkpointing", default=False, action='store_true', help='Use this flag if you want to force run on structures even if they show up in the checkpoint file')

args = parser.parse_args( sys.argv[1:] )
silent = args.__getattribute__("silent")
freeze_hbond_resis = args.__getattribute__("freeze_hbond_resis")

pack_no_design, softish_min, hard_min, fast_relax, ddg_filter, cms_filter, vbuns_filter, sbuns_filter, net_charge_filter, net_charge_over_sasa_filter = xml_loader.fast_relax(protocols,f'{prog_dir}/flags_and_weights/RM8B_torsional.wts',paths_dict['psi_pred_exe'],f'{prog_dir}/flags_and_weights/no_ref.rosettacon2018.beta_nov16_constrained.txt')

silent_out = args.silent_output_path_prefix + f'_{args.task_id}.silent'
sfxn = core.scoring.ScoreFunctionFactory.create_score_function(f'{prog_dir}/flags_and_weights/RM8B_torsional.wts')


def calculate_net_charge(sequence):
    # Define the charge of each amino acid
    amino_acid_charges = {
        'A': 0,  # Alanine (uncharged)
        'R': 1,  # Arginine (positive)
        'N': 0,  # Asparagine (uncharged)
        'D': -1,  # Aspartic acid (negative)
        'C': 0,  # Cysteine (uncharged)
        'E': -1,  # Glutamic acid (negative)
        'Q': 0,  # Glutamine (uncharged)
        'G': 0,  # Glycine (uncharged)
        'H': 0,  # Histidine (positive)
        'I': 0,  # Isoleucine (uncharged)
        'L': 0,  # Leucine (uncharged)
        'K': 1,  # Lysine (positive)
        'M': 0,  # Methionine (uncharged)
        'F': 0,  # Phenylalanine (uncharged)
        'P': 0,  # Proline (uncharged)
        'S': 0,  # Serine (uncharged)
        'T': 0,  # Threonine (uncharged)
        'W': 0,  # Tryptophan (uncharged)
        'Y': 0,  # Tyrosine (uncharged)
        'V': 0,  # Valine (uncharged)
    }

    # Initialize net charge to 0
    net_charge = 0

    # Calculate the net charge for the sequence
    for aa in sequence:
        net_charge += amino_acid_charges.get(aa, 0)

    return net_charge

def aln2pssm(aln_arr, pssm_out, tmp_dirname):
    """
    Input
        aln_f: Path to alignment file

    Output
        pssm_out: output PSSM
    """

    # Load alignment and convert to something useful
    curr_seq = aln_arr[0,:]
    pssm = msa2pssm(aln_arr, tmp_dirname)
    native_seq = ''.join(curr_seq.astype(str)).replace('-','')
    ctrl_seq = ''.join([l.split()[1] for l in pssm])

    # Check that the sequence from the combined PSSM
    # matches with the query sequence. (it must)
    assert native_seq == ctrl_seq

    # Finally save the pssm
    save_pssm(pssm, pssm_out)

def msa2pssm(msa, tmp_dirname):

    # write the msa
    msa_f = f'{tmp_dirname}/msa.fasta'
    write_charArr_2_fasta(msa, msa_f)

    # Write the query
    query_f = f'{tmp_dirname}/query.fasta'
    seq_no_gap = msa[0][msa[0]!=b'-']
    write_charArr_2_fasta([seq_no_gap], f'{tmp_dirname}/query.fasta')

    # Get the PSSM
    pssm_path = f'{tmp_dirname}/pssm.fasta'
    tmp_out = f'{tmp_dirname}/tmp.out'
    tmp_err = f'{tmp_dirname}/tmp.out'
    psiblast_cmd = f'{psiblast_exe} -subject {query_f} -in_msa {msa_f} -ignore_msa_master -out_ascii_pssm {pssm_path} > {tmp_out} 2>{tmp_err}'
    subprocess.run(psiblast_cmd, shell=True)
    #os.system(psiblast_cmd)

    pssm = read_pssm(pssm_path)

    return pssm


def save_pssm(pssm_vector_lines, out_f):
    out_str = ''
    out_str += '\n'
    out_str += 'Last position-specific scoring matrix computed, weighted observed percentages rounded down, information per position, and relative weight of gapless real matches to pseudocounts\n'
    out_str += '            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V\n'
    for l in pssm_vector_lines:
        out_str += l
    out_str += '\n'
    out_str += '                      K         Lambda\n'
    out_str += 'PSI Ungapped         0.1946     0.3179\n'
    out_str += 'PSI Gapped           0.0596     0.2670\n'
    with open(out_f,'w') as f_out:
        f_out.write(out_str)


def write_charArr_2_fasta(charArr, filename):
    seqs = [''.join(s.astype(str)) for s in charArr]
    with open(filename, 'w') as f_out:
        for i,seq in enumerate(seqs):
            f_out.write(f'>s_{i}\n')
            f_out.write(f'{seq}\n')

def read_pssm(filepath):
    datalines = []
    with open(filepath, 'r') as f_open:
        for line in f_open:
            is_pssm_line = (len(line.split()) == 44)
            if is_pssm_line:
                datalines.append(line)
    assert len(datalines) > 4 # if this fails, check that the format of the PSSM is as expected (pssm vectors when split should be 44 long)
    return datalines

#################################
# Function Definitions
#################################

# PDB Parse Util Functions

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
  # ["ARND"] -> [[0,1,2,3]]
  x = np.array(x);
  if x.ndim == 0: x = x[None]
  return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

# End PDB Parse Util Functions

# I/O Functions

def add2silent( pose, tag, sfd_out, silent_path):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, silent_path)

# End I/O Functions

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string

def thread_mpnn_seq( pose, binder_seq ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    for resi, mut_to in enumerate( binder_seq ):
        resi += 1 # 1 indexing
        name3 = aa_1_3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )

    return pose

def generate_seqopt_features(pdbstring, extra_res_param={}):
    # There's a lot of extraneous info in here for the ligand MPNN--didn't want to delete in case it breaks anything.
    return pdb_dict_list

def get_seq_from_pdb( pdb_fn, slash_for_chainbreaks ):
    to1letter = {
      "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
      "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
      "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
      "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

    seq = ''
    with open(pdb_fn) as fp:
      for line in fp:
        if line.startswith("TER"):
          if not slash_for_chainbreaks: continue
          seq += "/"
        if not line.startswith("ATOM"):
          continue
        if line[12:16].strip() != "CA":
          continue
        resName = line[17:20]
        #
        seq += to1letter[resName]
    return my_rstrip( seq, '/' )

def sequence_optimize(pdbstring, chains, fixed_positions_dict, num_seqs, pssm_dict=None ):

    t0 = time.time()

    ## get mpnn args
    argparser = run.get_argument_parser()
    mpnn_args = argparser.parse_args([])
    mpnn_args.model_type="ligand_mpnn"
    mpnn_args.seed=111
    mpnn_args.temperature=args.temperature
    mpnn_args.pdb_input_as_string=pdbstring
    mpnn_args.pdb_string_name='temptag'
    mpnn_args.batch_size=min( 1, args.seq_per_struct ) 
    mpnn_args.number_of_batches=args.seq_per_struct // mpnn_args.batch_size
    mpnn_args.ligand_mpnn_use_side_chain_context=1 
    mpnn_args.pack_side_chains=args.mpnn_sidechain_packing 
    mpnn_args.fixed_residues=fixed_positions_dict
    mpnn_args.checkpoint_ligand_mpnn=paths_dict['mpnn_model_checkpoint']
    mpnn_args.checkpoint_path_sc=paths_dict['sc_checkpoint_path']
    mpnn_args.return_output_no_save=1 
    return_dict = run.main(mpnn_args,paths_dict)
    designed_dict = return_dict["temptag"]["designs"]
    
    # Currently sequences, scores, and pdbs as string as output
    sequences_etc = [ ( designed_dict[i]['sequence'],designed_dict[i]['overall_confidence'],designed_dict[i]['backbone_PDB_string'] ) for i in designed_dict]

    print( f"MPNN generated {len(sequences_etc)} sequences in {int( time.time() - t0 )} seconds" )

    return sequences_etc

def get_final_dict(score_dict, string_dict):
    print(score_dict)
    final_dict = OrderedDict()
    keys_score = [] if score_dict is None else list(score_dict)
    keys_string = [] if string_dict is None else list(string_dict)

    all_keys = keys_score + keys_string

    argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

    for idx in argsort:
        key = all_keys[idx]

        if ( idx < len(keys_score) ):
            final_dict[key] = "%8.3f"%(score_dict[key])
        else:
            final_dict[key] = string_dict[key]

    return final_dict

def add2scorefile(tag, scorefilename, score_dict=None):

    write_header = not os.path.isfile(scorefilename)
    with open(scorefilename, "a") as f:
        add_to_score_file_open(tag, f, write_header, score_dict)

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
    final_dict = get_final_dict( score_dict, string_dict )
    if ( write_header ):
        f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))
    scores_string = " ".join(final_dict.values())
    f.write("SCORE:     %s        %s\n"%(scores_string, tag))

def generate_mut_string(seq):
    return 'MUT:' + '_'.join( [ f"{idx+1}.{aa_1_3[aa1]}" for idx, aa1 in enumerate(seq) ] )

def swap_mut_string(tag, mut_string, og_struct):
    outlines = []
    for line in og_struct:
        line = line.strip()
        if not 'PDBinfo-LABEL:' in line:
            # Swap out tags on all lines except for the remark line
            splits = line.split()
            if len(splits) == 0:
                outlines.append( '' )
                continue
            outline = splits[:-1]
            outline.append(tag)
            outlines.append( ' '.join(outline) )
            continue

        splits = line.split(' ')
        outsplits = []
        mut_found = False

        for split in splits:
            if not split.startswith('MUT:'):
                outsplits.append(split)
                continue
            mut_found = True
            outsplits.append(mut_string)

        if not mut_found: outsplits.append(mut_string)

        outlines.append( ' '.join(outsplits) )

    return '\n'.join(outlines)

# def add2silent( pose, tag, sfd_out ):
#     struct = sfd_out.create_SilentStructOP()
#     struct.fill_struct( pose, tag )
#     sfd_out.add_structure( struct )
#     sfd_out.write_silent_struct( struct, "out.silent" )

def delete_file_list( filelist ):
    for f in filelist: os.remove( f )

def pdbs2poses( pdbs ):
    return [ ( pose_from_pdb( pdb ), my_rstrip( pdb, ".pdb" ) ) for pdb in pdbs ]

def get_chains( pose ):
    lengths = [ p.size() for p in pose.split_by_chain() ]
    endA = pose.split_by_chain()[1].size()
    endB = endA + pose.split_by_chain()[2].size()

    chains = [ pose.pdb_info().chain( i ) for i in [ endA, endB ] ]

    return chains

def count_backbone_phosphate_contacts(pose, tag, df_scores, args) :
    '''
    Takes in a pose and returns the count of amino acid residues making backbone hydrogen bonds with DNA phosphate atoms.
    '''
    pose_hb = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose)
    pose_hb = pose.get_hbonds()
    backbone_phosphate_contacts = []
    for hbond in range(1,pose_hb.nhbonds()+1):
        hbond = pose_hb.hbond(hbond)
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        donor_hatm = hbond.don_hatm()
        acceptor_atm = hbond.acc_atm()
        don_atom_type = pose.residue(donor_res).atom_name(donor_hatm).strip(" ")
        acc_atom_type = pose.residue(acceptor_res).atom_name(acceptor_atm).strip(" ")
        if acc_atom_type in ['OP1','OP2',"O5'"] and don_atom_type == 'H':
            backbone_phosphate_contacts.append(donor_res)
    df_scores['n_backbone_phosphate_contacts'] = [len(backbone_phosphate_contacts)]
    print(f"n_phosphate_contacts: {len(backbone_phosphate_contacts)}")
    return df_scores

def count_base_interactions(pose, tag, df_scores, args, hydrophobic_distances = [4,5,6]) :
    '''
    Takes in a pose and returns the number of C and T bases with hydrophobic contacts.
    '''
    pose_hb = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose)
    pose_hb = pose.get_hbonds()
    DNA_base_names = ['ADE','GUA','THY','CYT', '5IU', 'BRU', 'RGU', 'RCY', 'RAD', 'URA']
    DNA_bases = ['ADE','GUA','THY','CYT']
    DNA_names3 = ['DA','DG','DT','DC']

    # aa_index = []
    dna_index = []

    dna_atoms_w_hbonds = []

    for hbond_num in range (1,pose_hb.nhbonds()+1):
        hbond = str(pose_hb.hbond(hbond_num))

        ## Exclude anything that has a backbone contact or that doesn't have both a protein and a dna interaction
        if ('dna' in hbond and 'protein' in hbond) and 'backbone' not in hbond: pass
        else: continue

        hbond_str = hbond.split('acc:')
        if 'dna' in hbond_str[0]: hbond_str = hbond_str[0].strip()
        else: hbond_str = hbond_str[1].strip()

        dna_hbond_parts = hbond_str.split(' ')

        dna_atoms_w_hbonds.append((dna_hbond_parts[1], dna_hbond_parts[2]))

    # The set part guarantees we don't double-count multiple hbonds to one atom--one satisfaction of an atom is sufficient
    base_specific_hbond_by_atom_count = len(set(dna_atoms_w_hbonds))

    all_protein_C_atoms = []

    # Gather all hydrophobic carbon atom coordinates in protein; gather DNA residue indices
    for residue in range(1,pose.total_residue()+1):
        current_residue = pose.residue(residue)
        if current_residue.is_protein() and current_residue.name1() != 'Z':
            # aa_index.append(residue)
            for i, aa_atom in enumerate(current_residue.atoms()):
                atom_name_rep = current_residue.atom_name(i+1).strip()
                if (current_residue.name3() in ['GLN','GLU'] and atom_name_rep == 'CD') or \
                   (current_residue.name3() in ['ASN','ASP'] and atom_name_rep == 'CG') or \
                   (current_residue.name3() in ['ARG'] and atom_name_rep == 'CZ'):
                    continue # Didn't seem fair to include clearly not hydrophobic carbon atoms.
                elif ('H' in atom_name_rep) or current_residue.atom_is_backbone(i+1) or ("C" not in atom_name_rep):
                    continue
                all_protein_C_atoms.append(list(aa_atom.xyz()))
        elif (current_residue.is_DNA() or current_residue.name3().strip() in DNA_names3) and current_residue.name1() not in 'XZ':
            # very fun--sometimes something has a 'DG' but isn't listed as DNA...see CRE for an example of a terminal DG
            ## very fun part 2: the !Z part is important because there are some modified DNA bases in the natives, and you don't want to count any of them because MPNN won't have predicted them
            dna_index.append(residue)

    all_protein_C_atoms = np.array(all_protein_C_atoms)

    hydrophobic_dict = dict.fromkeys(hydrophobic_distances,0)

    for dna_residue in dna_index:
        dna_resi_distances = []
        dna_resi = pose.residue(dna_residue)
        if pose.residue(dna_residue).name3().strip() in ['DA','DG']: continue
        for i, dna_atom in enumerate(dna_resi.atoms()):
            if 'H' in dna_resi.atom_name(i+1):
                continue
            if dna_resi.name3().strip() in ['DT','DC','C','U'] and dna_resi.atom_name(i+1).strip() in ['C4', 'C5', 'C6', 'C7']:
                dna_resi_distances.append(np.sqrt(np.sum((np.array(dna_atom.xyz())-all_protein_C_atoms)**2,axis=1)))
        shortest_hydrophobic_distances = np.min(np.array(dna_resi_distances),axis=0)
        for k in hydrophobic_distances:
            hydrophobic_dict[k] += np.sum(shortest_hydrophobic_distances < k)
    print('Number of C within hydrophobic distances:', hydrophobic_dict)

    for k in hydrophobic_distances:
        df_scores[f'n_hydrophobic_contacts_{k}A'] = [hydrophobic_dict[k]]
    df_scores['n_base_hbonds'] = [base_specific_hbond_by_atom_count]

    return df_scores

def count_hbonds_protein_dna(pose, energy_cutoff) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making hydrogen bonds with DNA.
    '''
    mc_bb_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "OP2",  "OP1",  "O5'",  "C5'",  "C4'",  "O4'",
               "C3'",  "O3'",  "C2'",  "C1'", "H5''",  "H5'",  "H4'",  "H3'", "H2''",  "H2'",  "H1'"]
    aa_bb_atoms = ['N', 'CA', 'C', 'O', 'CB', '1H', '2H', '3H', 'HA', 'OXT','H'] #I added this to avoid counting backbone - DNA hydrogen bonds
    DNA_base_names = ['ADE','GUA','THY','CYT', '5IU', 'BRU', 'RGU', 'RCY', 'RAD', 'RTH']

    pose_hb = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose)
    pose_hb = pose.get_hbonds()
    hbond_id = []
    hbond_dna = []
    hbonds_don_hatm = []
    hbonds_acc_atm = []
    hbonds_don_res = []
    hbonds_acc_res = []
    involves_dna = []
    hbonds_energy = []

    base_dict = {}
    base_count_dict = {}
    total_residues=pose.total_residue()
    dna_res_num = 1
    for residue in range(1, total_residues+1):
        if pose.residue(residue).name().split(':')[0] in DNA_base_names:
            base_dict[pose.residue(residue).name().split(':')[0]+'{0}'.format(dna_res_num)] = residue
            base_count_dict[pose.residue(residue).name().split(':')[0]+'{0}'.format(dna_res_num)] = 0
            dna_res_num +=1
    dna_base_list = list(base_dict.keys())
    dna_res_list = list(base_dict.values())


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

    protein_dna = 0
    protein_dna_specific = 0
    aa_identities_base = []
    aa_pos_base = []
    aa_identities_phosphate = []
    aa_pos_phosphate = []
    bidentate_list = []

    for residue in range(len(hbonds_acc_res)) :
        if hbonds_energy[residue] > float(energy_cutoff):
            continue

        don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
        acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")

        if pose.residue(hbonds_don_res[residue]).name().split(':')[0] in DNA_base_names :
            if not pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                protein_dna += 1
                if not don_atom_type in mc_bb_atoms and not acc_atom_type in aa_bb_atoms:
                    protein_dna_specific += 1
                    aa_identities_base.append(pose.residue(hbonds_acc_res[residue]).name1())
                    if not int(hbonds_acc_res[residue]) in aa_pos_base:
                        aa_pos_base.append(int(hbonds_acc_res[residue]))
                    else:
                        bidentate_list.append(int(hbonds_acc_res[residue]))
                    base_id = dna_base_list[dna_res_list.index(hbonds_don_res[residue])]
                    base_count_dict[base_id] += 1
                elif don_atom_type in mc_bb_atoms and not acc_atom_type in aa_bb_atoms:
                    aa_identities_phosphate.append(pose.residue(hbonds_acc_res[residue]).name1())
                    if not int(hbonds_acc_res[residue]) in aa_pos_phosphate:
                        aa_pos_phosphate.append(int(hbonds_acc_res[residue]))
        else :
            if pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                protein_dna += 1
                if not acc_atom_type in mc_bb_atoms and not don_atom_type in aa_bb_atoms:
                    protein_dna_specific += 1
                    aa_identities_base.append(pose.residue(hbonds_don_res[residue]).name1())
                    if not int(hbonds_don_res[residue]) in aa_pos_base:
                        aa_pos_base.append(int(hbonds_don_res[residue]))
                    else:
                        bidentate_list.append(int(hbonds_don_res[residue]))
                    base_id = dna_base_list[dna_res_list.index(hbonds_acc_res[residue])]
                    base_count_dict[base_id] += 1
                elif acc_atom_type in mc_bb_atoms and not don_atom_type in aa_bb_atoms:
                    aa_identities_phosphate.append(pose.residue(hbonds_acc_res[residue]).name1())
                    if not int(hbonds_don_res[residue]) in aa_pos_phosphate:
                        aa_pos_phosphate.append(int(hbonds_don_res[residue]))
                        
    return  aa_pos_base, aa_pos_phosphate, list(set(bidentate_list))

def prefilter_preemption(df_scores, prefilter_eq_file, prefilter_mle_cut_file):
    # ---------------------------------------------------------------------
    #              Preemption based on calibrated prefilters
    # ---------------------------------------------------------------------
    def get_first_line(f):
       with open(f,'r') as f_open:
           eq = [l for l in f_open][0]
       return eq

    eq = get_first_line(prefilter_eq_file)[1:-2]
    possible_features = df_scores.columns
    for f in possible_features:
        eq = eq.replace(f' {f} ', f' {df_scores[f].iloc[0]} ') # KEEP THE SPACES HERE!!

    EXP = np.exp
    try:
        log_prob = np.log10(-eval(eq)) # WHY is the equation being dumped with a - in front of?!
    except: log_prob = -100
    log_prob_cut = float(get_first_line(prefilter_mle_cut_file))

    # Store relevant information
    df_scores['log_prob_cut'] = log_prob_cut
    df_scores['log_prob'] = log_prob

    if log_prob<log_prob_cut:
        passed_prefilter = False
        print(f'Did not pass prefilters. Quitting')
    else:
        passed_prefilter = True
        print(f"Passed prefilters. Continuing")
    return passed_prefilter, df_scores

def hbond_score(pose, tag, df_scores, args):
    # Weights for hbond scoring
    base_hbond_score_weights = {
                     'ARG_g_hbonds':5,'LYS_g_hbonds':3,'ASP_g_hbonds':-3,'GLU_g_hbonds':-3,'ASN_g_hbonds':1, # g hbonds
                     'GLN_g_hbonds':2,'SER_g_hbonds':0,'THR_g_hbonds':-3,'TYR_g_hbonds':-3,'HIS_g_hbonds':1, # g hbonds

                     'ARG_c_hbonds':-3,'LYS_c_hbonds':0,'ASP_c_hbonds':7,'GLU_c_hbonds':4,'ASN_c_hbonds':3, # c hbonds
                     'GLN_c_hbonds':0,'SER_c_hbonds':0,'THR_c_hbonds':0,'TYR_c_hbonds':-3,'HIS_c_hbonds':0, # c hbonds

                     'ARG_a_hbonds':-3,'LYS_a_hbonds':-3,'ASP_a_hbonds':-3,'GLU_a_hbonds':-3,'ASN_a_hbonds':5, # a hbonds
                     'GLN_a_hbonds':5,'SER_a_hbonds':0,'THR_a_hbonds':-3,'TYR_a_hbonds':0,'HIS_a_hbonds':0, # a hbonds

                     'ARG_t_hbonds':1,'LYS_t_hbonds':1,'ASP_t_hbonds':-3,'GLU_t_hbonds':-3,'ASN_t_hbonds':1, # t hbonds
                     'GLN_t_hbonds':1,'SER_t_hbonds':-3,'THR_t_hbonds':-3,'TYR_t_hbonds':-3,'HIS_t_hbonds':-3, # t hbonds
                    }
    phosphate_hbond_score_weights = {
                     'ARG_phosphate_hbonds':7,'LYS_phosphate_hbonds':0,'GLN_phosphate_hbonds':10,'TYR_phosphate_hbonds':4,'ASN_phosphate_hbonds':1,
                     'SER_phosphate_hbonds':1,'THR_phosphate_hbonds':1,'HIS_phosphate_hbonds':1,
                    }
    bidentate_score_weights = {'ARG_g_bidentates':1,'ASN_a_bidentates':1,'GLN_a_bidentates':1}

    columns, result = count_hbond_types.count_hbonds_protein_dna(pose, tag, args.hbond_energy_cut)
    df_new = pd.Series(result, index = columns)
    df_scores['base_score'] = 0
    df_scores['unweighted_base_score'] = 0
    df_scores['phosphate_score'] = 0
    df_scores['unweighted_phosphate_score'] = 0
    df_scores['bidentate_score'] = 0
    for j in base_hbond_score_weights:
        df_scores[j] = df_new[j]
        df_scores['unweighted_base_score'] += df_new[j]
        df_scores['base_score'] += df_new[j]*base_hbond_score_weights[j]
    for j in phosphate_hbond_score_weights:
        df_scores[j] = df_new[j]
        df_scores['unweighted_phosphate_score'] += df_new[j]
        df_scores['phosphate_score'] += df_new[j]*phosphate_hbond_score_weights[j]
    for j in bidentate_score_weights:
        df_scores['bidentate_score'] += df_new[j]*bidentate_score_weights[j]

    return df_scores

def calc_bidentates(aa_dictionary, dna_1, dna_2):
    '''
    In case we decide to calculate a max_rboltz_bidentate.

    Takes in a dictionary of hydrogen bonds by amino acid as well as a list of the residues for each of the two DNA strands.
    '''
    num_bidentates = 0
    num_bridged_bidentates = 0
    num_cross_strand_bidentates = 0

    for aa in aa_dictionary.keys():
        if len(aa_dictionary[aa]) > 1:
#             print(f'{aa} makes a bidentate bond!')
            num_bidentates += 1
            unique_bp_list = list(set(aa_dictionary[aa]))
            if len(unique_bp_list) > 1:
                num_bridged_bidentates += 1
                for count, bp in enumerate(unique_bp_list):
                    for other_bp in unique_bp_list[count+1:]:
#                         print(f'{aa} is a bridged bidentate to {bp} and {other_bp}')
                        if (int(bp[3:]) in dna_1 and int(other_bp[3:]) in dna_2) or (int(bp[3:]) in dna_2 and int(other_bp[3:]) in dna_1):
                            num_cross_strand_bidentates += 1
    return num_bidentates, num_bridged_bidentates, num_cross_strand_bidentates

def calc_rboltz(pose, df):
    '''
    Takes in a pose and the existing DataFrame of scores for a design and returns the DataFrame with
    three new columns: largest RotamerBoltzmann of ARG/LYS/GLU/GLN residues; average of the top two
    RotamerBoltzmann scores (includes every amino acid type); and median RotamerBoltzmann (includes every amino acid type)
    '''
    notable_aas = ['ARG','GLU','GLN','LYS']
    base_hbond_residues, phosphate_hbond_residues, bidentates = count_hbonds_protein_dna(pose, args.hbond_energy_cut)

    cols = ['residue_num','residue_name','rboltz']
    design_df = pd.DataFrame(columns=cols)

    def get_rboltz_vals(test_resis,repack_neighbors=False):
        for j in test_resis:
            residue_info = [j, pose.residue(j).name()[:3]]

            hbond_position_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
            hbond_position_selector.set_index(j)

            # Set up task operations for RotamerBoltzmann...with standard settings
            tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
            tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
            tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
            tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

            # Allow extra rotamers
            extra_rots = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
            extra_rots.ex1(1)
            extra_rots.ex2(1)
            tf.push_back(extra_rots)

            # Prevent repacking on everything but the hbond position and neighbors
            prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
            if repack_neighbors == True:
                neighbor_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
                neighbor_selector.set_distance(12)
                neighbor_selector.set_focus_selector(hbond_position_selector)
                not_pack = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(neighbor_selector)
            else:
                not_pack = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(hbond_position_selector)
            prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, not_pack)
            tf.push_back(prevent_subset_repacking)

            sfxn = pyrosetta.rosetta.core.scoring.get_score_function()

            rboltz = pyrosetta.rosetta.protocols.calc_taskop_filters.RotamerBoltzmannWeight()
            rboltz.scorefxn(sfxn)
            rboltz.task_factory(tf)
            rboltz.skip_ala_scan(1)
            rboltz.no_modified_ddG(1)
            rboltz_val = rboltz.compute(pose)
            residue_info.append(rboltz_val)
            design_df.loc[len(design_df)] = residue_info.copy()
        return design_df
    
    # get base specific rboltz_metrics
    design_df = get_rboltz_vals(base_hbond_residues)
    RKQE_subset = design_df[design_df['residue_name'].isin(notable_aas)]
    if len(RKQE_subset) > 0:
        df['max_rboltz_RKQE'] = -1 * RKQE_subset['rboltz'].min()
    else:
        df['max_rboltz_RKQE'] = 0

    if len(design_df) > 0:
        df['avg_top_two_rboltz'] = -1 * np.average(design_df['rboltz'].nsmallest(2))
        df['median_rboltz'] = -1 * np.median(design_df['rboltz'])
    else:
        df['avg_top_two_rboltz'] = 0
        df['median_rboltz'] = 0
        
    # get phosphate specific rboltz metrics
    design_df = get_rboltz_vals(phosphate_hbond_residues)
    RKQE_subset = design_df[design_df['residue_name'].isin(notable_aas)]
    if len(RKQE_subset) > 0:
        df['max_rboltz_RKQE_phosphate'] = -1 * RKQE_subset['rboltz'].min()
    else:
        df['max_rboltz_RKQE_phosphate'] = 0

    if len(design_df) > 0:
        df['avg_top_two_rboltz_phosphate'] = -1 * np.average(design_df['rboltz'].nsmallest(2))
        df['median_rboltz_phosphate'] = -1 * np.median(design_df['rboltz'])
    else:
        df['avg_top_two_rboltz_phosphate'] = 0
        df['median_rboltz_phosphate'] = 0

    return df

def eval_model(pose, df_scores, tag, silent_in, silent_out, mpnn_score = None, is_prefilter = False, is_mpnn_prefilter = False):
    
    try: contact_molecular_surface = cms_filter.compute( pose )
    except: contact_molecular_surface = 1
    net_charge = net_charge_filter.compute( pose )
    net_charge_over_sasa = net_charge_over_sasa_filter.compute( pose )

    df_scores = pd.DataFrame()
    
    df_scores['contact_molecular_surface'] = [contact_molecular_surface]
    df_scores['net_charge'] = [net_charge]
    df_scores['net_charge_over_sasa'] = [net_charge_over_sasa]
    df_scores['tag'] = [tag]
    df_scores['silent_in'] = [silent]
    df_scores['silent_out'] = [silent_out]
    df_scores['sequence'] = [pose.sequence()]
    df_scores['mpnn_score'] = [mpnn_score]
    df_scores['is_prefilter'] = is_prefilter
    
    df_scores = hbond_score(pose, tag, df_scores, args)
    df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)
    
    if not is_mpnn_prefilter:
        df_scores['ddg'] = [ddg_filter.compute( pose )]
        df_scores['ddg_over_cms'] = df_scores['ddg'] / df_scores['contact_molecular_surface']
    
    # Calculate RotamerBoltzmann scores
    if not is_prefilter:
        df_scores = calc_rboltz(pose, df_scores)
        
    return df_scores

def dl_design( pose, tag, og_struct, sfd_out, dfs, silent, silent_out, pdb_folder):

    # this is to fix weird tags after motif motif_graft
    #tag = tag.replace('m','m_pdb')

    design_counter = args.start_num

    # Tags can't have .pdb in their name otherwise silentextract will stop reading after that ending.
    if args.silent_tag_prefix != '':
        prefix = f"{args.silent_tag_prefix}_{tag.replace('.pdb','')}_dldesign"
    else: prefix = f"{tag.replace('.pdb','')}_dldesign"

     # prefilter designs 
    if args.require_motif_in_rec_helix != 0:
        turn_resis = []
        motif_resis = []
        info = pose.pdb_info()
        for res in range(1,pose.total_residue()+1):
            reslabel = info.get_reslabels(res)
            if 'TURN' in reslabel or 'ss_L3' in reslabel:
                turn_resis.append(res)
            if 'MOTIF' in reslabel:
                motif_resis.append(res)
        motif_in_rec_helix = False
        if len(turn_resis) > 0 and len(motif_resis) > 0:
            for j in range(6):
                if (min(motif_resis) - j in turn_resis):
                    motif_in_rec_helix = True
        df_scores = pd.DataFrame()
        df_scores['motif_in_rec_helix'] = [motif_in_rec_helix]

        if motif_in_rec_helix == 0:
            print('Motif not in recognition helix. Quitting')
            return design_counter, dfs
        print('Motif is in recognition helix')

    if args.require_rifres_in_rec_helix != 0:
        RH_resis = []
        rif_resis = []
        info = pose.pdb_info()
        for res in range(1,pose.total_residue()+1):
            reslabel = info.get_reslabels(res)
            if 'RH' in reslabel or 'ss_RH' in reslabel:
                RH_resis.append(res)
            if 'RIFRES' in reslabel:
                rif_resis.append(res)

        rifres_in_rec_helix = False
        for res in rif_resis:
            if res in RH_resis:
                rifres_in_rec_helix = True

        df_scores = pd.DataFrame()
        df_scores['rifres_in_rec_helix'] = [rifres_in_rec_helix]

        if rifres_in_rec_helix == False:
            print('RIFRES not in recognition helix. Quitting')
            return design_counter, dfs
        print('RIFRES is in recognition helix')

    
    if args.bb_phos_cutoff != 0:
        df_scores = pd.DataFrame()
        df_scores['n_backbone_phosphate_contacts'] = [0]
        df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)
        if df_scores['n_backbone_phosphate_contacts'].iloc[0] < args.bb_phos_cutoff:
            print('Did not exceed bb-phos cutoff. Continuing to next tag.')
            return design_counter, dfs
        
    if args.compactness_prefilter != 0:
        min_ncontacts,max_loop_length = compactness_filter.filter(pose)
        if min_ncontacts < 2 or max_loop_length > 5:
            print('Did not pass compactness or loop length filter. Continuing to next tag.')
            return design_counter, dfs
        else:
            print(f"Input model has a minimum of {min_ncontacts} contacts per ss element and maximum loop length of {max_loop_length}")


    # selecting residues to freeze
    ### get inputs for frozen residues
    ### This whole section assumes you're designing exactly one protein to design and is in the first position of the chains,
    ### but regardless of the actual file the chain is labeled by MPNN as chain A because it's the first chain 
    if freeze_hbond_resis == 1:
        try: 
            hbond_residues, phosphate_hbond_residues, bidentates = count_hbonds_protein_dna(pose, args.hbond_energy_cut)
        except: # in some cases there are no hbonds and this causes an error. In this case, do not fix hbond residues.
            hbond_residues = []
        print(f"Fixed resis: {hbond_residues}")
        fixed_positions_dict={'temptag':{"A":hbond_residues}}
    elif args.freeze_resis != '':
        freeze_resis = args.freeze_resis.split(',')
        fixed_positions = [int(j) for j in freeze_resis]
        print(f'Fixed positions: {",".join(freeze_resis)}')
        fixed_positions_dict = {'temptag':{"A":fixed_positions}}
    elif args.freeze_hotspot_resis == 1:
        hotspot_resis = []
        info = pose.pdb_info()
        for res in range(1,pose.total_residue()+1):
            reslabel = info.get_reslabels(res)
            if 'HOTSPOT' in reslabel:
                hotspot_resis.append(res)
        hotspot_str = ",".join([str(j) for j in hotspot_resis])
        print(f'Fixed positions: {hotspot_str}')
        fixed_positions_dict = {'temptag':{"A":hotspot_resis}}
    elif args.freeze_seed_resis == 1:
        if args.init_seq != '':
            with open(args.init_seq,'r') as init_seqs_f:
                for line in init_seqs_f.readlines():
                    if tag in line:
                        init_seq = line.split(' ')[-1].rstrip()
        else:
            init_seq = pose.split_by_chain()[1].sequence()  # this is another place that it assumes protein is chain 1
        print(init_seq)
        hotspot_resis = [j+1 for j, resi in enumerate(init_seq) if resi != 'A']
        hotspot_str = ",".join([str(j) for j in hotspot_resis])
        print(f'Fixed positions: {hotspot_str}')
        fixed_positions_dict = {'temptag':{"A":hotspot_resis}}
    else: fixed_positions_dict = {'temptag':{"A":[]}}

    print(fixed_positions_dict)

    ### for fused_mpnn, the inputs are going to be the same thing you'd put in on the command line -- so it's "A16 A17 A24" etc.
    fixed_positions_str = ''
    for chain_w_fixed_residues in fixed_positions_dict['temptag'].keys():
        for residue in fixed_positions_dict['temptag'][chain_w_fixed_residues]:
            fixed_positions_str = f"{fixed_positions_str} {chain_w_fixed_residues}{residue}"
    pssm_dict=None
    
    in_pose = pose.clone()

    if args.relax_input == 1 or args.analysis_only == 1:
        print("Relaxing input sequence")
        relaxed_pose = in_pose.clone()
        if args.relax_input ==1:
            tag = f"{tag}_relaxed_input_0"
            t0 = time.time()
            fast_relax.apply(relaxed_pose)

        df_scores = pd.DataFrame()
        df_scores = eval_model(relaxed_pose,df_scores,tag,silent,silent_out)
        
        print(f"ddg of input model: {df_scores['ddg'].iloc[0]}")

        # Add setting info to the score file
        for key in args.__dict__:
            df_scores[key] = args.__dict__[key]
        dfs.append(df_scores)
        
        add2silent( relaxed_pose, tag, sfd_out, silent_out )
    
    if args.analysis_only == 1: 
        return design_counter, dfs

    
    ### This is where we get the string for the pdb file
    # pose.dump_pdb( pdbfile )
    converter = PoseToStructFileRepConverter()
    converter.init_from_pose(pose)
    pdbstring = create_pdb_contents_from_sfr(converter.sfr())
    

    chains = get_chains( pose )
    
    if args.preempt_bb_on_charge != 0:
        seqs_scores_pdbs = sequence_optimize( pdbstring, chains, fixed_positions_str, args.seq_per_struct, pssm_dict )
        pass_cut = False
        for idx, seq_score_pdb in enumerate( seqs_scores_pdbs ):
            seq, mpnn_score, pdb = seq_score_pdb
            net_charge = calculate_net_charge( seq )
            if net_charge < args.preempt_bb_on_charge:
                pass_cut = True
        if pass_cut == False: 
            print('Backbone did not pass net charge pre-emption. Continuing to next tag.')
            return design_counter, dfs
    
    ### Get PDB string
    seqs_scores_pdbs = sequence_optimize( pdbstring, chains, fixed_positions_str, args.seq_per_struct, pssm_dict )

    
    for idx, seq_score_pdb in enumerate( seqs_scores_pdbs ):
        print(seq_score_pdb[0],seq_score_pdb[1])
        tag = f"{prefix}_{design_counter}"
        seq, mpnn_score, pdb = seq_score_pdb
        #pose = thread_mpnn_seq( in_pose, seq )
        
        if args.net_charge_cutoff != 0:
            net_charge = calculate_net_charge( seq )
            print('Net charge:', net_charge)
            if net_charge > args.net_charge_cutoff: 
                print('Design does not pass net charge cutoff')
                continue
            
        pose = Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, pdb)
        
        
        if args.run_mpnn_sc_predictor == 1:
            t0 = time.time()
            
            df_scores = eval_model(pose,pd.DataFrame(),tag,silent,silent_out, mpnn_score=mpnn_score, is_prefilter=True, is_mpnn_prefilter=True)
            #print(f"MPNN-packed ddg is {df_scores['ddg'].iloc[0]}")
            print(f"MPNN-packed hbond_score is {df_scores['base_score'].iloc[0]}")
            print(f"MPNN-packed phosphate_score  is {df_scores['phosphate_score'].iloc[0]}")

            prefilter_eq_file = args.prefilter_eq_file
            prefilter_mle_cut_file = args.prefilter_mle_cut_file

            # Add setting info to the score file
            for key in args.__dict__:
                df_scores[key] = args.__dict__[key]
            dfs.append(df_scores)

            if prefilter_eq_file is not None and prefilter_mle_cut_file is not None:
                passed_prefilters, df_scores = prefilter_preemption(df_scores, prefilter_eq_file, prefilter_mle_cut_file)
                if passed_prefilters == False:
                    t1 = time.time()
                    print(f'prefiltering took {t1-t0}')
                    continue

            t1 = time.time()
            print(f'prefiltering took {t1-t0}')
            
        if args.run_predictor == 1:
            t0 = time.time()
            packed_pose = pose.clone()
            pack_no_design.apply( packed_pose )
            softish_min.apply( packed_pose )
            hard_min.apply( packed_pose )
            
            df_scores = eval_model(packed_pose,pd.DataFrame(),tag,silent,silent_out, mpnn_score=mpnn_score, is_prefilter=True)
            print(f"Prefilter ddg is {df_scores['ddg'].iloc[0]}")

            prefilter_eq_file = args.prefilter_eq_file
            prefilter_mle_cut_file = args.prefilter_mle_cut_file

            # Add setting info to the score file
            for key in args.__dict__:
                df_scores[key] = args.__dict__[key]
            dfs.append(df_scores)

            if prefilter_eq_file is not None and prefilter_mle_cut_file is not None:
                passed_prefilters, df_scores = prefilter_preemption(df_scores, prefilter_eq_file, prefilter_mle_cut_file)
                if passed_prefilters == False:
                    t1 = time.time()
                    print(f'prefiltering took {t1-t0}')
                    continue

            t1 = time.time()
            print(f'prefiltering took {t1-t0}')


        if args.run_relax == 1:
            t0 = time.time()
            fast_relax.apply(pose)
            df_scores = eval_model(pose,pd.DataFrame(),tag,silent,silent_out,mpnn_score=mpnn_score)

            print(f"Relax ddg is {df_scores['ddg'].iloc[0]}")
        

            # Add setting info to the score file
            for key in args.__dict__:
                df_scores[key] = args.__dict__[key]
            dfs.append(df_scores)


            t1 = time.time()
            print(f'relax took {t1-t0}\n')

        if args.run_relax == 0 and args.run_predictor == 0 and args.run_mpnn_sc_predictor == 0:
            df_scores = pd.DataFrame()

            df_scores['tag'] = [tag]
            df_scores['silent_in'] = [silent]
            df_scores['silent_out'] = [silent_out]
            df_scores['sequence'] = seq
            df_scores['mpnn_score'] = [mpnn_score]
            df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)

            # Add setting info to the score file
            for key in args.__dict__:
                df_scores[key] = args.__dict__[key]
            dfs.append(df_scores)
        

        # pose.dump_pdb(pdb_folder + '_mpnnredesign' + '/' + tag + '.pdb')
        add2silent( pose, tag, sfd_out, silent_out )
        design_counter += 1

    if args.fast_design == 1:
        t0 = time.time()
        tmpdir = tempfile.TemporaryDirectory()
        tmp_dirname = tmpdir.name
        

        # Generate a fasta file
        with open (f'{tmp_dirname}/{prefix}.fasta', 'w') as f:
            for idx,seq_score in enumerate( seqs_scores ):
                tag = f"{prefix}_{idx}"
                seq, mpnn_score = seq_score
                f.write('>'+tag+'\n'+seq+'\n')
        aln = AlignIO.read(f'{tmp_dirname}/{prefix}.fasta','fasta')
        print(aln)
        # Convert the alignment to some useful format
        aln_arr = np.array([list(rec) for rec in aln], np.character)
        pssmFile = f'{tmp_dirname}/{prefix}.pssm'
        print("Making PSSM")
        aln2pssm(aln_arr, pssmFile, tmp_dirname)

        task_relax, add_ca_csts, FSP, FastDesign, rm_csts = xml_loader.fast_design_interface(protocols, f'{prog_dir}/flags_and_weights/RM8B_torsional.wts', pssmFile, fixed_positions_dict, f'{prog_dir}/flags_and_weights/no_ref.rosettacon2018.beta_nov16_constrained.txt')

        tag = f"{prefix}_FD"
        seq, mpnn_score = seqs_scores[0]
        pose = thread_mpnn_seq( in_pose, seq )

        add_ca_csts.apply(pose)
        FSP.apply(pose)
        FastDesign.apply(pose)
        rm_csts.apply(pose)

        fast_relax.apply(pose)
        df_scores = eval_model(relaxed_pose,df_scores,tag,silent,silent_out)
        print(f"FD ddg is {df_scores['ddg'].iloc[0]}")
        

        add2silent( pose, tag, sfd_out, silent_out )
        design_counter += 1
        # Add setting info to the score file
        for key in args.__dict__:
            df_scores[key] = args.__dict__[key]
        dfs.append(df_scores)


        t1 = time.time()
        print(f'fast design took {t1-t0}\n')

    return design_counter, dfs

def main( pdb, silent_structure, sfd_in, sfd_out, dfs, silent, silent_out ):

    t0 = time.time()
    print( "Attempting pose: %s"%pdb )

    # Load pose
    if args.silent != '':
        pose = Pose()
        sfd_in.get_structure( pdb ).fill_pose( pose )
    elif args.pdb_folder != '':
        pose = pose_from_pdb(silent_structure)
    elif args.pdb != '':
        pose = pose_from_pdb(silent_structure)

    good_designs, dfs = dl_design( pose, pdb, silent_structure, sfd_out, dfs, silent, silent_out, args.pdb_folder )

    seconds = int(time.time() - t0)

    print( f"protocols.jd2.JobDistributor: {pdb} reported success. {good_designs} design(s) generated in {seconds} seconds" )

    return dfs

# Checkpointing Functions

def record_checkpoint( pdb, checkpoint_filename ):
    with open( checkpoint_filename, 'a' ) as f:
        f.write( pdb )
        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
    done_set = set()
    if not os.path.isfile( checkpoint_filename ): return done_set

    with open( checkpoint_filename, 'r' ) as f:
        for line in f:
            done_set.add( line.strip() )

    return done_set

# End Checkpointing Functions

#################################
# Begin Main Loop
#################################


if args.silent != '':
    silent_index = silent_tools.get_silent_index(silent)
    sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
    sfd_in.read_file(silent)
    tags = list(sfd_in.tags())
elif args.pdb_folder != '':
    tags = glob.glob(args.pdb_folder+'/*.pdb')
    silent = ''
elif args.pdb != '':
    tags = [args.pdb]

# if not os.path.exists(args.pdb_folder + '_mpnnredesign'):
#     os.makedirs(args.pdb_folder + '_mpnnredesign', exist_ok=True)

if args.silent != '':
    if not os.path.isfile(silent_out):
        with open(silent_out, 'w') as f: f.write(silent_tools.silent_header(silent_index))

sfd_out = core.io.silent.SilentFileData(f"out_{args.task_id}.silent", False, False, "binary", core.io.silent.SilentFileOptions())

checkpoint_filename = "check.point"
debug = True

finished_structs = determine_finished_structs( checkpoint_filename )
#mpnn_model, sc_model = init_seq_optimize_model()


if args.silent_tag != '' and args.silent_tag in tags:
    tags = [args.silent_tag]
    print(tags)

if args.silent_tag_file != '':
    new_tags = []
    with open(args.silent_tag_file, 'r') as f_in:
        input_tags = f_in.readlines()
        for tag in input_tags:
            tag = tag.rstrip()
            print(tag)
            if tag in tags:
                print('Tag exists in silent file.')
                new_tags.append(tag)
            else:
                print('Tag does not exist in silent file.')
    tags = list(new_tags)
    print(f'Designing {len(tags)} specified tags from {args.silent}')

if args.n_per_silent != 0:
    tags = random.choices(tags, k=args.n_per_silent)
    print(f'Designing {len(tags)} tags from {args.silent}')

job_index = args.task_id
chunk_size = len(tags) // args.n_chunks

start_index = (job_index - 1) * chunk_size
end_index = start_index + chunk_size


if job_index == args.n_chunks:
    chunk = tags[start_index:]
else:
    chunk = tags[start_index:end_index]

for pdb in chunk:
    if args.pdb_folder != '':
        pdb_path = pdb
        print(f'Attempting pdb: {pdb_path}:')
        pdb = pdb.split('/')[-1].replace('.pdb','')
    total_time_0 = time.time()

    if not args.ignore_checkpointing and pdb in finished_structs: continue

    dfs = []

    if debug:
        if args.silent != '':
            silent_structure = silent_tools.get_silent_structure( silent, silent_index, pdb )
        elif args.pdb_folder != '':
            silent_structure = pdb_path
            sfd_in = ''
        elif args.pdb != '':
            silent_structure = pdb
            sfd_in = ''
        dfs = main( pdb, silent_structure, sfd_in, sfd_out, dfs, silent, silent_out )

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try:
            if args.silent != '':
                silent_structure = silent_tools.get_silent_structure( silent, silent_index, pdb )
            elif args.pdb_folder != '':
                silent_structure = pdb_path
                sfd_in = ''
            elif args.pdb != '':
                silent_structure = pdb
                sfd_in = ''
            dfs = main( pdb, silent_structure, sfd_in, sfd_out, dfs, silent, silent_out)

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )

    # We are done with one pdb, record that we finished
    if args.start_num != '':
        record_checkpoint( pdb, checkpoint_filename )

    try: scores = pd.concat(dfs, axis=0, ignore_index=True)
    except: continue

    out_csv = args.out_path

    if os.path.isfile(out_csv):
        csv_done = pd.read_csv(out_csv,index_col=0)
        scores = pd.concat([csv_done,scores], axis=0, ignore_index=True)
    scores.to_csv(out_csv)

    total_time_1 = time.time()

    print(f"Total time was {total_time_1 - total_time_0} seconds")
