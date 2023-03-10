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

# import silent tools
sys.path.insert( 0, paths_dict['silent_tools_path'])
import silent_tools

# Build a PyRosetta with Python3.9
# Will it work?
sys.path.insert( 0, paths_dict['pyrosetta_path'] )

from pyrosetta import *
from pyrosetta.rosetta import *

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
import count_hbond_types

# import util for loading rosetta movers
sys.path.append(prog_dir)
import xml_loader

# import custom MPNN utilities
sys.path.append(prog_dir +  '/design/' )
import generate_sequences_s2s_chain as mpnn_util

# initiate pyrosetta
init( "-mute all -beta_nov16 -in:file:silent_struct_type binary" +
    " -holes:dalphaball /home/norn/software/DAlpahBall/DAlphaBall.gcc" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" +
    f"@{prog_dir}/flags_and_weights/RM8B.flags" )


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

# for connections=64 use this checkpoint: /projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m3_random_loss_bias_var/checkpoints/epoch32_step347928.pt

parser = argparse.ArgumentParser()
parser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run.' )
parser.add_argument( "-pdb_folder", type=str, default="", help='Folder if pdbs.' )
parser.add_argument( "-silent_tag", type=str, default = '', help="Input silent tag")
parser.add_argument( "-silent_tag_file", type=str, default = '', help="File containing tags to design in silent")
parser.add_argument( "-silent_tag_prefix", type=str, default = '', help="Prefix for silent tag outputs")
parser.add_argument( "-seq_per_struct", type=int, default=10, help='Number of MPNN sequences that will be produced' )
parser.add_argument( "-checkpoint_path", type=str, default=f'{paths_dict["mpnn_model_checkpoint"]}' )
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
parser.add_argument( "-freeze_hotspot_resis", type=int, default=0, help="Freeze hotspot resis, default 0")
parser.add_argument( "-run_predictor", type=int, default=0, help='By default (0), will go directly to relax. If 1, will send first to predictor' )
parser.add_argument( "-run_relax", type=int, default=0, help='By default (0), will not relax. If 1, will relax sequence.' )
parser.add_argument( "-relax_input", type=int, default=0, help='By default (0), will not relax input. If 1, will relax input sequence.' )
parser.add_argument( "-fast_design", type=int, default=0, help='By default (0), will not run FD. If 1, will generate PSSM from MPNN seqs and run FD.' )
parser.add_argument( "-prefilter_eq_file", type=str, default = None, help="Sigmoid equation for prefiltering")
parser.add_argument( "-prefilter_mle_cut_file", type=str, default = None, help="MLE cutoff for prefiltering")
parser.add_argument( "-hbond_energy_cut", type=float, default=-0.5,help="hbond energy cutoff")
parser.add_argument( "-bb_phos_cutoff", type=int, default=0, help="# of required backbone phosphate contacts to continue to design. Default 0.")
parser.add_argument( "-require_motif_in_rec_helix", type=int, default = 0, help="Default 0. If 1, will require motif in recognition helix.")
parser.add_argument( "-require_rifres_in_rec_helix", type=int, default = 0, help="Default 0. If 1, will require motif in recognition helix.")
parser.add_argument( "-n_per_silent", type=int, default=0, help="Number of random tags to design per silent. 0 for all tags.")
parser.add_argument( "-start_num", type=int, default=0, help="Numbering of first design. Default = 0.")


args = parser.parse_args( sys.argv[1:] )
silent = args.__getattribute__("silent")
freeze_hbond_resis = args.__getattribute__("freeze_hbond_resis")

pack_no_design, softish_min, hard_min, fast_relax, ddg_filter, cms_filter, vbuns_filter, sbuns_filter, net_charge_filter, net_charge_over_sasa_filter = xml_loader.fast_relax(protocols,f'{prog_dir}/flags_and_weights/RM8B_torsional.wts',paths_dict['psi_pred_exe'])

silent_out = f"{os.getcwd()}/out.silent"

sfxn = core.scoring.ScoreFunctionFactory.create_score_function(f'{prog_dir}/flags_and_weights/RM8B_torsional.wts')

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

def add2silent( pose, tag, sfd_out ):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, "out.silent" )

# End I/O Functions

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string

def init_seq_optimize_model():

    hidden_feat_dim = 128
    num_layers = 3

    model = mpnn_util.ProteinMPNN(num_letters=21, node_features=hidden_feat_dim, edge_features=hidden_feat_dim, hidden_dim=hidden_feat_dim, num_encoder_layers=num_layers,num_decoder_layers=num_layers, augment_eps=args.augment_eps, k_neighbors=args.num_connections)
    model.to(mpnn_util.device)
    print('Number of model parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(mpnn_util.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None,lig_params=[]):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-atcgdryuJ")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP',
              ' DA', ' DT', ' DC', ' DG',  '  A',  '  U',  '  C',  '  G',
             'LIG']

  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

  lig_to_atms = {}
  if len(lig_params) > 0:
    for lig_param in lig_params:
        lig_to_atms.update(parse_extra_res_fa_param(lig_param))
  lig_names = list(lig_to_atms.keys())

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

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    #Ligands will start with HETATM but for noncanonial stuff (may start with ATOM ?)? GRL
    #Currently one chain should be just the ligand itself.
    parse_atm_line = False
    if len(lig_names) > 0 and line[17:20] in lig_names:
        parse_atm_line = True
    if line[:4] == "ATOM":
        parse_atm_line = True

    if (parse_atm_line):
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha():
            resa,resn = resn[-1],int(resn[:-1])-1
        else:
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn:
            min_resn = resn
        if resn > max_resn:
            max_resn = resn
        if resn not in xyz:
            xyz[resn] = {}
        if resa not in xyz[resn]:
            xyz[resn][resa] = {}
        if resn not in seq:
            seq[resn] = {}
        #for alternative coords
        if resa not in seq[resn]:
            seq[resn][resa] = resi
        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_,atype_ = [],[],[]
  is_lig_chain = False
  try:
      resn_to_ligName = {}
      for resn in range( min_resn,max_resn+1):
        if resn in seq:
          #20: not in the list, treat as gap
          for k in sorted(seq[resn]):
              resi = seq[resn][k]
              if resi in lig_names:
                  resn_to_ligName[resn] = resi
                  is_lig_chain = True
                  seq_.append(aa_3_N.get('LIG',29)) ###GRL:hard-coding 29 ok?
              else:
                  seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        #
        #
        if is_lig_chain:
            #Get new atoms list just for the ligand as defined in the params file
            atoms = list(lig_to_atms[resn_to_ligName[resn]].keys())
        #
        #Ligand atoms in the same order with xyz_ (matching atom name -> atype as defined in the params file)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
        #
        if is_lig_chain:
            lig_atm_d = lig_to_atms[resn_to_ligName[resn]]
            for atom in atoms:
                atype_.append(lig_atm_d[atom])
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_)), np.array(atype_)
  except TypeError:
      return 'no_chain', 'no_chain', 'no_chain'

def thread_mpnn_seq( pose, binder_seq ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    for resi, mut_to in enumerate( binder_seq ):
        resi += 1 # 1 indexing
        name3 = aa_1_3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )

    return pose

def generate_seqopt_features( path_to_pdb, extra_res_param={}):
    # There's a lot of extraneous info in here for the ligand MPNN--didn't want to delete in case it breaks anything.

    ref_atype_to_element = {'CNH2': 'C', 'COO': 'C', 'CH0': 'C', 'CH1': 'C', 'CH2': 'C', 'CH3': 'C', 'aroC': 'C', 'Ntrp': 'N', 'Nhis': 'N', 'NtrR': 'N', 'NH2O': 'N', 'Nlys': 'N', 'Narg': 'N', 'Npro': 'N', 'OH': 'O', 'OW': 'O', 'ONH2': 'O', 'OOC': 'O', 'Oaro': 'O', 'Oet2': 'O', 'Oet3': 'O', 'S': 'S', 'SH1': 'S', 'Nbb': 'N', 'CAbb': 'C', 'CObb': 'C', 'OCbb': 'O', 'Phos': 'P', 'Pbb': 'P', 'Hpol': 'H', 'HS': 'H', 'Hapo': 'H', 'Haro': 'H', 'HNbb': 'H', 'Hwat': 'H', 'Owat': 'O', 'Opoint': 'O', 'HOH': 'O', 'Bsp2': 'B', 'F': 'F', 'Cl': 'CL', 'Br': 'BR', 'I': 'I', 'Zn2p': 'ZN', 'Co2p': 'CO', 'Cu2p': 'CU', 'Fe2p': 'FE', 'Fe3p': 'FE', 'Mg2p': 'MG', 'Ca2p': 'CA', 'Pha': 'P', 'OPha': 'O', 'OHha': 'O', 'Hha': 'H', 'CO3': 'C', 'OC3': 'O', 'Si': 'Si', 'OSi': 'O', 'Oice': 'O', 'Hice': 'H', 'Na1p': 'NA', 'K1p': 'K', 'He': 'HE', 'Li': 'LI', 'Be': 'BE', 'Ne': 'NE', 'Al': 'AL', 'Ar': 'AR', 'Sc': 'SC', 'Ti': 'TI', 'V': 'V', 'Cr': 'CR', 'Mn': 'MN', 'Ni': 'NI', 'Ga': 'GA', 'Ge': 'GE', 'As': 'AS', 'Se': 'SE', 'Kr': 'KR', 'Rb': 'RB', 'Sr': 'SR', 'Y': 'Y', 'Zr': 'ZR', 'Nb': 'NB', 'Mo': 'MO', 'Tc': 'TC', 'Ru': 'RU', 'Rh': 'RH', 'Pd': 'PD', 'Ag': 'AG', 'Cd': 'CD', 'In': 'IN', 'Sn': 'SN', 'Sb': 'SB', 'Te': 'TE', 'Xe': 'XE', 'Cs': 'CS', 'Ba': 'BA', 'La': 'LA', 'Ce': 'CE', 'Pr': 'PR', 'Nd': 'ND', 'Pm': 'PM', 'Sm': 'SM', 'Eu': 'EU', 'Gd': 'GD', 'Tb': 'TB', 'Dy': 'DY', 'Ho': 'HO', 'Er': 'ER', 'Tm': 'TM', 'Yb': 'YB', 'Lu': 'LU', 'Hf': 'HF', 'Ta': 'TA', 'W': 'W', 'Re': 'RE', 'Os': 'OS', 'Ir': 'IR', 'Pt': 'PT', 'Au': 'AU', 'Hg': 'HG', 'Tl': 'TL', 'Pb': 'PB', 'Bi': 'BI', 'Po': 'PO', 'At': 'AT', 'Rn': 'RN', 'Fr': 'FR', 'Ra': 'RA', 'Ac': 'AC', 'Th': 'TH', 'Pa': 'PA', 'U': 'U', 'Np': 'NP', 'Pu': 'PU', 'Am': 'AM', 'Cm': 'CM', 'Bk': 'BK', 'Cf': 'CF', 'Es': 'ES', 'Fm': 'FM', 'Md': 'MD', 'No': 'NO', 'Lr': 'LR', 'SUCK': 'Z', 'REPL': 'Z', 'REPLS': 'Z', 'HREPS': 'Z', 'VIRT': 'X', 'MPct': 'X', 'MPnm': 'X', 'MPdp': 'X', 'MPtk': 'X'}
    chem_elements = ['C','N','O','P','S','AC','AG','AL','AM','AR','AS','AT','AU','B','BA','BE','BI','BK','BR','CA','CD','CE','CF','CL','CM','CO','CR','CS','CU','DY','ER','ES','EU','F','FE','FM','FR','GA','GD','GE','H','HE','HF','HG','HO','I','IN','IR','K','KR','LA','LI','LR','LU','MD','MG','MN','MO','NA','NB','ND','NE','NI','NO','NP','OS','PA','PB','PD','PM','PO','PR','PT','PU','RA','RB','RE','RH','RN','RU','SB','SC','SE','SM','SN','SR','Si','TA','TB','TC','TE','TH','TI','TL','TM','U','V','W','X','XE','Y','YB','Z','ZN','ZR']
    ref_atypes_dict = dict(zip(chem_elements, range(len(chem_elements))))



    pdb_dict_list = []
    c = 0
    dna_list = 'atcg'
    rna_list = 'dryu'
    protein_list = 'ARNDCQEGHILKMFPSTWYVX'
    protein_list_check = 'ARNDCQEGHILKMFPSTWYV'
    k_DNA = 10
    ligand_dumm_list = 'J'

    dna_rna_dict = {
    "a" : ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8','N9', "", ""],
    "t" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "C5", "C6", "C7", "", "", ""],
    "c" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", "", ""],
    "g" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
    "d" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
    "r" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "", ""],
    "y" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", ""],
    "u" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "X" : 22*[""]}

    dna_rna_atom_types = np.array(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7", ""])

    idxAA_22_to_27 = np.zeros((9, 22), np.int32)
    for i, AA in enumerate(dna_rna_dict.keys()):
        for j, atom in enumerate(dna_rna_dict[AA]):
            idxAA_22_to_27[i,j] = int(np.argwhere(atom==dna_rna_atom_types)[0][0])
    ### \end This was just a big copy-paste from the training script ###

    atoms = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        'CZ3', 'NZ']  # These are the 36 atom types mentioned in Justas's script

    all_atom_types = atoms + list(dna_rna_atom_types)
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet# + extra_alphabet

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_seq_DNA = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        visible_list = []
        chain_list = []
        Cb_list = []
        P_list = []
        dna_atom_list = []
        dna_atom_mask_list = []
        #
        ligand_atom_list = []
        ligand_atype_list = []
        ligand_total_length = 0
        #
        #Check if ligand params file is given
        lig_params = []
        if biounit in list(extra_res_param.keys()):
            lig_params = extra_res_param[biounit]

        for letter in chain_alphabet:
#                 print(f'started parsing {letter}')
            xyz, seq, atype = parse_PDB_biounits(biounit, atoms = all_atom_types, chain=letter, lig_params=lig_params)
            # print("chain_seq", seq)
#             print(f'finished parsing {letter}')
            if type(xyz) != str:
                protein_seq_flag = any([(item in seq[0]) for item in protein_list_check])
                dna_seq_flag = any([(item in seq[0]) for item in dna_list])
                rna_seq_flag = any([(item in seq[0]) for item in rna_list])
                lig_seq_flag = any([(item in seq[0]) for item in ligand_dumm_list])
#                     print(protein_seq_flag, dna_seq_flag, rna_seq_flag, lig_seq_flag)

                if protein_seq_flag: xyz, seq, atype = parse_PDB_biounits(biounit, atoms = atoms, chain=letter)
                elif (dna_seq_flag or rna_seq_flag): xyz, seq, atype = parse_PDB_biounits(biounit, atoms = list(dna_rna_atom_types), chain=letter)
                elif (lig_seq_flag): xyz,seq, atype = parse_PDB_biounits(biounit, atoms=[], chain=letter, lig_params=lig_params)

                if protein_seq_flag:
                    my_dict['seq_chain_'+letter]=seq[0]
                    concat_seq += seq[0]
                    chain_list.append(letter)
                    all_atoms = np.array(xyz) #[L, 14, 3] # deleted res index on xyz--I think this was useful when there were batches of structures at once?
                    b = all_atoms[:,1] - all_atoms[:,0]
                    c = all_atoms[:,2] - all_atoms[:,1]
                    a = np.cross(b, c, -1)
                    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + all_atoms[:,1] #virtual
                    Cb_list.append(Cb)
                    coords_dict_chain = {}
                    coords_dict_chain['all_atoms_chain_'+letter]=xyz.tolist()
                    my_dict['coords_chain_'+letter]=coords_dict_chain
                elif dna_seq_flag or rna_seq_flag: # This section is important for moving from 22-atom representation to the 27-atom representation...unless it's already in 27 format??
                    
                    seq_ = ''.join(list(np.array(list(seq))[0,])) # Edited this section on 1/10/23 to delete gaps when residues are not numbered sequentially
                    all_atoms = np.array(xyz)
                    if len(seq_.replace('-','')) == len(all_atoms): # could be expanded to include things besides '-' gaps
                        pass
                    else:
                        all_atoms = np.array([all_atoms[i] for i, elem in enumerate(seq_) if elem in list(dna_rna_dict.keys())]) # if elem in != '-'
                        seq_ = ''.join([seq_[i] for i, elem in enumerate(seq_) if elem in list(dna_rna_dict.keys())])
                        
                    P_list.append(all_atoms[:,0])
                    all_atoms_ones = np.ones((all_atoms.shape[0], 22)) # I believe all_atoms.shape[0] is supposed to be the length of the sequence
                    concat_seq_DNA += seq_
                    all_atoms27_mask = np.zeros((len(seq_), 27))
                    # print(seq_)
                    idx = np.array([idxAA_22_to_27[np.argwhere(AA==np.array(list(dna_rna_dict.keys())))[0][0]] for AA in seq_])
                    np.put_along_axis(all_atoms27_mask, idx, all_atoms_ones, 1)
                    dna_atom_list.append(all_atoms) # was all_atoms27, but all_atoms is already in that format!!
                    dna_atom_mask_list.append(all_atoms27_mask)
                elif lig_seq_flag:
                    temp_atype = -np.ones(len(atype))
                    for k_, ros_type in enumerate(atype):
                        if ros_type in list(ref_atype_to_element):
                            temp_atype[k_] = ref_atypes_dict[ref_atype_to_element[ros_type]]
                        else:
                            temp_atype[k_] = ref_atypes_dict['X']
                    all_atoms = np.array(xyz)
                    ligand_atype = np.array(temp_atype)
                    if (1-np.isnan(all_atoms)).sum() != 0:
                        tmp_idx = np.argwhere(1-np.isnan(all_atoms[0,].mean(-1))==1.0)[-1][0] + 1
                        ligand_atom_list.append(all_atoms[:tmp_idx,:])
                        ligand_atype_list.append(ligand_atype[:tmp_idx])
                        ligand_total_length += tmp_idx


        if len(P_list) > 0:
            Cb_stack = np.concatenate(Cb_list, 0) #[L, 3]
            P_stack = np.concatenate(P_list, 0) #[K, 3]
            dna_atom_stack = np.concatenate(dna_atom_list, 0)
            dna_atom_mask_stack = np.concatenate(dna_atom_mask_list, 0)

            D = np.sqrt(((Cb_stack[:,None,:]-P_stack[None,:,:])**2).sum(-1) + 1e-7)
            idx_dna = np.argsort(D,-1)[:,:k_DNA] #top 10 neighbors per residue
            dna_atom_selected = dna_atom_stack[idx_dna]
            dna_atom_mask_selected = dna_atom_mask_stack[idx_dna]
            my_dict['dna_context'] = dna_atom_selected[:,:,:-1,:].tolist()
            my_dict['dna_context_mask'] = dna_atom_mask_selected[:,:,:-1].tolist()
        else:
            my_dict['dna_context'] = 'no_DNA'
            my_dict['dna_context_mask'] = 'no_DNA'
        if ligand_atom_list:
            ligand_atom_stack = np.concatenate(ligand_atom_list, 0)
            ligand_atype_stack = np.concatenate(ligand_atype_list, 0)
            my_dict['ligand_context'] = ligand_atom_stack.tolist()
            my_dict['ligand_atype'] = ligand_atype_stack.tolist()
        else:
            my_dict['ligand_context'] = 'no_ligand'
            my_dict['ligand_atype'] = 'no_ligand'
        my_dict['ligand_length'] = int(ligand_total_length)
        #
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s < len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
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

def sequence_optimize( pdbfile, chains, model, fixed_positions_dict, pssm_dict=None ):

    t0 = time.time()

    # sequence = get_seq_from_pdb( pdbfile, False )

    feature_dict = generate_seqopt_features( pdbfile )

    arg_dict = mpnn_util.set_default_args( args.seq_per_struct, decoding_order='random' )
    arg_dict['temperature'] = args.temperature

    # in a world where the binder is first and is the only chain to redesign, this is fine
    masked_chains = [ chains[0] ]
    visible_chains = []

    sequences = mpnn_util.generate_sequences( model, feature_dict, arg_dict, masked_chains, visible_chains, args, fixed_positions_dict=fixed_positions_dict, pssm_dict=pssm_dict )

    print( f"MPNN generated {len(sequences)} sequences in {int( time.time() - t0 )} seconds" )

    return sequences

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

def add2silent( pose, tag, sfd_out ):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, "out.silent" )

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

def count_hbonds_protein_dna(pose) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making hydrogen bonds with DNA.
    '''
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
        hbonds_don_hatm.append(donor_hatm)
        hbonds_acc_atm.append(acceptor_atm)
        hbonds_don_res.append(donor_res)
        hbonds_acc_res.append(acceptor_res)

    aa_pos = []


    for residue in range(len(hbonds_acc_res)) :
        don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
        acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")

        if acc_atom_type in ['OP1','OP2']:
            continue

        if pose.residue(hbonds_don_res[residue]).name().split(':')[0] in DNA_base_names :
            if not pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                if not int(hbonds_acc_res[residue]) in aa_pos:
                    aa_pos.append(int(hbonds_acc_res[residue]))
        else :
            if pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                if not int(hbonds_don_res[residue]) in aa_pos:
                    aa_pos.append(int(hbonds_don_res[residue]))
    return aa_pos

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
    hbond_energy_cut = float(args.hbond_energy_cut)
    columns, result = count_hbond_types.count_hbonds_protein_dna(pose, tag, hbond_energy_cut)
    df_new = pd.Series(result, index = columns)
    df_scores['base_score'] = 0
    df_scores['phosphate_score'] = 0
    df_scores['bidentate_score'] = 0
    for j in base_hbond_score_weights:
        df_scores['base_score'] += df_new[j]*base_hbond_score_weights[j]
    for j in phosphate_hbond_score_weights:
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
            num_bidentates += 1
            unique_bp_list = list(set(aa_dictionary[aa]))
            if len(unique_bp_list) > 1:
                num_bridged_bidentates += 1
                for count, bp in enumerate(unique_bp_list):
                    for other_bp in unique_bp_list[count+1:]:
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
    hbond_residues = count_hbonds_protein_dna(pose)

    cols = ['residue_num','residue_name','rboltz']
    design_df = pd.DataFrame(columns=cols)

    for j in hbond_residues:
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

        # Prevent repacking on everything but the hbond position
        prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
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

    return df

def dl_design( pose, tag, og_struct, mpnn_model, sfd_out, dfs, silent, silent_out ):

    # this is to fix weird tags after motif motif_graft
    #tag = tag.replace('m','m_pdb')

    design_counter = args.start_num

    if args.silent_tag_prefix != '':
        prefix = f"{args.silent_tag_prefix}_{tag}_dldesign"
    else: prefix = f"{tag}_dldesign"

    pdbfile = f"tmp.pdb"

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

    if freeze_hbond_resis == 1:
        hbond_residues = count_hbonds_protein_dna(pose)
        print(f"Fixed resis: {hbond_residues}")
        fixed_positions_dict={"tmp":{"A":hbond_residues}}
    elif args.freeze_resis != '':
        freeze_resis = args.freeze_resis.split(',')
        fixed_positions = [int(j) for j in freeze_resis]
        print(f'Fixed positions: {",".join(freeze_resis)}')
        fixed_positions_dict = {"tmp":{"A":fixed_positions}}
    elif args.freeze_hotspot_resis == 1:
        hotspot_resis = []
        info = pose.pdb_info()
        for res in range(1,pose.total_residue()+1):
            reslabel = info.get_reslabels(res)
            if 'HOTSPOT' in reslabel:
                hotspot_resis.append(res)
        hotspot_str = ",".join([str(j) for j in hotspot_resis])
        print(f'Fixed positions: {hotspot_str}')
        fixed_positions_dict = {"tmp":{"A":hotspot_resis}}
    else: fixed_positions_dict = {"tmp":{"A":[]}}
    pssm_dict=None

    pose.dump_pdb( pdbfile )
    chains = get_chains( pose )
    seqs_scores = sequence_optimize( pdbfile, chains, mpnn_model, fixed_positions_dict, pssm_dict )
    print(seqs_scores)

    os.remove( pdbfile )

    in_pose = pose.clone()

    if args.relax_input == 1:
        print("Relaxing input sequence")
        relaxed_pose = in_pose.clone()
        tag = f"{tag}_input_0"
        t0 = time.time()
        fast_relax.apply(pose)
        ddg = ddg_filter.compute( pose )
        try: contact_molecular_surface = cms_filter.compute( pose )
        except: contact_molecular_surface = 1
        vbuns = vbuns_filter.apply( pose )            
        sbuns = sbuns_filter.apply( pose )
        net_charge = net_charge_filter.compute( pose )
        net_charge_over_sasa = net_charge_over_sasa_filter.compute( pose )

        df_scores = pd.DataFrame()
        df_scores['ddg'] = [ddg]
        df_scores['contact_molecular_surface'] = [contact_molecular_surface]
        df_scores['ddg_over_cms'] = df_scores['ddg'] / df_scores['contact_molecular_surface']
        df_scores['vbuns5.0_heavy_ball_1.1D'] = [vbuns]
        df_scores['sbuns5.0_heavy_ball_1.1D'] = [sbuns]
        df_scores['net_charge'] = [net_charge]
        df_scores['net_charge_over_sasa'] = [net_charge_over_sasa]
        df_scores['tag'] = [tag]
        df_scores['silent_in'] = [silent]
        df_scores['silent_out'] = [silent_out]
        df_scores['sequence'] = [pose.sequence()]
        df_scores['mpnn_score'] = []
        df_scores['is_prefilter'] = [False]


        print(f"Relax ddg is {ddg}")
        # Score hbonds to DNA
        df_scores = hbond_score(pose, tag, df_scores, args)
        df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)


        # Calculate RotamerBoltzmann scores
        df_scores = calc_rboltz(pose, df_scores)


        # Add setting info to the score file
        for key in args.__dict__:
            df_scores[key] = args.__dict__[key]
        dfs.append(df_scores)

    for idx, seq_score in enumerate( seqs_scores ):
        tag = f"{prefix}_{design_counter}"
        seq, mpnn_score = seq_score
        pose = thread_mpnn_seq( in_pose, seq )

        if args.run_predictor == 1:
            t0 = time.time()
            packed_pose = in_pose.clone()
            pack_no_design.apply( packed_pose )
            softish_min.apply( packed_pose )
            hard_min.apply( packed_pose )
            ddg = ddg_filter.compute( packed_pose )
            try:
                contact_molecular_surface = cms_filter.compute( packed_pose )
            except: contact_molecular_surface = 1
            df_scores = pd.DataFrame()

            df_scores['ddg'] = [ddg]
            df_scores['contact_molecular_surface'] = [contact_molecular_surface]
            df_scores['ddg_over_cms'] = df_scores['ddg'] / df_scores['contact_molecular_surface']
            df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)
            df_scores['tag'] = [tag]
            df_scores['silent_in'] = [silent]
            df_scores['silent_out'] = [silent_out]
            df_scores['sequence'] = seq
            df_scores['mpnn_score'] = mpnn_score
            df_scores['is_prefilter'] = [True]
            print(f"Prefilter ddg is {ddg}")

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
            ddg = ddg_filter.compute( pose )
            try: contact_molecular_surface = cms_filter.compute( pose )
            except: contact_molecular_surface = 1
            vbuns = vbuns_filter.apply( pose )
            sbuns = sbuns_filter.apply( pose )
            net_charge = net_charge_filter.compute( pose )
            net_charge_over_sasa = net_charge_over_sasa_filter.compute( pose )

            df_scores = pd.DataFrame()
            df_scores['ddg'] = [ddg]
            df_scores['contact_molecular_surface'] = [contact_molecular_surface]
            df_scores['ddg_over_cms'] = df_scores['ddg'] / df_scores['contact_molecular_surface']
            df_scores['vbuns5.0_heavy_ball_1.1D'] = [vbuns]
            df_scores['sbuns5.0_heavy_ball_1.1D'] = [sbuns]
            df_scores['net_charge'] = [net_charge]
            df_scores['net_charge_over_sasa'] = [net_charge_over_sasa]
            df_scores['tag'] = [tag]
            df_scores['silent_in'] = [silent]
            df_scores['silent_out'] = [silent_out]
            df_scores['sequence'] = seq
            df_scores['mpnn_score'] = [mpnn_score]
            df_scores['is_prefilter'] = [False]


            print(f"Relax ddg is {ddg}")
            # Score hbonds to DNA
            df_scores = hbond_score(pose, tag, df_scores, args)
            df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)


            # Calculate RotamerBoltzmann scores
            df_scores = calc_rboltz(pose, df_scores)

            if args.ddg_cutoff is not None:
                if ddg > args.ddg_cutoff:
                    df_scores['passed_ddg_cutoff'] = False
                    design_counter += 1
                    continue
                else:
                    df_scores['passed_ddg_cutoff'] = True

            # Add setting info to the score file
            for key in args.__dict__:
                df_scores[key] = args.__dict__[key]
            dfs.append(df_scores)


            t1 = time.time()
            print(f'relax took {t1-t0}\n')

        if args.run_relax == 0 and args.run_predictor == 0:
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

        add2silent( pose, tag, sfd_out )
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

        task_relax, add_ca_csts, FSP, FastDesign, rm_csts = xml_loader.fast_design_interface(protocols, f'{prog_dir}/flags_and_weights/RM8B_torsional.wts', pssmFile, fixed_positions_dict)

        tag = f"{prefix}_FD"
        seq, mpnn_score = seqs_scores[0]
        pose = thread_mpnn_seq( in_pose, seq )

        add_ca_csts.apply(pose)
        FSP.apply(pose)
        FastDesign.apply(pose)
        rm_csts.apply(pose)

        fast_relax.apply(pose)
        ddg = ddg_filter.compute( pose )
        try: contact_molecular_surface = cms_filter.compute( pose )
        except: contact_molecular_surface = 1
        vbuns = vbuns_filter.apply( pose )
        sbuns = sbuns_filter.apply( pose )
        net_charge = net_charge_filter.compute( pose )
        net_charge_over_sasa = net_charge_over_sasa_filter.compute( pose )

        df_scores = pd.DataFrame()
        df_scores['ddg'] = [ddg]
        df_scores['contact_molecular_surface'] = [contact_molecular_surface]
        df_scores['ddg_over_cms'] = df_scores['ddg'] / df_scores['contact_molecular_surface']
        df_scores['vbuns5.0_heavy_ball_1.1D'] = [vbuns]
        df_scores['sbuns5.0_heavy_ball_1.1D'] = [sbuns]
        df_scores['net_charge'] = [net_charge]
        df_scores['net_charge_over_sasa'] = [net_charge_over_sasa]
        df_scores['tag'] = [tag]
        df_scores['silent_in'] = [silent]
        df_scores['silent_out'] = [silent_out]
        df_scores['sequence'] = seq
        df_scores['mpnn_score'] = [0]
        df_scores['is_prefilter'] = [False]


        print(f"FD ddg is {ddg}")
        # Score hbonds to DNA
        df_scores = hbond_score(pose, tag, df_scores, args)
        df_scores = count_backbone_phosphate_contacts(pose, tag, df_scores, args)


        # Calculate RotamerBoltzmann scores
        df_scores = calc_rboltz(pose, df_scores)

        if args.ddg_cutoff is not None:
            if ddg > args.ddg_cutoff:
                df_scores['passed_ddg_cutoff'] = False
            else:
                df_scores['passed_ddg_cutoff'] = True

        add2silent( pose, tag, sfd_out )
        design_counter += 1
        # Add setting info to the score file
        for key in args.__dict__:
            df_scores[key] = args.__dict__[key]
        dfs.append(df_scores)


        t1 = time.time()
        print(f'fast design took {t1-t0}\n')

    return design_counter, dfs

def main( pdb, silent_structure, mpnn_model, sfd_in, sfd_out, dfs, silent, silent_out ):

    t0 = time.time()
    print( "Attempting pose: %s"%pdb )

    # Load pose
    if args.silent != '':
        pose = Pose()
        sfd_in.get_structure( pdb ).fill_pose( pose )
    elif args.pdb_folder != '':
        pose = pose_from_pdb(silent_structure)

    good_designs, dfs = dl_design( pose, pdb, silent_structure, mpnn_model, sfd_out, dfs, silent, silent_out )

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
    tags = sfd_in.tags()
elif args.pdb_folder != '':
    tags = glob.glob(args.pdb_folder+'/*.pdb')
    silent = ''

if args.silent != '':
    if not os.path.isfile(silent_out):
        with open(silent_out, 'w') as f: f.write(silent_tools.silent_header(silent_index))

sfd_out = core.io.silent.SilentFileData("out.silent", False, False, "binary", core.io.silent.SilentFileOptions())


checkpoint_filename = "check.point"
debug = True

finished_structs = determine_finished_structs( checkpoint_filename )
mpnn_model = init_seq_optimize_model()


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
    tags = new_tags
    print(f'Designing {len(tags)} specified tags from {args.silent}')

if args.n_per_silent != 0:
    tags = random.choices(tags, k=args.n_per_silent)
    print(f'Designing {len(tags)} tags from {args.silent}')

for pdb in tags:
    if args.pdb_folder != '':
        pdb_path = pdb
        print(f'Attempting pdb: {pdb_path}:')
        pdb = pdb.split('/')[-1].replace('.pdb','')
    total_time_0 = time.time()

    if pdb in finished_structs: continue

    dfs = []

    if debug:
        if args.silent != '':
            silent_structure = silent_tools.get_silent_structure( silent, silent_index, pdb )
        elif args.pdb_folder != '':
            silent_structure = pdb_path
            sfd_in = ''
        dfs = main( pdb, silent_structure, mpnn_model, sfd_in, sfd_out, dfs, silent, silent_out )

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try:
            if args.silent != '':
                silent_structure = silent_tools.get_silent_structure( silent, silent_index, pdb )
            elif args.pdb_folder != '':
                silent_structure = pdb_path
                sfd_in = ''
            dfs = main( pdb, silent_structure, mpnn_model, sfd_in, sfd_out, dfs, silent, silent_out)

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )

    # We are done with one pdb, record that we finished
    if args.start_num != '':
        record_checkpoint( pdb, checkpoint_filename )

    try: scores = pd.concat(dfs, axis=0, ignore_index=True)
    except: continue

    out_csv = 'out.csv'

    if os.path.isfile(out_csv):
        csv_done = pd.read_csv(out_csv,index_col=0)
        scores = pd.concat([csv_done,scores], axis=0, ignore_index=True)
    scores.to_csv(out_csv)

    total_time_1 = time.time()

    print(f"Total time was {total_time_1 - total_time_0} seconds")
