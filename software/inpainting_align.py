import sys,os,json
import tempfile
import numpy as np
import pandas as pd
import pandas
from optparse import OptionParser
import time
import glob
#import seaborn as sns
import string
import random
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score
#from pyrosetta.rosetta import core
import itertools
#import matplotlib.pyplot as plt
import shutil
import subprocess
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import itertools
import sklearn
import re
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neural_network
from sklearn import svm
from scipy.stats import norm
from sklearn.metrics import confusion_matrix,roc_curve,auc
from collections import defaultdict
from scipy.optimize import curve_fit
import sys

script, inpainting_dir, inpainted_pdbs = sys.argv


init()

with open('pdbs.list', 'r') as f_in:
    lines = f_in.readlines()
    pdbs = [line.rstrip() for line in lines]

print(pdbs)


def write_new_pdb(pdb,pdb_out):
    with open(pdb_out,'w') as f_out:
        with open(pdb, 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                if line.startswith('ATOM') and line[21:23].strip() == 'A':
                    f_out.write(line)
        return

for pdb in pdbs[:1]:
    pdb_name = pdb.split('/')[-1].replace('.pdb','')
    inpainted_pdbs = glob.glob(f'{inpainted_pdbs}/{pdb_name}*pdb')
    print(len(inpainted_pdbs))
    ref_pose = pose_from_pdb(pdb)
    for inpainted_pdb in inpainted_pdbs:
        
        trb_file = inpainted_pdb.replace('.pdb','.trb')
        trb = np.load(trb_file,allow_pickle=True)

        ref_align_resis = [resi[1] for resi in trb['con_ref_pdb_idx']]
        inpaint_align_resis = [resi[1] for resi in trb['con_hal_pdb_idx']]
        
        REMARK_pdb = pdb
        # Get input REMARKS
        REMARKS = {}
        with open(REMARK_pdb,'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                if line.startswith('REMARK PDBinfo-LABEL:') and 'CONTEXT' not in line:
                    REMARKS[int(line[23:26].strip())] = line
        
        
        pose = pose_from_pdb(inpainted_pdb)
        
        # align pose to reference pose
        align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
        pose_len = pose.total_residue()
        ref_pose_len = ref_pose.total_residue()

        res_map = {}

        for j, ref_resi in enumerate(ref_align_resis):
            inpaint_resi = inpaint_align_resis[j]
            ref_res = ref_pose.residue(ref_resi)
            ref_pose_atom_idx = int(ref_res.atom_index('CA'))
            inpaint_res = pose.residue(inpaint_resi)
            inpaint_pose_atom_idx = int(inpaint_res.atom_index('CA'))
            atom_id_ref_pose = pyrosetta.rosetta.core.id.AtomID(ref_pose_atom_idx,ref_resi+1)
            atom_id_inpaint_pose = pyrosetta.rosetta.core.id.AtomID(inpaint_pose_atom_idx,inpaint_resi+1)
            align_map[atom_id_inpaint_pose] = atom_id_ref_pose
        align = pyrosetta.rosetta.core.scoring.superimpose_pose(pose, ref_pose, align_map)

        new_pose = pose.clone()
        dna_pose = ref_pose.split_by_chain()[2]
        new_pose.append_pose_by_jump(dna_pose,1)
        if len(ref_pose.split_by_chain()) > 2:
            new_pose.append_pose_by_jump(ref_pose.split_by_chain()[3],1)

        outpdb = inpainted_pdb.split('/')[-1].replace('.pdb','_aligned.pdb')
        
        new_pose.dump_pdb(outpdb)
        
        # Fix chain numbering
        if len(ref_pose.split_by_chain()) > 2:
            os.system(f"sed 's/DG C/DG B/g' {outpdb} | sed 's/DC C/DC B/g' | sed 's/DA C/DA B/g' | sed 's/DT C/DT B/g' | sed '/TER/d' > tmp.pdb; mv tmp.pdb {outpdb}")
        # Append remarks to pdb
        with open(outpdb,'a') as f:
            for resi in REMARKS.keys():
                if resi in ref_align_resis:
                    index = ref_align_resis.index(resi)
                    line_pose = REMARKS[resi][:23] + format(inpaint_align_resis[index], ' 3d') + ' ' + REMARKS[resi][27:]
                    f.write(line_pose)

print('Run complete!')
