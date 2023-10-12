#!/usr/bin/env python
#GRL
import numpy as np
#from scipy import stats
from libProtein import three_aa_to_one_aa
#SVDSuperimposer This doesn't work properly  Dont use
#from Bio.SVDSuperimposer import SVDSuperimposer

class Atom:
    def __init__(self,line):
        self.raw_line = line
        self.line = line.strip()
        self.chainID = line[21]
        self.atmNo = int(line[7:11])
        self.atmName = line[11:17].strip()
        self.resNo = int(line[22:26])
        self.alt_state = line[16]
        self.resName = line[17:21].strip()
        self.oneLetter = three_aa_to_one_aa(self.resName)
        self.coord = [float(line[30:38]),\
                      float(line[38:46]),\
                      float(line[46:54])]
        #
        self.rep_resNo = '%s%d'%(self.chainID,self.resNo)
        self.template = '%s'%(line[:30])
        self.full_template = line.strip()
        return

def write_bb_lines(pdbfn,outfn):
    cont = []
    with open(pdbfn) as fp:
        for line in fp:
            if line.startswith('ATOM'):
                atm = Atom(line.strip())
                if atm.atmName in ['N','CA','C','O']:
                    cont.append(line.strip())
            else:
                cont.append(line.strip())
    fout = open(outfn,'w')
    fout.write('%s\n'%('\n'.join(cont)))
    fout.close()
    return
    
def dist_two_coords(crd_a,crd_b):
    a = np.array(crd_a)
    b = np.array(crd_b)
    return np.linalg.norm(a-b)
    
def no_align_rmsd(crd_a,crd_b):
    a = np.array(crd_a)
    n_a = len(a)
    b = np.array(crd_b)
    n_b = len(b)
    assert (n_a == n_b), 'lengths of two coordinate lists are different'
    d = a - b
    rmsd = np.sqrt(np.sum(np.square(d))/float(n_a))
    return rmsd

"""
def superimpose_model(ref_R_s, cmp_R_s):
    sup = SVDSuperimposer()
    a_ref_R_s = np.array(ref_R_s,'f')
    a_cmp_R_s = np.array(cmp_R_s,'f')
    #
    sup.set(a_ref_R_s, a_cmp_R_s)
    sup.run()
    rot, tran = sup.get_rotran()
    sup_cmp_R_s = sup.get_transformed()
#    sup_cmp_R_s = np.dot(cmp_R_s, rot) + tran
#    sup_rmsd = sup.get_rms()
    return sup_cmp_R_s, rot, tran

def get_sup_rmsd(ref_R_s, cmp_R_s):
    sup = SVDSuperimposer()
    a_ref_R_s = np.array(ref_R_s,'f')
    a_cmp_R_s = np.array(cmp_R_s,'f')
    #
    sup.set(a_ref_R_s, a_cmp_R_s)
    sup.run()
    rot, tran = sup.get_rotran()
    sup_cmp_R_s = sup.get_transformed()
#    sup_cmp_R_s = np.dot(cmp_R_s, rot) + tran
    sup_rmsd = sup.get_rms()
    return sup_rmsd
"""

def rotate_given_coord(R_s, rot, tran):
    a_R_s = np.array(R_s,'f')
    rot_R_s = np.dot(a_R_s, rot) + tran
    return rot_R_s

def read_pdb_coords(pdbname, use_atom=[]):
    original_lines = []
    coord_s = []
    with open('%s'%pdbname) as fp:
        for line in fp:
            if line.startswith('ATOM'):
                atom = Atom(line)
                if len(use_atom) == 0:
                    coord_s.append(atom.coord)
                    original_lines.append(atom.raw_line)
                elif len(use_atom) > 0:
                    if atom.atmName in use_atom:
                        coord_s.append(atom.coord)
                        original_lines.append(atom.raw_line)
    return coord_s, original_lines

"""
def get_superimposed_coord_from_pdb(ref_R_s, cmp_pdb):
    cmp_R_s, cmp_ori_lines = read_pdb_coords(cmp_pdb)
    sup_cmp_R_s, rot, tran = superimpose_model(ref_R_s, cmp_R_s)
    return sup_cmp_R_s, cmp_ori_lines
"""

def write_pdb_w_new_Rs(cmp_ori_lines, sup_cmp_R_s, new_name):
    cont = []
    fmt = '%8.3f%8.3f%8.3f'
    for i_r, R in enumerate(sup_cmp_R_s):
        tuple_R = tuple(R)
        ori_line = cmp_ori_lines[i_r]
        cont.append('%s%s%s'%(ori_line[:30],fmt%tuple_R,ori_line[54:]))
    fout = open('%s'%new_name,'w')
    fout.write(''.join(cont))
    fout.close()
    return

"""
def write_superimposed_model(ref_pdb, cmp_pdb, new_name):
    ref_R_s, ref_ori_lines = read_pdb_coords(ref_pdb)
    cmp_R_s, cmp_ori_lines = read_pdb_coords(cmp_pdb)
    sup_cmp_R_s, rot, tran = superimpose_model(ref_R_s, cmp_R_s)
    #
    cont = []
    fmt = '%8.3f%8.3f%8.3f'
    for i_r, R in enumerate(sup_cmp_R_s):
        tuple_R = tuple(R)
        ori_line = cmp_ori_lines[i_r]
        cont.append('%s%s%s'%(ori_line[:30],fmt%tuple_R,ori_line[54:]))
    fout = open('%s'%new_name,'w')
    fout.write(''.join(cont))
    fout.close()
    return
"""
        

    
    
