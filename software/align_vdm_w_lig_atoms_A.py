#!/usr/bin/env python
#Gyu Rie Lee, Nov 2020
#Implementation of vdM alignment following the concept from "A defined structural unit enables de novo design of small-molecule–binding proteins" by N.F.Polizzi and W.F.DeGrado, Science, 2020.
# modified by Cameron Glasscock to align by ligand atoms

import os
import sys
import numpy as np
from kabsch_align import kabsch_align_coords
sys.path.insert(0,'/home/gyurie/lib')
from libPDB import Atom

class VDMCloud:
    def __init__(self,wdir,list_fn,aln_res_name,func_name,target_pdb,target_resis,align_atoms=['N7','C2','N6']):
        self.wdir = os.path.abspath('%s'%wdir)
        self.list_fn = list_fn.strip()
        self.align_res_name = aln_res_name
        self.func_name = func_name
        self.target_pdb = target_pdb
        self.target_resis = target_resis.split(',')
        self.align_atoms = align_atoms
        self.header = '_'.join(self.list_fn.split('_')[:-1])

    def read_vdm_pdbs(self):
        self.pdbs = []
        with open('%s'%self.list_fn) as fp:
            for line in fp:
                self.pdbs.append(line.strip())
        return
    def read_target_coords(self):
        #1. Read the coords of aligning atoms to get the transfomration matrix only
        self.ref_root_R = []
        self.ref_all_R = []
        root_R = {}
        with open('%s'%self.target_pdb) as fp:
            for line in fp:
                if not line.startswith('ATOM') and not line.startswith('HETATM'):
                    continue
                atm = Atom(line)
                self.ref_all_R.append(atm.coord)
                if atm.atmName in self.align_atoms and \
                   atm.resNo  == int(self.target_res):
                       root_R[atm.atmName] = atm.coord
        self.ref_root_R = [root_R['C2'],root_R['N6'],root_R['N7']]
        return
    def read_vdm_coords(self):
        #1. Read the coords of aligning atoms to get the transfomration matrix only
        #2. Read all coords for final superimposition
        #
        self.root_Rs = []
        self.all_Rs = []
        self.template_s = []
        atom = []
        #
        for pdbname in self.pdbs:
            root_R = {}
            all_R = []
            template = []
            with open('%s'%pdbname) as fp:
                for line in fp:
                    if not line.startswith('ATOM') and not line.startswith('HETATM'):
                        continue
                    atm = Atom(line)
                    if line.startswith('ATOM') and 'DA' not in line:
                        all_R.append(atm.coord)
                        template.append(atm.template)
                    elif atm.atmName in self.align_atoms and \
                            'DA' in line: #line.startswith('HETATM'):'
                        root_R[atm.atmName] = atm.coord
#                        all_R.append(atm.coord)
#                    else:
#                        all_R.append(atm.coord)
            #
            root_R = [root_R['C2'],root_R['N6'],root_R['N7']]
            self.root_Rs.append(np.array(root_R))
            self.all_Rs.append(np.array(all_R))
            self.template_s.append(template)
        return
    def run_alignment(self):
        ref_root_R = self.ref_root_R
        ref_all_R = self.ref_all_R
        #
        #First coord becomes the reference and the others will be superimposed
        transformed_all_Rs = []
        for i in range(len(self.root_Rs)):
            cmp_root_R = self.root_Rs[i]
            cmp_all_R = self.all_Rs[i]
            #
            #NOTE: ref_root_R changes inside kabsch function, so added copy.deepcopy of input xyz1 in there
            t_cmp_R = kabsch_align_coords(ref_root_R,cmp_root_R,cmp_all_R)
            transformed_all_Rs.append(t_cmp_R)
        #
        self.Rs_to_write = transformed_all_Rs
        #self.Rs_to_write.extend(transformed_all_Rs)
        return
    ##TODO
    #Before writing these coords into pdbs, can directly use the arrays for clustering.
    #But will have to save which are SC atoms, functional group atoms, etc
    def write_superimposed_coords(self,target_re):
        #
        dirname = '%s/base_aligned_%s_hotspots/%s'%(self.wdir,self.align_res_name,self.target_pdb.split('/')[-1].replace('.pdb',''))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        #TODO
        #def get_template
        #all vdMs should be clean, with same atom names and number of lines
        #can make this simpler if all the vdMs have been standardized before
        """
        template = []
        i_atm = 0
        with open('%s'%self.pdbs[0]) as fp:
            for line in fp:
                i_atm += 1
                if line.startswith('ATOM'):
                    template.append('%6s%5d%11s%4d'%(line[:6],i_atm,line[11:22],1))
                elif line.startswith('HETATM'):
                    template.append('%6s%5d%11s%4d'%(line[:6],i_atm,line[11:22],2))
        """
        #
        i_model = 0
        for R in self.Rs_to_write:
            i_model += 1
            #
            cont = []
            #
            template = self.template_s[i_model-1]
            for i in range(len(template)):
                w_R = list(R[i])
                cont.append('%s%8.3f%8.3f%8.3f'%(template[i],w_R[0],w_R[1],w_R[2]))
            fout = open('%s/%s_%s_%d.pdb'%(dirname,\
                                           self.align_res_name,\
                                           self.target_res,i_model),'w')
            fout.write('%s\n'%('\n'.join(cont)))
            fout.close()
        return
    def superimpose_vdMs(self):
        self.read_vdm_pdbs()
        for target_res in self.target_resis:
            self.target_res = target_res
            self.read_target_coords()
            self.read_vdm_coords()
            self.run_alignment()
            #this is optional
            self.write_superimposed_coords(target_res)
        return

def main():
    dirname = sys.argv[1]
    vdm_list_fn = sys.argv[2]
    align_res_name = sys.argv[3]
    func_name = sys.argv[4]
    target_pdb = sys.argv[5]
    target_resis = sys.argv[6]

    vdmCloud = VDMCloud(dirname,vdm_list_fn,align_res_name,\
                        func_name,target_pdb,target_resis)
    vdmCloud.superimpose_vdMs()
    return

if __name__ == '__main__':
    main()
