import pyrosetta
import os
import glob
import math
from statistics import median

from sys import argv

pyrosetta.init("-mute all")
sfxn = pyrosetta.rosetta.core.scoring.ScoreFunction()
sfxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1)

def load_pose(name):
    if type(name) == str:
        if name[-4:] == ".pdb":
            pose = pyrosetta.io.pose_from_pdb(name)
        elif name[-8:] == ".pdb.bz2":
            with open(name, "rb") as f:  # read bz2 bytestream, decompress and decode
                pose = io.to_pose(io.pose_from_pdbstring(bz2.decompress(f.read()).decode()))
        else:
            print("unrecognized file")
    else:
        try:
            pose = name.clone()
        except:
            print("input must be a pose or a path to a pose")
    return pose

### select good diffusion outputs

def get_ss_dict(pose):
    dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    ss = ''
    for i in range(1,pose.size()+1):
        if dssp.get_dssp_secstruct(i) in ['E','L','H']:
            ss = ss + dssp.get_dssp_secstruct(i)
        else:
            ss = ss + 'L'
    ss = ss + 't'
    #print(ss)
    ss_elements = {'L':[],'E':[],'H':[]}
    if ss[0] == ss[1]:
        start = 0
    else:
        start = 1
    ss = ss.replace('LEL','LLL').replace('LHL','LLL')
    for i in range(1,pose.size()):
        if ss[i] != ss[i+1]:
            end = i
            ss_elements[ss[i]].append((start+1,end+1))
            start = i+1
    return ss_elements

def trim_chain(pose):
    ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    start = pose.size()
    end = pose.size()
    for i in range(pose.size()):
        if ss.get_dssp_secstruct(i+1) in ["H","E"]:
            start = i
            break
    for i in range(pose.size()):
        if ss.get_dssp_secstruct(pose.size()-i-1) in ["H","E"]:
            end = pose.size()-i
            break
    newpose = pyrosetta.rosetta.core.pose.Pose()
    if end - start > 40:
        for i in range(start,end+1):
            newpose.append_residue_by_bond(pose.residue(i))
    return newpose

def residue_distance(pose,resa,resb):
    return math.sqrt(sum(dim**2 for dim in (pose.residues[resa].xyz("CB") - pose.residues[resb].xyz("CB"))))

def split_and_trim(pose):
    breaks = []
    newpose = pyrosetta.rosetta.core.pose.Pose()
    start = 1
    for i in range(2,pose.size()):
        dist = math.sqrt(sum(dim**2 for dim in (pose.residues[i].xyz("CA") - pose.residues[i+1].xyz("CA"))))
        if dist > 4:
            breaks.append((start,i))
            start = i+1
    breaks.append((start,pose.size()))

    #print(breaks)
    for start,end in breaks:
        chpose = pyrosetta.rosetta.core.pose.Pose()
        for i in range(start,end+1):
            chpose.append_residue_by_bond(pose.residue(i))
        trim = trim_chain(chpose)
        if trim.size() > 1:
            pyrosetta.rosetta.core.pose.append_pose_to_pose(newpose,trim,1)
    return(newpose)


def filter(p):
    protein = p.split_by_chain()[1]
    dna = p.split_by_chain()[2]

    trim = split_and_trim(protein)

    ssdict = get_ss_dict(protein)
    llengths = [b-a for (a,b) in ssdict['L']]
    
    ncontact = []
    sscount = pyrosetta.rosetta.protocols.fldsgn.filters.SecondaryStructureCountFilter()
    for (a,b) in ssdict['H'] + ssdict['E']:
        nc = 0
        sse_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueSpanSelector(a,b)
        nbselector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(sse_selector,10,False)
        for (c,d) in ssdict['H'] + ssdict['E']:
            other_sse_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueSpanSelector(c,d)
            cont_sel = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(nbselector, other_sse_selector)
            cont = pyrosetta.rosetta.core.select.get_residues_from_subset(cont_sel.apply(protein))
            #print(a,c,cont)
            if len(cont) >2:
                nc +=1
        ncontact.append(nc)

    
    return min(ncontact), max(llengths)
    
    
    
