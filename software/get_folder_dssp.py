#!/home/davidcj/.conda/envs/pyrosetta_env/bin/python
from pyrosetta import * 
import pyrosetta  
init('-mute all')

import glob
import numpy as np
import pandas as pd 

import argparse 


def get_ss(pdb):
    pose = pose_from_pdb(pdb)
    tmp  = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    ss = tmp.get_dssp_secstruct()
    return ss


def get_folder_ss(folder, outfolder):
    """
    Calculates the DSSP for all pdbs in a folder and saves to a csv file
    """
    pdbs = glob.glob(os.path.join(folder,'*pdb'))

    dssp_all = []

    for i,p in enumerate(pdbs):
        if i%10== 0:
            print(f'On protein {i}')
        ss = get_ss(p)
        dssp_all.append(ss)

    with open(os.path.join(outfolder, 'dssp_all.csv'), 'w') as fp:
        fp.write('path,dssp\n')
        for pdb,ss in zip(pdbs,dssp_all):
            fp.write(pdb + ',' + ss + '\n')

    print(f'All done with SS calculation. Wrote results to {os.path.join(outfolder, "dssp_all.csv")}')

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-folder', '-f', '--folder', dest='folder', required=True, 
            help='Folder of pdbs to calculate DSSP for')

    args = parser.parse_args()


    get_folder_ss(args.folder, args.folder)

if __name__ ==r'__main__':
    main()
