import numpy as np
import random
from pyrosetta import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import subprocess




def get_seq_for_pdb(pdb_path):
     result = subprocess.run(['/home/norn/scripts/pdb2seq', pdb_path], stdout=subprocess.PIPE)
     return result.stdout.decode('utf-8').strip()

def get_cdhit_clustermap(clusterfile):
    shortname2clusterid = {}
    with open(clusterfile,'r') as fp:
        for line in fp:
            x = line.strip().split()
            if line.startswith('>Cluster'):
                cl_id = int(x[1])
            else:
                shortname = x[2].split('.pdb')[0][1:][:-3]
                shortname2clusterid[shortname] = cl_id
    return shortname2clusterid

def gen_rst_from_point_measurement(npz, residue_idxes1, native_idx, restrain_phi_psi=False, hbnet_weight=10.0):
    """
    Generates restraints based on observed distances. test
    
    npz: observed geometries across multiple native examples (axis=-1)
    residue_set: The residue numbers (1-numbered).
    hbnet_weight: weight of the restraint functions
    """
    dist = npz['reshaped_dist6d_mats'][:,:,native_idx]
    omega = npz['reshaped_omega6d_mats'][:,:,native_idx]
    theta = npz['reshaped_theta6d_mats'][:,:,native_idx]
    phi = npz['reshaped_phi6d_mats'][:,:,native_idx]
    bb_phi = np.deg2rad(npz['reshaped_bbphi_mats'][:,native_idx])
    bb_psi = np.deg2rad(npz['reshaped_bbpsi_mats'][:,native_idx])
    
    # Setup the constraints
    rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : [], 'bb_phi' : [], 'bb_psi' : []}    
    sd_dist = 0.5 # distance
    sd_angle = np.deg2rad(5.0) # angle
    
    residue_idxes0 = residue_idxes1 - 1
    
    for a in residue_idxes0:
        for b in residue_idxes0:
            # distance
            id1 = rosetta.core.id.AtomID(5,a+1) # CB
            id2 = rosetta.core.id.AtomID(5,b+1) # CB
            func = rosetta.core.scoring.func.HarmonicFunc(dist[a,b], sd_dist)
            w_func = rosetta.core.scoring.func.ScalarWeightedFunc(hbnet_weight, func)
            rst['dist'].append([a,b,rosetta.core.scoring.constraints.AtomPairConstraint(id1, id2, w_func)])
            
            # omega
            if b<a: # half matrix
                id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
                id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
                id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
                id4 = rosetta.core.id.AtomID(2,b+1) # CA-j
                func = rosetta.core.scoring.func.CircularHarmonicFunc(omega[a,b], sd_angle)
                w_func = rosetta.core.scoring.func.ScalarWeightedFunc(hbnet_weight, func)
                rst['omega'].append([a,b,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, w_func)])
            
            # theta
            id1 = rosetta.core.id.AtomID(1,a+1) #  N-i
            id2 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id3 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id4 = rosetta.core.id.AtomID(5,b+1) # CB-j
            func = rosetta.core.scoring.func.CircularHarmonicFunc(theta[a,b], sd_angle)
            w_func = rosetta.core.scoring.func.ScalarWeightedFunc(hbnet_weight, func)
            rst['theta'].append([a,b,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, w_func)])
            
            # phi
            id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
            func = rosetta.core.scoring.func.HarmonicFunc(phi[a,b], sd_angle)
            w_func = rosetta.core.scoring.func.ScalarWeightedFunc(hbnet_weight, func)
            rst['phi'].append([a,b,rosetta.core.scoring.constraints.AngleConstraint(id1,id2,id3, w_func)])
    
    if restrain_phi_psi:
        for a in residue_idxes0:
            # bb phi
            id1 = rosetta.core.id.AtomID(3, a)     # C,  i-1
            id2 = rosetta.core.id.AtomID(1, a+1)   # N,  i
            id3 = rosetta.core.id.AtomID(2, a+1)   # CA, i
            id4 = rosetta.core.id.AtomID(3, a+1)   # C,  i
            func = rosetta.core.scoring.func.CircularHarmonicFunc(bb_phi[a], sd_angle)
            w_func = rosetta.core.scoring.func.ScalarWeightedFunc(hbnet_weight, func)
            rst['bb_phi'].append([a,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, w_func)])
            
            # bb psi
            id1 = rosetta.core.id.AtomID(1, a+1)   # N,  i
            id2 = rosetta.core.id.AtomID(2, a+1)   # CA, i
            id3 = rosetta.core.id.AtomID(3, a+1)   # C,  i
            id4 = rosetta.core.id.AtomID(1, a+2)   # N,  i+1
            func = rosetta.core.scoring.func.CircularHarmonicFunc(bb_psi[a], sd_angle)
            w_func = rosetta.core.scoring.func.ScalarWeightedFunc(hbnet_weight, func)
            rst['bb_psi'].append([a,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, w_func)])
    
    print("distance restraints: %d"%(len(rst['dist'])))
    print("omega restraints: %d"%(len(rst['omega'])))
    print("phi restraints:   %d"%(len(rst['phi'])))
    print("theta restraints: %d"%(len(rst['theta'])))
    print("bb phi restraints: %d"%(len(rst['bb_phi'])))
    print("bb psi restraints: %d"%(len(rst['bb_phi'])))
    
    return rst

def read_pssm(filepath):
    aas = 'pos nativeAA A R N D C Q E G H I L K M F P S T W Y V Ap Rp Np Dp Cp Qp Ep Gp Hp Ip Lp Kp Mp Fp Pp Sp Tp Wp Yp Vp x1 x2'.split()
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=[0,1,2], skipfooter=4, engine='python',names=aas)
    pssm = df.iloc[:,2:22].values
    return pssm

def save_pssm(pssm, outfile):
    f_out = open(outfile,'w')
    f_out.write('\n')
    f_out.write('Last position-specific scoring matrix computed, weighted observed percentages rounded down, information per position, and relative weight of gapless real matches to pseudocounts\n')
    f_out.write('            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V\n')
    
    for i in range(0, pssm.shape[0]):
        aa = 'A'
        pos = str(i+1)
        odds = pssm[i,:]
        odds_str = ' '.join([str(x) for x in odds])
        occ_str = ' '.join([str(1.0) for x in odds])
        f_out.write(pos+' '+aa+' '+odds_str+' '+occ_str+' 0.00 0.00'+'\n')
    
    f_out.write('\n\n\n\n')

def gen_rst(npz, params, include_more=False, name=""): 
    """
    Generates restraints based on a binned distribution of
    observed distances.
    
    npz: Binned distribution
    params: parameters
    w: weight on the restraint function
    """
    dist, omega, theta, phi, bb_phi, bb_psi, n_contacts, n_phipsi, mean_dist = npz['dist6d'], npz['omega6d'], npz['theta6d'], npz['phi6d'], npz['bb_phi'], npz['bb_psi'], npz['recorded_contacts'], npz['recorded_bb_phipsi'], npz['mean_dist']
    w = 1.0

    # dictionary to store Rosetta restraints
    rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : [], 'bb_phi' : [], 'bb_psi' : []}    
    spline_coordinates  = {'dist' : {}, 'omega' : {}, 'theta' : {}, 'phi' : {}, 'bb_phi' : {}, 'bb_psi' : {}}    
    
    ########################################################
    # assign parameters
    ########################################################
    PCUT  = params['PCUT']
    PCUT1 = params['PCUT1']
    EBASE = params['EBASE']
    EREP  = params['EREP']
    DREP  = params['DREP']
    PREP  = params['PREP']
    SIGD  = params['SIGD']
    SIGM  = params['SIGM']
    MEFF  = params['MEFF']
    DCUT  = params['DCUT']
    ALPHA = params['ALPHA']
    DSTEP = params['DSTEP']
    ASTEP = np.deg2rad(params['ASTEP'])
    seq = params['seq']
                
    # ------------------------------------------------------
    # Compute the distance constraints
    # ------------------------------------------------------
    bins = np.arange(0, 20, DSTEP)
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    dist = -np.log(dist[:,:,1:]) # first bin is placeholder for fraction not aligned 
    # Only make constraints for positions where at least 30% align to something 
    # and only make constraints for positions that are in contact (<12)
    dist[~np.isfinite(dist)] = 1000.0 # In rare cases the probability is so low, that numpy throws inf, which causes trouble with rosetta.

    if include_more:
        I,J = np.where((n_contacts>=1) & (mean_dist<10000))
    else:
        I,J = np.where((n_contacts>PCUT) & (mean_dist<15.0))
    
    #print("6D constraints for")
    #print('select resi','+'.join([str(x+1) for x in list(set(I))]))
    n = 0
    for a,b in zip(I,J):
        if b>a:
            y = pyrosetta.rosetta.utility.vector1_double()
            _ = [y.append(v) for v in dist[a,b]] 
            spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, DSTEP, x,y)
            ida = rosetta.core.id.AtomID(5,a+1)
            idb = rosetta.core.id.AtomID(5,b+1)
            rst['dist'].append([a,b,rosetta.core.scoring.constraints.AtomPairConstraint(ida, idb, spline)])
            spline_coordinates['dist'][(a,b)] = (x, y)
    
    print("distance restraints: %d"%(len(rst['dist'])))

    # ------------------------------------------------------
    # Compute omega constraints: -pi..pi
    # ------------------------------------------------------
    bins = np.arange(-np.pi + ASTEP*0.5, np.pi + ASTEP*0.5, ASTEP) # starting at -180+0.5*15/180, inc: 15 deg
    omega = omega[:,:,1:] # first bin is placeholder for fraction not aligned 
    x = pyrosetta.rosetta.utility.vector1_double()
    omega = -np.log(omega)
    
    bins_expanded = np.concatenate(([bins[0]-2*ASTEP, bins[0]-ASTEP], bins, [bins[-1]+ASTEP, bins[-1]+2*ASTEP]))
    _ = [x.append(v) for v in bins_expanded]
    
    for a,b in zip(I,J):
        if b>a:
            y = pyrosetta.rosetta.utility.vector1_double()
            vals_expanded = np.concatenate(([omega[a,b][-2],omega[a,b][-1]], omega[a,b], [omega[a,b][0], omega[a,b][1]]))
            _ = [y.append(v) for v in vals_expanded]            
            spline = rosetta.core.scoring.func.SplineFunc("", w, 0.0, ASTEP, x,y)
            id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
            id4 = rosetta.core.id.AtomID(2,b+1) # CA-j
            rst['omega'].append([a,b,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])
            spline_coordinates['omega'][(a,b)] = (x, y)
            

    print("omega restraints: %d"%(len(rst['omega'])))

    # ------------------------------------------------------
    # Compute the phi / theta potential
    # ------------------------------------------------------
    # theta
    bins = np.arange(-np.pi + ASTEP*0.5, np.pi + ASTEP*0.5, ASTEP)
    theta = theta[:,:,1:]
    x = pyrosetta.rosetta.utility.vector1_double()
    theta = -np.log(theta)
    
    bins_expanded = np.concatenate(([bins[0]-2*ASTEP, bins[0]-ASTEP], bins, [bins[-1]+ASTEP, bins[-1]+2*ASTEP]))
    _ = [x.append(v) for v in bins_expanded]
    for a,b in zip(I,J):
        if b!=a:
            y = pyrosetta.rosetta.utility.vector1_double()
            vals_expanded = np.concatenate(([theta[a,b][-2],theta[a,b][-1]], theta[a,b], [theta[a,b][0], theta[a,b][1]]))            
            _ = [y.append(v) for v in vals_expanded]
            spline = rosetta.core.scoring.func.SplineFunc("", w, 0.0, ASTEP, x,y)
            id1 = rosetta.core.id.AtomID(1,a+1) #  N-i
            id2 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id3 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id4 = rosetta.core.id.AtomID(5,b+1) # CB-j
            rst['theta'].append([a,b,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])
            spline_coordinates['theta'][(a,b)] = (x, y)
            
    # phi
    bins = np.arange(ASTEP*0.5, np.pi + ASTEP*0.5, ASTEP) # starting at 0.5*15/180, inc: 15 deg
    phi = phi[:,:,1:]
    x = pyrosetta.rosetta.utility.vector1_double()
    phi = -np.log(phi)
    
    bins_expanded = np.concatenate(([bins[0]-2*ASTEP, bins[0]-ASTEP], bins, [bins[-1]+ASTEP, bins[-1]+2*ASTEP]))
    
    _ = [x.append(v) for v in bins_expanded]
    for a,b in zip(I,J):
        if b!=a:
            y = pyrosetta.rosetta.utility.vector1_double()
            
            # To fill in the extra bins, do a linear interpolation from the outermost two points
            dlower = phi[a,b][1] - phi[a,b][0] 
            dupper = phi[a,b][-1] - phi[a,b][-2] 
            vals_expanded = np.concatenate(([phi[a,b][0]-2*dlower,phi[a,b][0]-dlower], phi[a,b], [phi[a,b][-1]+dupper, phi[a,b][-1]+2*dupper]))
            _ = [y.append(v) for v in vals_expanded]
            spline = rosetta.core.scoring.func.SplineFunc("", w, 0.0, ASTEP, x,y)
            id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
            rst['phi'].append([a,b,rosetta.core.scoring.constraints.AngleConstraint(id1,id2,id3, spline)])
            spline_coordinates['phi'][(a,b)] = (x, y)
            
                            
    print("phi restraints:   %d"%(len(rst['phi'])))
    print("theta restraints: %d"%(len(rst['theta'])))
    
            
    # ------------------------------------------------------
    # Compute the backbone torsion constraints
    # ------------------------------------------------------
    bins = np.arange(-np.pi + ASTEP*0.5, np.pi + ASTEP*0.5, ASTEP)
    
    # We need to add a couple of datapoints on either side of the bins
    # so that we have control over the spline in the relevant range
    # -pi to pi
    bins_expanded = np.concatenate(([bins[0]-2*ASTEP, bins[0]-ASTEP], bins, [bins[-1]+ASTEP, bins[-1]+2*ASTEP]))
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins_expanded]
    
    K = np.where(n_phipsi > PCUT)[0]
    
    print("phi psi constraints for")
    #print('select resi','+'.join([str(x+1) for x in list(set(K))]))
    
    # bb phi
    bb_phi = -np.log(bb_phi[:,1:])
    for a in K:
        y = pyrosetta.rosetta.utility.vector1_double()
        vals_expanded = np.concatenate(([bb_phi[a][-2],bb_phi[a][-1]], bb_phi[a], [bb_phi[a][0], bb_phi[a][1]]))        
        _ = [y.append(v) for v in vals_expanded]
        spline = rosetta.core.scoring.func.SplineFunc("", w, 0.0, ASTEP, x,y)
        id1 = rosetta.core.id.AtomID(3, a)     # C,  i-1
        id2 = rosetta.core.id.AtomID(1, a+1)   # N,  i
        id3 = rosetta.core.id.AtomID(2, a+1)   # CA, i
        id4 = rosetta.core.id.AtomID(3, a+1)   # C,  i
        rst['bb_phi'].append([a,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])
        spline_coordinates['bb_phi'][a] = (x, y)
    
    # bb psi
    bb_psi = -np.log(bb_psi[:,1:])
    for a in K:
        y = pyrosetta.rosetta.utility.vector1_double()
        vals_expanded = np.concatenate(([bb_psi[a][-2],bb_psi[a][-1]], bb_psi[a], [bb_psi[a][0], bb_psi[a][1]]))
        _ = [y.append(v) for v in vals_expanded]
        spline = rosetta.core.scoring.func.SplineFunc("", w, 0.0, ASTEP, x,y)
        id1 = rosetta.core.id.AtomID(1, a+1)   # N,  i
        id2 = rosetta.core.id.AtomID(2, a+1)   # CA, i
        id3 = rosetta.core.id.AtomID(3, a+1)   # C,  i
        id4 = rosetta.core.id.AtomID(1, a+2)   # N,  i+1
        rst['bb_psi'].append([a,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])
        spline_coordinates['bb_psi'][a] = (x, y)
        

    print("bb phi restraints: %d"%(len(rst['bb_phi'])))
    print("bb psi restraints: %d"%(len(rst['bb_phi'])))

    return rst, spline_coordinates

def read_fasta(file):
    fasta=""
    with open(file, "r") as f:
        for line in f:
            if(line[0] == ">"):
                continue
            else:
                line=line.rstrip()
                fasta = fasta + line;
    return fasta


def add_rst(pose, rst, sep1, sep2, params):
    # Pairwise restraints
    array = []
    array += [r for a,b,r in rst['dist'  ] if abs(a-b)>=sep1 and abs(a-b)<sep2]
    array += [r for a,b,r in rst['omega' ] if abs(a-b)>=sep1 and abs(a-b)<sep2]
    array += [r for a,b,r in rst['theta' ] if abs(a-b)>=sep1 and abs(a-b)<sep2]
    array += [r for a,b,r in rst['phi'   ] if abs(a-b)>=sep1 and abs(a-b)<sep2]

    # Backbone phi psi
    n_term = 0
    c_term = len(str(pose.sequence()))-1
    array += [r for a,r in rst['bb_phi'] if a!=n_term and a!=c_term]
    array += [r for a,r in rst['bb_psi'] if a!=n_term and a!=c_term]
        
    if len(array) < 1:
        print("No restraints added")
        return
    print("Applied ", len(array), " constraints!")
    
    random.shuffle(array) # Why random shuffle here?

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in array]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)

def extract_pose_from_silent(silent_file, tag=None):
    input_stream = rosetta.core.import_pose.pose_stream.SilentFilePoseInputStream(silent_file)
    pose = rosetta.core.pose.Pose()
    while input_stream.has_another_pose():
            input_stream.fill_pose(pose)
            this_tag = rosetta.core.pose.tag_from_pose(pose)
            if tag is None:
                return pose
            elif tag==this_tag:
                return pose

def dump_poses_from_silent(silent_file, tags, outdir):
    input_stream = rosetta.core.import_pose.pose_stream.SilentFilePoseInputStream(silent_file)
    pose = rosetta.core.pose.Pose()    
    while input_stream.has_another_pose():
            input_stream.fill_pose(pose)
            this_tag = rosetta.core.pose.tag_from_pose(pose)
            if this_tag in tags:
                pose.dump_pdb(outdir + this_tag + '.pdb')

def get_ss_extent(ss_str):
  """
  This function is supossed to take ss string like the one 
  this HHHHLLLLHHHEELL
  and return
  'H1':[0,1,2,3],
  'L1':[4,5,6,7],
  'H2':[8,9,10]
  'E1':[11,12] 
  'L2':[13,14]
  """
  d_annotate = {}

  current_char = ss_str[0]
  current_char_list = []
  ss_strech_type_counters = {'H':0, 'E':0, 'L':0} 

  for idx, char in enumerate(ss_str):
    if char != current_char:
      # We are about to look at a new secondary structure element
      # First we store the current element which we keep in 
      # current_char_list.
      ss_strech_type_counters[current_char] += 1 # keeping track of ss element type indexes
      current_ss_element_string = current_char +  str(ss_strech_type_counters[current_char]) # this could contain 'H1' for instance
      d_annotate[current_ss_element_string] = current_char_list
      # And now we reinitialize the array and the current_char 
      current_char_list = [idx]
      current_char = char
    else:
      current_char_list.append(idx)

  return d_annotate
