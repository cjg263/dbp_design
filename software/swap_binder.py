import os, sys
import math

#sys.path.insert(0,'/software/pyrosetta3.10/latest/')
from pyrosetta import *
from pyrosetta.rosetta import *

import numpy as np
from collections import defaultdict
import time
import argparse
import itertools
import subprocess
import time
import pandas as pd
import glob
from decimal import Decimal

# Getting Brian's stuff
# This line is needed to avoid inconsistent crashes due to cache failures. 
# It may need to be changed depending on where / how you want to run the script, but should work on digs
os.environ['NUMBA_CACHE_DIR'] = '/net/scratch/' + os.environ['USER'] + '/numba_cache/'

# This package is nice to have since it offers a simple way to write score files
sys.path.append("/projects/protein-DNA-binders/scripts/npose")
#import npose_util as nu
import npose_util_pyrosetta as nup
# End importing Brian's stuff

import distutils.spawn

sys.path.append('/projects/protein-DNA-binders/scripts/silent_tools/')
import silent_tools

init( "-beta_nov16 -in:file:silent_struct_type binary" +
    " -holes:dalphaball /work/tlinsky/Rosetta/main/source/external/DAlpahBall/DAlphaBall.macgcc" +
    " -use_terminal_residues true -mute basic.io.database core.scoring -mute core.io.silent" )

def cmd(command, wait=True):
    the_command = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if (not wait):
        return
    the_stuff = the_command.communicate()
    return str( the_stuff[0]) + str(the_stuff[1] )

def range1( iterable ): return range( 1, iterable + 1 )

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string


#################################
# Argument Validation Functions
#################################

def contact_radius_check( input_val ):
    distance = float( input_val )
    if distance <= 0:
        raise argparse.ArgumentTypeError( 'The argument -contact_radius must be a positive float, you used %s'%input_val )
    return distance

def temperature_check( input_val ):
    cast_val = float( input_val )
    if cast_val <= 0:
        raise argparse.ArgumentTypeError( 'The argument -temperature must be a positive float, you used %s'%input_val )
    return cast_val


#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()
parser.add_argument("-in:file:silent", type=str, help='The name of a silent file to run this metric on. pdbs are not accepted at this point in time')
parser.add_argument("-af2_apo_binders", type=str, help='Optional, the radius around an atom used to determine contact density (default 7.35A)')
parser.add_argument("-af2_binder_suffix", type=str, default = '', help='Optional, the radius around an atom used to determine contact density (default 7.35A)')

args = parser.parse_args( sys.argv[1:] )
silent = args.__getattribute__("in:file:silent")

sfxn = core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")

chainA = core.select.residue_selector.ChainSelector("A")
DNA_property = core.chemical.ResidueProperty(6)
chainB = core.select.residue_selector.ResiduePropertySelector(DNA_property)
interface_on_A = core.select.residue_selector.NeighborhoodResidueSelector(chainB, 10.0, False)
interface_on_B = core.select.residue_selector.NeighborhoodResidueSelector(chainA, 10.0, False)
interface_by_vector = core.select.residue_selector.InterGroupInterfaceByVectorSelector(interface_on_A, interface_on_B)
interface_by_vector.cb_dist_cut(8)
#interface_by_vector.cb_dist_cut(5.5)
interface_by_vector.vector_angle_cut(75)
interface_by_vector.vector_dist_cut(9)

#################################
# Function Definitions
#################################

# align two things using sequence and accepting gaps - from bcov
def pymol_align( move_pose, to_pose, res_move=None, res_to=None, atoms=["N", "CA", "C"], throw_away=0.1 ):

    move_res = np.array(res_move)
   
    to_res = np.array(res_to)

    seq_move = "x" + move_pose.sequence()
    seq_to = "x" + to_pose.sequence()
    print(seq_move)
    print(seq_to)

    seq_move = "".join(np.array(list(seq_move))[move_res])
    seq_to = "".join(np.array(list(seq_to))[to_res])

    from Bio import pairwise2
    alignment = align_move, align_to, idk1, idk2, idk3 = pairwise2.align.globalxs(seq_move,seq_to, -2, -1)[0]
    print( align_move )
    print( align_to )
    
    move_to_pairs = []
    coords_move = utility.vector1_numeric_xyzVector_double_t()
    coords_to = utility.vector1_numeric_xyzVector_double_t()
    
    i_move = 0
    i_to = 0
    for i in range(len(align_move)):
        if ( align_move[i] == align_to[i] ):

            seqpos_move = move_res[i_move]
            seqpos_to = to_res[i_to]

            move_to_pairs.append((seqpos_move, seqpos_to))

            for atom in atoms:
                coords_move.append(move_pose.residue(seqpos_move).xyz(atom))
                coords_to.append(to_pose.residue(seqpos_to).xyz(atom))


        if ( align_move[i] != "-" ):
            i_move += 1
        if ( align_to[i] != "-" ):
            i_to += 1
    print(move_to_pairs)
    if len(move_to_pairs) == 1: raise Exception('Not enough pairs to align to, skipping pose')
    move_pose_copy = move_pose.clone()


    rmsd = 0

    distances = []

    if ( len(move_to_pairs) > 0 ):

        rotation_matrix = numeric.xyzMatrix_double_t()
        move_com = numeric.xyzVector_double_t()
        ref_com = numeric.xyzVector_double_t()

        protocols.toolbox.superposition_transform( coords_move, coords_to, rotation_matrix, move_com, ref_com )
        
        protocols.toolbox.apply_superposition_transform( move_pose, rotation_matrix, move_com, ref_com )
        count = 0
        for seqpos_move, seqpos_to in move_to_pairs:
            count += 1
            print(f'iteration {count}')
            for atom in atoms:
#                 print(move_pose.residue(seqpose_move))
#                 print(move_pose.residue(seqpos_move).xyz(atom))
#                 print(to_pose.residue(seqpos_to))
#                 print(to_pose.residue(seqpos_to).xyz(atom))
                distance = move_pose.residue(seqpos_move).xyz(atom).distance_squared(to_pose.residue(seqpos_to).xyz(atom))
                rmsd += distance
                distances.append(distance)

        rmsd /= len(move_to_pairs)*len(atoms)
        rmsd = np.sqrt(rmsd)
    
    move_pose = move_pose_copy

    distances = np.array(distances)

    print("Initial RMSD: %.3f"%rmsd)

    cutoff = np.percentile(distances, 90)
    print("Cutoff %.3f"%cutoff)

    mask = distances <= cutoff

    print(mask.sum(), len(mask))

    coords_move_old = list(coords_move)
    coords_to_old = list(coords_to)
    # move_to_pairs_old = move_to_pairs

    # move_to_pairs = []
    coords_move = utility.vector1_numeric_xyzVector_double_t()
    coords_to = utility.vector1_numeric_xyzVector_double_t()

    for i in range(len(coords_move_old)):
        if ( not mask[i] ):
            continue
        coords_move.append(coords_move_old[i])
        coords_to.append(coords_to_old[i])
        # move_to_pairs.append(move_to_pairs_old[i])

    print(len(coords_move), len(coords_move_old))

    if len(move_to_pairs) < 3:
        print('Too few interface residues found, skipping pose')
        raise Exception('Too few interface residues found, skipping pose')

    rmsd = 0
    imask = -1
    if ( len(move_to_pairs) > 0 ):

        rotation_matrix = numeric.xyzMatrix_double_t()
        move_com = numeric.xyzVector_double_t()
        ref_com = numeric.xyzVector_double_t()

        protocols.toolbox.superposition_transform( coords_move, coords_to, rotation_matrix, move_com, ref_com )
        
        protocols.toolbox.apply_superposition_transform(move_pose, rotation_matrix, move_com, ref_com)

        for seqpos_move, seqpos_to in move_to_pairs:
            for atom in atoms:
                imask += 1
                if ( not mask[imask] ):
                    continue
                distance = move_pose.residue(seqpos_move).xyz(atom).distance_squared(to_pose.residue(seqpos_to).xyz(atom))
                rmsd += distance

        rmsd /= imask
        rmsd = np.sqrt(rmsd)

        zero = numeric.xyzVector_double_t(0, 0, 0)
        xform = nup.vector_to_xform( zero - ref_com ) @ nup.matrix_to_xform( rotation_matrix ) \
                    @ nup.vector_to_xform( move_com )


    print("Final RMSD: %.3f over %i atoms"%(rmsd, mask.sum()))

    # return rmsd, move_to_pairs, move_pose, xform
    return rmsd, move_to_pairs, move_pose, rotation_matrix, move_com, ref_com

def find_counterpart( curr_tag, parent_tags ):

    # This assumes that the child tags start with the truncated parent tag
    for parent_tag in parent_tags:

        trunc_parent_tag = parent_tag
        if args.af2_binder_suffix != '': trunc_parent_tag = my_rstrip( parent_tag, args.af2_binder_suffix )

#        if curr_tag.startswith( trunc_parent_tag ): return parent_tag  ## old version that caused us some headache with repeated sequences!
        if curr_tag == trunc_parent_tag : return parent_tag
    
    print( f'Could not find a parent for tag {curr_tag}, this was my trunc tag {trunc_parent_tag}' )
    raise Exception( f'Could not find a parent for tag {curr_tag}, this was my trunc tag {trunc_parent_tag}' )

def modify_pose( pose, af2_binder ):
    # This is where you would actually run whatever calculations you want to on your pose

    by_vector = interface_by_vector.apply( pose )
    by_vector_res = []

    for seqpos in range(1, af2_binder.size()+1):
        if ( af2_binder.residue( seqpos ).name() in ["GLY","GLY:NtermProteinFull","GLY:CtermProteinFull"] ): continue

        if ( by_vector[seqpos] ):
            by_vector_res.append(seqpos)
    
    if len( by_vector_res ) <= 1:
        print( f'Insufficiently large interface by vector found, skipping' )
        raise Exception( f'Insufficiently large interface by vector found, skipping' )

    # move_pose, to_pose
    _, _, aligned_af2_binder, _, _, _ = pymol_align( af2_binder, pose, res_move=by_vector_res, res_to=by_vector_res, atoms=["CB"] )

    aligned_af2_binder.append_pose_by_jump( pose.split_by_chain()[2], aligned_af2_binder.size() )
    if pose.num_chains() == 3:
        aligned_af2_binder.append_pose_by_jump( pose.split_by_chain()[3], aligned_af2_binder.size() )


    return aligned_af2_binder


# I/O Functions

def add2silent( tag, pose, sfd_out ):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )

def pose_from_silent_lines(structure, tag):
    vec = utility.vector1_std_string()
    vec.append(tag)

    stream = std.istringstream(structure)

    sfd = core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
    sfd.read_stream(stream, vec, True, "fake")

    pose = core.pose.Pose()
    sfd.get_structure(tag).fill_pose(pose)

    return pose

def main( pdb, sfd_in, sfd_out, af2_binder_silent, af2_tags ):

    t0 = time.time()
    print( "Attempting pose: %s"%pdb )
    pose = Pose()
    sfd_in.get_structure( pdb ).fill_pose( pose )

    print('loaded_pose')

    #af2_apo_binder_tag = find_counterpart( pdb, af2_tags['tags'] )
    af2_apo_binder_tag = pdb + '_af2pred'
    print(af2_apo_binder_tag)
    #structure = (
    #    silent_tools.silent_header( af2_tags ) +
    #    "".join(silent_tools.get_silent_structure( af2_binder_silent, af2_tags, af2_apo_binder_tag ))
    #    )

    #og_binder = pose_from_silent_lines("".join(structure), af2_apo_binder_tag)
    
    og_binder = Pose()
    af2_binder_silent.get_structure(af2_apo_binder_tag).fill_pose(og_binder)
    print('about to modify pose')
    outpose = modify_pose( pose, og_binder )
    print('about to add to silent')
    add2silent( pdb, outpose, sfd_out )

    seconds = int(time.time() - t0)
    print( "protocols.jd2.JobDistributor: %s reported success in %i seconds"%( pdb, seconds ) )

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

sfd_out = core.io.silent.SilentFileData( "out.silent", False, False, "binary", core.io.silent.SilentFileOptions())

sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
sfd_in.read_file(silent)

sfd_af2_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
sfd_af2_in.read_file(args.af2_apo_binders)

pdbs = silent_tools.get_silent_index(silent)["tags"]
#af2_binder_tags = silent_tools.get_silent_index(sfd_af2_in)["tags"] #args.af2_apo_binders)
af2_binder_tags = []
checkpoint_filename = "check.point"
debug = False

finished_structs = determine_finished_structs( checkpoint_filename )

for pdb in pdbs:

    if pdb in finished_structs: continue

    if debug: main( pdb, sfd_in, sfd_out, args.af2_apo_binders, af2_binder_tags )

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try: main( pdb, sfd_in, sfd_out, sfd_af2_in, af2_binder_tags )

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )

    # We are done with one pdb, record that we finished
    record_checkpoint( pdb, checkpoint_filename )

sfd_out.write_all("out.silent", False)





