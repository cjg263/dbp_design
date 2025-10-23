#from ajasja_utils import *

import sys,os,json
sys.path.append('/rhf/allocations/cg160/lab_scripts/')
import tempfile
import numpy as np
import pandas as pd
import pandas
from optparse import OptionParser
import time
import glob
#import design_utils
import string
import random
import count_hbond_types

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score
#from pyrosetta.rosetta import core
#sys.path.append('/home/norn/software/silent_tools')
#import silent_tools
import shutil
import subprocess
from collections import Counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def make_binder_design_xml(weights):
    xml_script = f"""
<ROSETTASCRIPTS>
	# This is the relax script that Cameron adapted from the af2 superposition relax (originally written by Nate)
		<SCOREFXNS>
			<ScoreFunction name="sfxn" weights="{weights}" />
			<ScoreFunction name="sfxn_design" weights="{weights}" >
				<Reweight scoretype="arg_cation_pi" weight="3" />
			</ScoreFunction>

			</SCOREFXNS>
    <TASKOPERATIONS>
        <SelectBySASA name="PR_monomer_core" mode="sc" state="monomer" probe_radius="2.2" core_asa="15" surface_asa="15" core="0" boundary="1" surface="1" verbose="0" />
    </TASKOPERATIONS>
    <RESIDUE_SELECTORS>
        <Chain name="chainA" chains="1"/>
        <Not name="chainB" selector="chainA"/>
        <Neighborhood name="interface_chA" selector="chainB" distance="14.0" />
        <Neighborhood name="interface_chB" selector="chainA" distance="14.0" />
        <And name="AB_interface" selectors="interface_chA,interface_chB" />
        <Not name="Not_interface" selector="AB_interface" />
        <And name="actual_interface_chA" selectors="AB_interface,chainA" />
        <And name="actual_interface_chB" selectors="AB_interface,chainB" />
        <And name="chainB_Not_interface" selectors="Not_interface,chainB" />
        <Or name="chainB_or_not_interface" selectors="chainB,Not_interface" />

        <Task name="all_cores" fixed="true" task_operations="PR_monomer_core" packable="false" designable="false"/>
        <And name="monomer_core" selectors="all_cores,chainA" />
        <Not name="not_monomer_core" selector="monomer_core" />

        <ResidueName name="ala_gly_pro_cys" residue_name3="ALA,GLY,PRO,CYS" />
        <Not name="not_ala_gly_pro_cys" selector="ala_gly_pro_cys" />

        <ResidueName name="apolar" residue_name3="ALA,CYS,PHE,ILE,LEU,MET,THR,PRO,VAL,TRP,TYR" />
        <Not name="polar" selector="apolar" />

        <ResidueName name="water" residue_name3="HOH" />

        <And name="chainB_fixed" >
            <Or selectors="chainB_Not_interface,water" />
        </And>
        <And name="chainB_not_fixed" selectors="chainB">
            <Not selector="chainB_fixed"/>
        </And>

				<ResiduePDBInfoHasLabel name="rec_helix" property="ss_RH" />
        <ResiduePDBInfoHasLabel name="motif_res" property="MOTIF" />
        <Or name="rec_helix_or_motif_res" selectors="rec_helix,motif_res" />

    </RESIDUE_SELECTORS>


    <TASKOPERATIONS>
        <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
        <InitializeFromCommandline name="init" />
        <IncludeCurrent name="current" />
        <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1" />
				<LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />

	    <OperateOnResidueSubset name="restrict_to_binder_interface" selector="chainB_or_not_interface">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>

        <OperateOnResidueSubset name="restrict_target_Not_interface" selector="chainB_fixed">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>

    </TASKOPERATIONS>
    <MOVERS>
        <SwitchChainOrder name="chain1onlypre" chain_order="1" />
        <ScoreMover name="scorepose" scorefxn="sfxn" verbose="false" />
        <ParsedProtocol name="chain1only">
            <Add mover="chain1onlypre" />
            <Add mover="scorepose" />
        </ParsedProtocol>

				<FastRelax name="relax_chain1" scorefxn="sfxn" repeats="1" batch="false" ramp_down_constraints="false" cartesian="false" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" task_operations="ex1_ex2,limitchi2" >
            <MoveMap name="MM" >
                <Chain number="1" chi="true" bb="true" />
            </MoveMap>
        </FastRelax>

				<ParsedProtocol name="chain1only_relax">
            <Add mover="chain1onlypre" />
            <Add mover="relax_chain1" />
        </ParsedProtocol>

        <TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />

        <DeleteRegionMover name="delete_polar" residue_selector="polar" rechain="false" />
    </MOVERS>
    <FILTERS>
				<TotalSasa name="total_sasa"
					threshold="0"
					hydrophobic="0"
					polar="0" />

				# Net Charge
				<NetCharge name="net_charge" chain="1" />

				# Charge/SASA
				<CalculatorFilter name="net_charge_over_sasa" equation="net_charge / total_sasa * 1000" threshold="0" confidence="0">
						<Var name="net_charge" filter="net_charge"/>
						<Var name="total_sasa" filter="total_sasa"/>
				</CalculatorFilter>

				<Sasa name="interface_buried_sasa" confidence="0" />

				<Ddg name="ddg"  threshold="-10" jump="1" repeats="5" repack="1" relax_mover="min" confidence="0" scorefxn="sfxn" extreme_value_removal="1" />

				<ShapeComplementarity name="interface_sc" verbose="0" min_sc="0.55" write_int_area="1" write_median_dist="1" jump="1" confidence="0"/>

				<ScoreType name="total_score_MBF" scorefxn="sfxn" score_type="total_score" threshold="0" confidence="0" />
				<MoveBeforeFilter name="total_score_monomer" mover="chain1only" filter="total_score_MBF" confidence="0" />
				<ResidueCount name="res_count_MBF" max_residue_count="9999" confidence="0"/>
				<MoveBeforeFilter name="res_count_monomer" mover="chain1only" filter="res_count_MBF" confidence="0" />


				<CalculatorFilter name="score_per_res" equation="total_score_monomer / res" threshold="-3.5" confidence="0">
						<Var name="total_score_monomer" filter="total_score_monomer"/>
						<Var name="res" filter="res_count_monomer"/>
				</CalculatorFilter>

				<MoveBeforeFilter name="total_score_relax_monomer" mover="chain1only_relax" filter="total_score_MBF" confidence="0" />

				<CalculatorFilter name="score_per_res_relax" equation="total_score_relax_monomer / res" threshold="-3.5" confidence="0">
						<Var name="total_score_relax_monomer" filter="total_score_relax_monomer"/>
						<Var name="res" filter="res_count_monomer"/>
				</CalculatorFilter>


				<BuriedUnsatHbonds name="sbuns5.0_heavy_ball_1.1D" use_reporter_behavior="true" report_all_heavy_atom_unsats="true"
													 scorefxn="sfxn" residue_selector="AB_interface" ignore_surface_res="false" print_out_info_to_pdb="true"
													 confidence="0" use_ddG_style="true" burial_cutoff="0.01" dalphaball_sasa="true" probe_radius="1.1"
													 atomic_depth_selection="5.0" atomic_depth_deeper_than="false" burial_cutoff_apo="0.2"
													 atomic_depth_resolution="0.49" max_hbond_energy="1.5"/>
				<BuriedUnsatHbonds name="vbuns5.0_heavy_ball_1.1D" use_reporter_behavior="true" report_all_heavy_atom_unsats="true"
													 scorefxn="sfxn" residue_selector="AB_interface" ignore_surface_res="false" print_out_info_to_pdb="true"
													 confidence="0" use_ddG_style="true" dalphaball_sasa="true" probe_radius="1.1" atomic_depth_selection="5.0"
													 burial_cutoff="1000" burial_cutoff_apo="0.2" atomic_depth_apo_surface="5.5" atomic_depth_resolution="0.49" max_hbond_energy="1.5"/>

				<ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainB" binder_selector="chainA" confidence="0" />
				<ContactMolecularSurface name="cms_rec_helix" distance_weight="0.5" target_selector="chainB" binder_selector="rec_helix_or_motif_res" confidence="0" />


				<SSPrediction name="pre_sspred_overall" cmd="/rhf/allocations/cg160/software/scipred/runpsipred_single" use_probability="0" use_svm="0" threshold="0.85" confidence="0" />
				<MoveBeforeFilter name="sspred_overall" mover="chain1only" filter="pre_sspred_overall" confidence="0" />

				<SSPrediction name="pre_mismatch_probability" confidence="0" cmd="/rhf/allocations/cg160/software/scipred/runpsipred_single" use_probability="1" mismatch_probability="1" use_svm="0" />
				<MoveBeforeFilter name="mismatch_probability" mover="chain1only" filter="pre_mismatch_probability" confidence="0" />

				ScorePoseSegmentFromResidueSelectorFilter name="was_done" in_context="1" residue_selector="DONE_res" scorefxn="sfxn" confidence="0" />


				<SSShapeComplementarity name="ss_sc_pre" verbose="0" confidence="0" />
				<MoveBeforeFilter name="ss_sc" mover="chain1only" filter="ss_sc_pre" confidence="0"/>
    </FILTERS>

    <JUMP_SELECTORS>
        <JumpIndex name="jump1" jump="1"/>
        <Not name="notjump1" selector="jump1"/>
    </JUMP_SELECTORS>

    <MOVE_MAP_FACTORIES>
    <MoveMapFactory name="mm_design" bb="0" chi="0" jumps="0">
        <Backbone enable="False" residue_selector="actual_interface_chA" />
        <Chi enable="True" residue_selector="actual_interface_chA" />
        <Jumps enable="False" jump_selector="notjump1" />
        <Jumps enable="False" jump_selector="jump1" />
    </MoveMapFactory>
    </MOVE_MAP_FACTORIES>

    <MOVERS>
        <FastRelax name="FastRelax"
									 scorefxn="sfxn_design"
									 movemap_factory="mm_design"
									 repeats="3"
									 batch="false"
									 ramp_down_constraints="false"
									 cartesian="true"
									 bondangle="false"
									 bondlength="false"
									 min_type="dfpmin_armijo_nonmonotone"
									 task_operations="current,restrict_to_binder_interface,ex1_ex2"
									 relaxscript="/rhf/allocations/cg160/software/dbp_design/flags_and_weights/no_ref.rosettacon2018.beta_nov16_constrained.txt" >
        </FastRelax>

    </MOVERS>

    <APPLY_TO_POSE>
    </APPLY_TO_POSE>
    <PROTOCOLS>

        Add mover="FastRelax" />
		<Add filter_name="interface_buried_sasa" />
        <Add filter_name="ddg" />
        <Add filter_name="interface_sc" />
        <Add filter_name="score_per_res" />
        <Add filter_name="score_per_res_relax" />
        Add filter="vbuns5.0_heavy_ball_1.1D" />
        Add filter="sbuns5.0_heavy_ball_1.1D" />
        Add filter="mismatch_probability" />
        Add filter="sspred_overall" />
        <Add filter="contact_molecular_surface" />
        Add filter="cms_rec_helix" />
        <Add filter="ss_sc" />
	    <Add filter="total_sasa" />
		<Add filter="net_charge" />
        <Add filter="net_charge_over_sasa" />

    </PROTOCOLS>
</ROSETTASCRIPTS>


    """
    return xml_script

def hbond_score(pose, tag, df_scores, opts):
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

    columns, result = count_hbond_types.count_hbonds_protein_dna(pose, tag, opts.hbond_energy_cut)
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

def count_backbone_phosphate_contacts(pose, df_scores, opts) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making backbone hydrogen bonds with DNA phosphate atoms.
    '''

    energy_cutoff = opts.hbond_energy_cut
    pose_hb = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose)
    pose_hb = pose.get_hbonds()
    backbone_phosphate_contacts = []
    for hbond in range(1,pose_hb.nhbonds()+1):
        hbond = pose_hb.hbond(hbond)
        if hbond.energy() > float(energy_cutoff):
            continue
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        donor_hatm = hbond.don_hatm()
        acceptor_atm = hbond.acc_atm()
        don_atom_type = pose.residue(donor_res).atom_name(donor_hatm).strip(" ")
        acc_atom_type = pose.residue(acceptor_res).atom_name(acceptor_atm).strip(" ")
        if acc_atom_type in ['OP1','OP2'] and don_atom_type == 'H':
            backbone_phosphate_contacts.append(donor_res)
    df_scores['n_backbone_phosphate_contacts'] = [len(backbone_phosphate_contacts)]
    print(f"n_posphate_contacts: {len(backbone_phosphate_contacts)}")
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
#                             print(f"We've got a cross-strand bidentate! {aa} to {bp} and {other_bp}")
                            num_cross_strand_bidentates += 1
    return num_bidentates, num_bridged_bidentates, num_cross_strand_bidentates

def count_hbonds_protein_dna(pose, opts) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making base-specific hydrogen bonds with DNA.
    Used in calculating the RotamerBoltzmann score (see calc_rboltz)
    '''
    mc_bb_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "OP2",  "OP1",  "O5'",  "C5'",  "C4'",  "O4'",
               "C3'",  "O3'",  "C2'",  "C1'", "H5''",  "H5'",  "H4'",  "H3'", "H2''",  "H2'",  "H1'"]
    aa_bb_atoms = ['N', 'CA', 'C', 'O', 'CB', '1H', '2H', '3H', 'HA', 'OXT','H'] #I added this to avoid counting backbone - DNA hydrogen bonds
    DNA_base_names = ['ADE','GUA','THY','CYT', '5IU', 'BRU', 'RGU', 'RCY', 'RAD', 'RTH']

    energy_cutoff = opts.hbond_energy_cut
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
        hbonds_don_hatm.append(donor_hatm)
        hbonds_acc_atm.append(acceptor_atm)
        hbonds_don_res.append(donor_res)
        hbonds_acc_res.append(acceptor_res)
        hbond_energy = hbond.energy()
        hbonds_energy.append(hbond_energy)

    protein_dna = 0
    protein_dna_specific = 0
    aa_identities_base = []
    aa_pos_base = []
    aa_identities_phosphate = []
    aa_pos_phosphate = []
    bidentate_list = []

#     print(hbonds_don_hatm)

    for residue in range(len(hbonds_acc_res)) :
        if hbonds_energy[residue] > float(energy_cutoff):
            continue

        don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
        acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")

        if pose.residue(hbonds_don_res[residue]).name().split(':')[0] in DNA_base_names :
            if not pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                protein_dna += 1
                if not don_atom_type in mc_bb_atoms and not acc_atom_type in aa_bb_atoms:
                    print("We've got a protein-DNA interaction")
                    protein_dna_specific += 1
                    aa_identities_base.append(pose.residue(hbonds_acc_res[residue]).name1())
                    if not int(hbonds_acc_res[residue]) in aa_pos_base:
                        aa_pos_base.append(int(hbonds_acc_res[residue]))
                    else:
                        bidentate_list.append(int(hbonds_acc_res[residue]))
                    base_id = dna_base_list[dna_res_list.index(hbonds_don_res[residue])]
                    base_count_dict[base_id] += 1
                elif don_atom_type in mc_bb_atoms and not acc_atom_type in aa_bb_atoms:
                    print("We've got a phosphate interactions")
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
                    print("We've got phosphate interactions")
                    aa_identities_phosphate.append(pose.residue(hbonds_acc_res[residue]).name1())
                    if not int(hbonds_don_res[residue]) in aa_pos_phosphate:
                        aa_pos_phosphate.append(int(hbonds_don_res[residue]))
#     print(f"Total number of base-specific hydrogen bonds: {protein_dna_specific}")
#     print(f"Final list of base-specific hydrogen bonds: {aa_pos}")

    return aa_pos_base, aa_pos_phosphate, list(set(bidentate_list))

def calc_rboltz(pose, df, opts):
    '''
    Takes in a pose and the existing DataFrame of scores for a design and returns the DataFrame with
    three new columns: largest RotamerBoltzmann of ARG/LYS/GLU/GLN residues; average of the top two
    RotamerBoltzmann scores (includes every amino acid type); and median RotamerBoltzmann (includes every amino acid type)
    '''
    notable_aas = ['ARG','GLU','GLN','LYS']
    base_hbond_residues, phosphate_hbond_residues, bidentates = count_hbonds_protein_dna(pose, opts)

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
    
    repack_neighbors = False
    # get base specific rboltz metrics
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

    # get base specific rboltz metrics with repacking
    if repack_neighbors == True:
        design_df = get_rboltz_vals(base_hbond_residues,repack_neighbors = True)
        RKQE_subset = design_df[design_df['residue_name'].isin(notable_aas)]
        if len(RKQE_subset) > 0:
            df['max_rboltz_RKQE_repack'] = -1 * RKQE_subset['rboltz'].min()
        else:
            df['max_rboltz_RKQE_repack'] = 0

        if len(design_df) > 0:
            df['avg_top_two_rboltz_repack'] = -1 * np.average(design_df['rboltz'].nsmallest(2))
            df['median_rboltz_repack'] = -1 * np.median(design_df['rboltz'])
        else:
            df['avg_top_two_rboltz_repack'] = 0
            df['median_rboltz_repack'] = 0

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

    if repack_neighbors == True:
    # get phosphate specific rboltz metrics
        design_df = get_rboltz_vals(phosphate_hbond_residues, repack_neighbors=True)
        RKQE_subset = design_df[design_df['residue_name'].isin(notable_aas)]
        if len(RKQE_subset) > 0:
            df['max_rboltz_RKQE_phosphate_repack'] = -1 * RKQE_subset['rboltz'].min()
        else:
            df['max_rboltz_RKQE_phosphate_repack'] = 0

        if len(design_df) > 0:
            df['avg_top_two_rboltz_phosphate_repack'] = -1 * np.average(design_df['rboltz'].nsmallest(2))
            df['median_rboltz_phosphate_repack'] = -1 * np.median(design_df['rboltz'])
        else:
            df['avg_top_two_rboltz_phosphate_repack'] = 0
            df['median_rboltz_phosphate_repack'] = 0

    return df

def run_pose(pose, opts, dfs, sfd_out=None, silent_tag='', rand_string=''):

    # define output and output directory
    silent_out = opts.silent_out
    outdir = os.path.dirname(silent_out)

    weight_file = opts.weights

    protein_len = pose.total_residue()


    # =============================================================================
    #                            Run relax
    # =============================================================================

    print("Computing Metrics")
    t0 = time.time()
    is_fast = False

    out_tag = silent_tag + '_' + opts.rand_string
    # align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    # pose_len = pose.total_residue()
    # ref_pose_len = ref_pose.total_residue()
    #
    # for residue in range(1,ref_pose_len+1):
    #     target_residue = pose_len - ref_pose_len + residue
    #     ref_res = ref_pose.residue(residue)
    #     ref_pose_atom_idx = int(ref_res.atom_index('P'))
    #     target_res = pose.residue(target_residue)
    #     target_pose_atom_idx = int(target_res.atom_index('P'))
    #     atom_id_ref_pose = pyrosetta.rosetta.core.id.AtomID(ref_pose_atom_idx,residue)
    #     atom_id_target_pose = pyrosetta.rosetta.core.id.AtomID(target_pose_atom_idx,target_residue)
    #     align_map[atom_id_target_pose] = atom_id_ref_pose
    # align = pyrosetta.rosetta.core.scoring.superimpose_pose(pose, ref_pose, align_map)
    #
    # binder = pose.split_by_chain()[1]
    # binder.append_pose_by_jump(ref_pose,1)
    # pose = binder.clone()


    xml_full = make_binder_design_xml(weights=opts.weights)
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_full)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])
    #df_scores = pd.DataFrame() #Change this if you want no ddg calc
    df_scores['out_tag'] = out_tag

    # Add setting info to the pose and the score file
    for key in opts.__dict__:
        core.pose.add_comment(pose, str(key), str(opts.__dict__[key]))
        df_scores[key] = str(opts.__dict__[key])


    if opts.silent_out != '' and opts.silent_tag != '':
        sfd_out = core.io.silent.SilentFileData( silent_out, False, False, "binary", core.io.silent.SilentFileOptions())
        struct = sfd_out.create_SilentStructOP()
        struct.fill_struct( pose, out_tag )
        sfd_out.add_structure( struct )
        sfd_out.write_silent_struct( struct, silent_out )
    else:
        struct = sfd_out.create_SilentStructOP()
        struct.fill_struct( pose, out_tag )
        sfd_out.add_structure( struct )
        sfd_out.write_silent_struct( struct, silent_out )

    # Score hbonds to DNA
    df_scores = hbond_score(pose, out_tag, df_scores, opts)

    # Score backbone phosphate contacts
    df_scores['n_backbone_phosphate_contacts'] = [0]
    df_scores = count_backbone_phosphate_contacts(pose, df_scores, opts)
    
    # Calculate RotamerBoltzmann scores
    df_scores = calc_rboltz(pose, df_scores, opts)
    
    # Append sequence to pose
    df_scores['sequence'] = pose.split_by_chain()[1].sequence()
    dfs.append(df_scores)


    t1 = time.time()
    print(f'full-scale design for took {t1-t0}')

    return dfs

def clean_pose(pose):
    protein_chain = pose.split_by_chain()[1]
    DNA_pose = pose.split_by_chain()[2]
    DNA_chains = DNA_pose.split_by_chain()
    if len(DNA_chains) > 1:
        print("Combining DNA into single chain")
        for chain in range(1, len(DNA_chains) + 1 ):
            DNA_pose.append_pose_by_jump( DNA_chains[chain], DNA_pose.size() )
    ft = pyrosetta.FoldTree()
    ft.simple_tree(DNA_pose.size())
    DNA_pose.fold_tree(ft)
    pose = protein_chain.clone()
    pose.append_pose_by_jump( DNA_pose, pose.size() )
    return pose


def id_generator(size=12, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

def main():
    t0 = time.time()
    #========================================================================
    # Build option parser
    #========================================================================
    parser = OptionParser(usage="usage: %prog [options] FILE", version="0.1")

    # Pose
    parser.add_option("--silent_in", type="string", dest="silent_in", metavar="STR", default = '', help="Path to input silent file")
    parser.add_option("--silent_tag", type="string", dest="silent_tag", metavar="STR", default = '', help="Input silent tag")
    parser.add_option("--rand_string", type="string", dest="rand_string", metavar="STR", default = '', help="Random string")

    #parser.add_option("--ref_pose", type="string", dest="ref_pose", metavar="STR", default = '', help="Path to input reference pose")

    parser.add_option("--silent_out", type="string", dest="silent_out", metavar="STR", default = '', help="path to output silent file")

    # score function
    parser.add_option("--weights", type="string", dest="weights", metavar="STR", default = '', help="Path to score function weights")
    parser.add_option("--flags_file", type="string", dest="flags_file", metavar="STR", default = '', help="Path to score function flags file")


    # Metric settings
    parser.add_option("--hbond_energy_cut", type="float", dest="hbond_energy_cut", default=-0.5, metavar="STR", help="hbond energy cutoff")

    # Number of random tags to design per silent
    parser.add_option("--n_per_silent", type="int", dest="n_per_silent", default=0, metavar="STR", help="Number of random tags to design per silent. 0 for all tags.")



    # Use energy cut
    (opts, args) = parser.parse_args()
    parser.set_defaults()

    #========================================================================
    # Initialize pyrosetta and pose
    #========================================================================

    # INITIALIZE

    if opts.silent_in != '':
        silent_in_init = '-in:file:silent_struct_type binary'
    else: silent_in_init = ''
    if opts.silent_out != '':
        silent_out_init = '-out:file:silent_struct_type binary'
    else: silent_out_init = ''

    if opts.flags_file != '':
        flags = opts.flags_file
    else:
        flags = '/rhf/allocations/cg160/software/dbp_design/flags_and_weights/RM8B_flags'

    init(f'-ignore_zero_occupancy {silent_in_init} {silent_out_init} -beta_nov16 -output_virtual 1 -precompute_ig 1  \
           -run:preserve_header true -holes:dalphaball /rhf/allocations/cg160/software/dbp_design/DAlphaBall/DAlphaBall.gcc @{flags}')



    # Load reference pose
#    ref_pdb = opts.ref_pose
#    print(ref_pdb)
#    ref_pose = pose_from_pdb(ref_pdb)
#    print(ref_pose)


    ## Load silent file
    sfd = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
    sfd.read_file(opts.silent_in)
    if opts.silent_tag != '' and opts.silent_tag in sfd.tags():
        ss = sfd.get_structure(opts.silent_tag)
        pose = Pose()
        ss.fill_pose(pose)
        pose = clean_pose(pose)
    else:
        sys. exit

    # define output and output directory
    if opts.silent_out != '':
        silent_out = opts.silent_out
        outdir = os.path.dirname(silent_out)
    else:
        print('No specified output file. Aborting job.')
        sys. exit
    os.makedirs(outdir, exist_ok=True)

    # Define output csv file
    out_csv_f = silent_out.replace('.silent','.csv')
    
    dfs = []
    if os.path.isfile(out_csv_f):
        csv_done = pd.read_csv(out_csv_f)
        csv_is_file = True
        dfs.append(csv_done)
    else: csv_is_file = False

    if opts.silent_in != '' and opts.silent_tag != '':
        if csv_is_file:
            return
        rand_string = id_generator()
        dfs = run_pose(pose, opts, dfs)
        scores = pd.concat(dfs, axis=0, ignore_index=True)
        scores.to_csv(out_csv_f)
    else:
        sfd_out = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
        if opts.n_per_silent != 0:
            rand_tags = random.choices(sfd.tags(), k=opts.n_per_silent)
            print(f'Designing {len(rand_tags)} tags from {opts.silent_in}')
            for silent_tag in rand_tags:
                if csv_is_file:
                    is_done=csv_done['out_tag'].str.contains(silent_tag).sum()
                    if is_done>0:
                        print('Tag complete. Skipping to next.')
                        continue
                print(f'\nrunning {silent_tag}')
                ss = sfd.get_structure(silent_tag)
                pose = Pose()
                ss.fill_pose(pose)
                pose = clean_pose(pose)
                dfs = run_pose(pose, opts, dfs, sfd_out, silent_tag)
                scores = pd.concat(dfs, axis=0, ignore_index=True)
                scores.to_csv(out_csv_f)
        else:
            for silent_tag in sfd.tags():
                if csv_is_file:
                    is_done=csv_done['out_tag'].str.contains(silent_tag).sum()
                    if is_done>0:
                        print('Tag complete. Skipping to next.')
                        continue
                print(f'\nrunning {silent_tag}')
                ss = sfd.get_structure(silent_tag)
                pose = Pose()
                ss.fill_pose(pose)
                pose = clean_pose(pose)
                dfs = run_pose(pose, opts, dfs, sfd_out, silent_tag)
                scores = pd.concat(dfs, axis=0, ignore_index=True)
                scores.to_csv(out_csv_f)

if __name__ == '__main__':
    main()
