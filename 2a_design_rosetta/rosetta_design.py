import sys,os,json

# get the program directory
from sys import argv
prog_dir = os.path.dirname(sys.argv[0])
sys.path.append(prog_dir)

import tempfile
import numpy as np
import pandas as pd
import pandas
from optparse import OptionParser
import time
import glob
import string
import random

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score
import shutil
import subprocess
from collections import Counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def make_binder_design_xml(weight_file, \
                            pssm_f='', pssm_weight='0.5', pssmCut=0, hbnet_restrictions='', is_fast=''):
    xml_script = f"""
    <ROSETTASCRIPTS>
          <SCOREFXNS>
            <ScoreFunction name="sfxn" weights="{weight_file}" />
            <ScoreFunction name="sfxn_fa_atr" weights="empty" >
                <Reweight scoretype="fa_atr" weight="1" />
            </ScoreFunction>
            <ScoreFunction name="sfxn_relax" weights="{weight_file}" >
                <Reweight scoretype="arg_cation_pi" weight="3" />
            </ScoreFunction>
            <ScoreFunction name="sfxn_design" weights="{weight_file}" >
                <Reweight scoretype="res_type_constraint" weight="{pssm_weight}" />
                <Reweight scoretype="aa_composition" weight="1.0" />
	            <Reweight scoretype="arg_cation_pi" weight="3" />
            </ScoreFunction>
            <ScoreFunction name="sfxn_design_fast" weights="{weight_file}">
                 <Reweight scoretype="res_type_constraint" weight="{pssm_weight}"/>
                 <Reweight scoretype="aa_composition" weight="1.0" />
                 <Reweight scoretype="arg_cation_pi" weight="3"/>

                 # lk_ball is slooooooooooow
                 <Reweight scoretype="lk_ball" weight="0" />
                 <Reweight scoretype="lk_ball_iso" weight="0" />
                 <Reweight scoretype="lk_ball_bridge" weight="0" />
                 <Reweight scoretype="lk_ball_bridge_uncpl" weight="0" />

                 # turn off the next slowest parts of the score function
                 Set etable_no_hydrogens="true" />
                 <Reweight scoretype="fa_elec" weight="0" />
                 <Reweight scoretype="fa_intra_atr_xover4" weight="0" />
                 <Reweight scoretype="fa_intra_rep_xover4" weight="0" />
                 <Reweight scoretype="fa_intra_sol_xover4" weight="0" />
                 <Reweight scoretype="fa_intra_elec" weight="0" />
            </ScoreFunction>
            <ScoreFunction name="vdw_sol" weights="empty" >
                <Reweight scoretype="fa_atr" weight="1.0" />
                <Reweight scoretype="fa_rep" weight="0.55" />
                <Reweight scoretype="fa_sol" weight="1.0" />
            </ScoreFunction>
            <ScoreFunction name="sfxn_softish" weights="{weight_file}" >
                <Reweight scoretype="arg_cation_pi" weight="3"/>
                <Reweight scoretype="fa_rep" weight="0.15" />
                <Reweight scoretype="pro_close" weight="0"/>
            </ScoreFunction>
            <ScoreFunction name="pssm_score">
                <Reweight scoretype="res_type_constraint" weight="1"/>
            </ScoreFunction>
        </SCOREFXNS>

        <TASKOPERATIONS>
          <SelectBySASA name="PR_monomer_core_sel" mode="sc" state="monomer" probe_radius="2.2" core_asa="15" surface_asa="15" core="0" boundary="1" surface="1" verbose="0" />
        </TASKOPERATIONS>

        <RESIDUE_SELECTORS>
            <Chain name="chainA" chains="A"/>
            <Chain name="chainB" chains="B"/>
            <Neighborhood name="interface_chA" selector="chainB" distance="12.0" />
            <Neighborhood name="interface_chB" selector="chainA" distance="12.0" />
            <And name="AB_interface" selectors="interface_chA,interface_chB" />
            <Not name="Not_interface" selector="AB_interface" />
            <And name="actual_interface_chA" selectors="AB_interface,chainA" />
            <And name="actual_interface_chB" selectors="AB_interface,chainB" />
            <And name="chainA_not_interface" selectors="Not_interface,chainA" />

            <ResidueName name="pro_and_gly_positions" residue_name3="PRO,GLY" />
            <ResidueName name="apolar" residue_name3="ALA,CYS,PHE,ILE,LEU,MET,THR,PRO,VAL,TRP,TYR" />
            <Not name="polar" selector="apolar" />

            <InterfaceByVector name="interface_by_vector" cb_dist_cut="11" nearby_atom_cut="5.5" vector_angle_cut="75" vector_dist_cut="9" grp1_selector="actual_interface_chA" grp2_selector="actual_interface_chB"/>

            <Task name="all_cores" fixed="true" task_operations="PR_monomer_core_sel" packable="false" designable="false"/>

            <And name="for_hydrophobic" selectors="actual_interface_chA,interface_by_vector">
                <Not selector="all_cores" />
            </And>

            <And name="for_polar" selectors="actual_interface_chA,interface_by_vector">
                <Not selector="all_cores" />
            </And>

            <True name="true_sel" />

            <ResiduePDBInfoHasLabel name="HOTSPOT_res" property="HOTSPOT" />
            <ResiduePDBInfoHasLabel name="RIFRES" property="RIFRES" />

            <ResiduePDBInfoHasLabel name="hbr_R" property="hbnet_resn_R"/>
            <ResiduePDBInfoHasLabel name="hbr_N" property="hbnet_resn_N"/>
            <ResiduePDBInfoHasLabel name="hbr_D" property="hbnet_resn_D"/>
            <ResiduePDBInfoHasLabel name="hbr_Q" property="hbnet_resn_Q"/>
            <ResiduePDBInfoHasLabel name="hbr_E" property="hbnet_resn_E"/>
            <ResiduePDBInfoHasLabel name="hbr_H" property="hbnet_resn_H"/>
            <ResiduePDBInfoHasLabel name="hbr_K" property="hbnet_resn_K"/>
            <ResiduePDBInfoHasLabel name="hbr_S" property="hbnet_resn_S"/>
            <ResiduePDBInfoHasLabel name="hbr_T" property="hbnet_resn_T"/>
            <ResiduePDBInfoHasLabel name="hbr_W" property="hbnet_resn_W"/>
            <ResiduePDBInfoHasLabel name="hbr_Y" property="hbnet_resn_Y"/>
            <Or name="hbnet_res" selectors="hbr_R,hbr_N,hbr_D,hbr_Q,hbr_E,hbr_H,hbr_K,hbr_S,hbr_T,hbr_W,hbr_Y"/>

            <ResiduePDBInfoHasLabel name="rec_helix" property="ss_RH" />
            <ResiduePDBInfoHasLabel name="motif_res" property="MOTIF" />
            <Or name="rec_helix_or_motif_res" selectors="rec_helix,motif_res" />

            <ResiduePDBInfoHasLabel name="DONE_res" property="DONE" />
            <Index name="res1" resnums="1" />

            <ResidueName name="met" residue_name3="MET" />
            <ResidueName name="cys" residue_name3="CYS" />
            <Not name="not_cys" selector="cys"/>

            <!-- Layer Design -->
            <Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
            <Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
            <Layer name="core" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true"/>
            <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
            <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
            <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
            <And name="helix_cap" selectors="entire_loop">
                <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
            </And>
            <And name="helix_start" selectors="entire_helix">
                <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
            </And>
            <And name="helix" selectors="entire_helix">
                <Not selector="helix_start"/>
            </And>
            <And name="loop" selectors="entire_loop">
                <Not selector="helix_cap"/>
            </And>

        </RESIDUE_SELECTORS>

        <MOVERS>

            <AddCompositionConstraintMover name="3trp" >
                <Comp entry="PENALTY_DEFINITION;TYPE TRP;ABSOLUTE 0;PENALTIES 0 3;DELTA_START 0;DELTA_END 1;BEFORE_FUNCTION CONSTANT;AFTER_FUNCTION LINEAR;END_PENALTY_DEFINITION;" />
            </AddCompositionConstraintMover>

            <AddCompositionConstraintMover name="2met" >
                <Comp entry="PENALTY_DEFINITION;TYPE MET;ABSOLUTE 0;PENALTIES 0 2;DELTA_START 0;DELTA_END 1;BEFORE_FUNCTION CONSTANT;AFTER_FUNCTION LINEAR;END_PENALTY_DEFINITION;" />
            </AddCompositionConstraintMover>

            <AddCompositionConstraintMover name="limit_ser" >
                <Comp entry="PENALTY_DEFINITION;TYPE SER;ABSOLUTE 0;PENALTIES 0 7;DELTA_START 0;DELTA_END 1;BEFORE_FUNCTION CONSTANT;AFTER_FUNCTION LINEAR;END_PENALTY_DEFINITION;" />
            </AddCompositionConstraintMover>

            <AddCompositionConstraintMover name="limit_thr" >
                <Comp entry="PENALTY_DEFINITION;TYPE THR;ABSOLUTE 0;PENALTIES 0 7;DELTA_START 0;DELTA_END 1;BEFORE_FUNCTION CONSTANT;AFTER_FUNCTION LINEAR;END_PENALTY_DEFINITION;" />
            </AddCompositionConstraintMover>

        </MOVERS>

        <TASKOPERATIONS>
            <SeqprofConsensus name="pssm_cutoff" filename="{pssm_f}" keep_native="1" min_aa_probability="{pssmCut}" convert_scores_to_probabilities="0" probability_larger_than_current="0" debug="1" ignore_pose_profile_length_mismatch="1"/>
            <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
            <ProteinProteinInterfaceUpweighter name="upweight_interface" interface_weight="3" />
            <ProteinInterfaceDesign name="pack_long" design_chain1="1" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
            <IncludeCurrent name="current" />
            <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
            <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2aro="1" />


            <OperateOnResidueSubset name="restrict_target" selector="chainB">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="restrict_to_interface" selector="Not_interface">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="hbnet_prp" selector="hbnet_res">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="hbnet_rtr" selector="hbnet_res">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="restrict_rifres2repacking" selector="RIFRES">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="restrict_hotspots2repacking" selector="HOTSPOT_res">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="restrict_cores" selector="core">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="no_new_cys" selector="not_cys">
                <RestrictAbsentCanonicalAASRLT aas="ADEFGHIKLMNPQRSTVWY"/>
            </OperateOnResidueSubset>

            <DisallowIfNonnative name="disallow_GLY" resnum="0" disallow_aas="G" />
            <DisallowIfNonnative name="disallow_PRO" resnum="0" disallow_aas="P" />

        	  <OperateOnResidueSubset name="restrict_PRO_GLY" selector="pro_and_gly_positions">
        		    <PreventRepackingRLT/>
        	  </OperateOnResidueSubset>

            <SelectBySASA name="PR_monomer_core" mode="sc" state="monomer" probe_radius="2.2" core_asa="10" surface_asa="10" core="0" boundary="1" surface="1" verbose="0" />
            <InitializeFromCommandline name="init"/>
        </TASKOPERATIONS>

        <MOVERS>
            <SwitchChainOrder name="chain1onlypre" chain_order="1" />

            <ScoreMover name="scorepose" scorefxn="sfxn" verbose="false" />
            <ParsedProtocol name="chain1only">
                <Add mover="chain1onlypre" />
                <Add mover="scorepose" />
            </ParsedProtocol>

            <TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />

            <DeleteRegionMover name="delete_polar" residue_selector="polar" rechain="false" />

            <FastRelax name="relax_chain1" scorefxn="sfxn" repeats="1" batch="false" ramp_down_constraints="false" cartesian="false" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" task_operations="ex1_ex2,limitchi2" >
                <MoveMap name="MM" >
                    <Chain number="1" chi="true" bb="true" />
                </MoveMap>
            </FastRelax>

            <ParsedProtocol name="chain1only_relax">
                <Add mover="chain1onlypre" />
                <Add mover="relax_chain1" />
            </ParsedProtocol>

            <DeleteRegionMover name="delete_hydrophobic" residue_selector="apolar" rechain="false" />

            <ClearConstraintsMover name="clear_constraints" />


            <SavePoseMover name="save_output" restore_pose="0" reference_name="pose_output" />
            <SavePoseMover name="load_output" restore_pose="1" reference_name="pose_output" />

            <LabelPoseFromResidueSelectorMover name="remove_done" remove_property="DONE"  residue_selector="res1" />

            <LabelPoseFromResidueSelectorMover name="add_done" property="DONE" residue_selector="res1" />

        </MOVERS>
        <FILTERS>

            <Sasa name="interface_buried_sasa" confidence="0" />

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


            <SSPrediction name="pre_sspred_overall" cmd="/software/psipred4/runpsipred_single" use_probability="0" use_svm="0" threshold="0.85" confidence="0" />
            <MoveBeforeFilter name="sspred_overall" mover="chain1only" filter="pre_sspred_overall" confidence="0" />

            <SSPrediction name="pre_mismatch_probability" confidence="0" cmd="/software/psipred4/runpsipred_single" use_probability="1" mismatch_probability="1" use_svm="0" />
            <MoveBeforeFilter name="mismatch_probability" mover="chain1only" filter="pre_mismatch_probability" confidence="0" />

            <ScorePoseSegmentFromResidueSelectorFilter name="was_done" in_context="1" residue_selector="DONE_res" scorefxn="sfxn" confidence="0" />


            <SSShapeComplementarity name="ss_sc_pre" verbose="0" confidence="0" />
            <MoveBeforeFilter name="ss_sc" mover="chain1only" filter="ss_sc_pre" confidence="0"/>


            <Time name="timed"/>
            <TrueFilter name="my_true_filter" />
        </FILTERS>

        <JUMP_SELECTORS>
            <JumpIndex name="jump1" jump="1"/>
            <Not name="notjump1" selector="jump1"/>
        </JUMP_SELECTORS>

        <MOVE_MAP_FACTORIES>
            <MoveMapFactory name="mm_design" bb="0" chi="0" jumps="0">
                <Backbone enable="True" residue_selector="chainA" />
                <Chi enable="True" residue_selector="chainA" />
                <Jumps enable="False" jump_selector="notjump1" />
                <Jumps enable="True" jump_selector="jump1" />
            </MoveMapFactory>
        </MOVE_MAP_FACTORIES>

      <MOVERS>
          TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />
          <FavorSequenceProfile name="FSP" scaling="none" weight="1" pssm="{pssm_f}" scorefxns="sfxn_design"/>

          <FastRelax name="FastRelax"
            scorefxn="sfxn_relax"
            movemap_factory="mm_design"
            repeats="1"
            batch="false"
            ramp_down_constraints="false"
            cartesian="false"
            bondangle="false"
            bondlength="false"
            min_type="dfpmin_armijo_nonmonotone"
            task_operations="current,ex1_ex2,restrict_target,limitchi2" >
          </FastRelax>

          <ClearConstraintsMover name="rm_csts" />

          <PackRotamersMover name="hard_pack" scorefxn="sfxn_design_fast" task_operations="current,ex1_ex2,init,limitchi2,pssm_cutoff,no_new_cys,upweight_interface,restrict_to_interface,restrict_target{hbnet_restrictions}"/>
          <TaskAwareMinMover name="softish_min" scorefxn="sfxn_softish" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_to_interface,restrict_target"/>
          <TaskAwareMinMover name="hard_min" scorefxn="sfxn" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_to_interface,restrict_target"/>

          <FastDesign name="FastDesign"
                scorefxn="sfxn_design"
                movemap_factory="mm_design"
                repeats="3"
                task_operations="init,current,limitchi2,ex1_ex2,restrict_to_interface,restrict_target{hbnet_restrictions},no_new_cys,pssm_cutoff,upweight_interface,restrict_PRO_GLY"
                batch="false"
                ramp_down_constraints="false"
                cartesian="False"
                bondangle="false"
                bondlength="false"
                min_type="dfpmin_armijo_nonmonotone"
                relaxscript="/home/bcov/sc/scaffold_comparison/relax_scripts/no_ref.rosettacon2018.beta_nov16.txt"/>

          <SavePoseMover name="start_pose" restore_pose="0" reference_name="ref_start_pose"/>

      </MOVERS>

      <SIMPLE_METRICS>
          <SapScoreMetric name="sap_score" score_selector="chainA" />
          <SapScoreMetric name="sap_score_target" score_selector="chainB" />
          <SapScoreMetric name="binder_blocked_sap" score_selector="chainA" sap_calculate_selector="chainA" sasa_selector="true_sel" />
          <SapScoreMetric name="target_blocked_sap" score_selector="chainB" sap_calculate_selector="chainB" sasa_selector="true_sel" />

          <CalculatorMetric name="binder_delta_sap" equation="binder_sap_score - binder_blocked_sap" >
              <VAR name="binder_sap_score" metric="sap_score"/>
              <VAR name="binder_blocked_sap" metric="binder_blocked_sap"/>
          </CalculatorMetric>

          <CalculatorMetric name="target_delta_sap" equation="target_sap_score - target_blocked_sap" >
             <VAR name="target_sap_score" metric="sap_score_target"/>
             <VAR name="target_blocked_sap" metric="target_blocked_sap"/>
          </CalculatorMetric>

	      <TotalEnergyMetric name="pre_pssm_score" scorefxn="pssm_score"/>
	      <TotalEnergyMetric name="post_pssm_score" scorefxn="pssm_score"/>

          <SequenceRecoveryMetric name="seqrec" reference_name="ref_start_pose" residue_selector="chainA"/>
      </SIMPLE_METRICS>

    <PROTOCOLS>
         <Add filter="timed" />
         <Add mover="start_pose"/>
         <Add mover="limit_ser" />
         <Add mover="limit_thr" />
         <Add mover="FSP"/>
         <Add metrics="pre_pssm_score"/>

         {'<Add mover="hard_pack"/>'   if is_fast else ''}
         {'<Add mover="softish_min"/>' if is_fast else ''}
         {'<Add mover="hard_min"/>'    if is_fast else ''}
         {'<Add mover="FastDesign"/>'  if not is_fast else ''}
         {'<Add mover="FastRelax"/>'   if not is_fast else ''}

         <Add metrics="post_pssm_score"/>

         <Add mover="clear_constraints"/>

         <Add filter_name="interface_buried_sasa" />
         <Add filter_name="ddg" />
         <Add filter_name="interface_sc" />
         <Add filter="contact_molecular_surface" />
         {'<Add filter_name="score_per_res" />'           if not is_fast else ''}
         {'<Add filter_name="score_per_res_relax" />'     if not is_fast else ''}
         {'<Add filter="vbuns5.0_heavy_ball_1.1D" />'     if not is_fast else ''}
         {'<Add filter="sbuns5.0_heavy_ball_1.1D" />'     if not is_fast else ''}
         {'<Add filter="mismatch_probability" />'         if not is_fast else ''}
         {'<Add filter="sspred_overall" />'               if not is_fast else ''}
         {'<Add filter="cms_rec_helix" />'                if not is_fast else ''}
         {'<Add filter="ss_sc" />'                        if not is_fast else ''}
         {'<Add filter="total_sasa" />'                   if not is_fast else ''}
         {'<Add filter="net_charge" />'                   if not is_fast else ''}
         {'<Add filter="net_charge_over_sasa" />'         if not is_fast else ''}


         <Add filter="timed" />

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

def count_hbonds_protein_dna(pose) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making base-specific hydrogen bonds with DNA.
    Used in calculating the RotamerBoltzmann score (see calc_rboltz)
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

    protein_dna = 0
    protein_dna_specific = 0
    aa_identities = []
    aa_pos = []
    bidentate_list = []

    for residue in range(len(hbonds_acc_res)) :
        don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
        acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")

        if pose.residue(hbonds_don_res[residue]).name().split(':')[0] in DNA_base_names :
            if not pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                protein_dna += 1
                if not don_atom_type in mc_bb_atoms and not acc_atom_type in aa_bb_atoms:
                    print("We've got a protein-DNA interaction")
                    protein_dna_specific += 1
                    aa_identities.append(pose.residue(hbonds_acc_res[residue]).name1())
                    if not int(hbonds_acc_res[residue]) in aa_pos:
                        aa_pos.append(int(hbonds_acc_res[residue]))
                    else:
                        bidentate_list.append(int(hbonds_acc_res[residue]))
                    base_id = dna_base_list[dna_res_list.index(hbonds_don_res[residue])]
                    base_count_dict[base_id] += 1
        else :
            if pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                protein_dna += 1
                if not acc_atom_type in mc_bb_atoms and not don_atom_type in aa_bb_atoms:
                    protein_dna_specific += 1
                    aa_identities.append(pose.residue(hbonds_don_res[residue]).name1())
                    if not int(hbonds_don_res[residue]) in aa_pos:
                        aa_pos.append(int(hbonds_don_res[residue]))
                    else:
                        bidentate_list.append(int(hbonds_don_res[residue]))
                    base_id = dna_base_list[dna_res_list.index(hbonds_acc_res[residue])]
                    base_count_dict[base_id] += 1

    return aa_pos, list(set(bidentate_list))

def calc_rboltz(pose, df):
    '''
    Takes in a pose and the existing DataFrame of scores for a design and returns the DataFrame with
    three new columns: largest RotamerBoltzmann of ARG/LYS/GLU/GLN residues; average of the top two
    RotamerBoltzmann scores (includes every amino acid type); and median RotamerBoltzmann (includes every amino acid type)
    '''
    notable_aas = ['ARG','GLU','GLN','LYS']
    hbond_residues, bidentates = count_hbonds_protein_dna(pose)

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

def run_pose(pose, opts, dfs, sfd_out=None, silent_tag='', rand_string=''):

    # define output and output directory
    if opts.outpdb != '':
        out_pdb = opts.outpdb
        outdir = os.path.dirname(out_pdb)
    elif opts.silent_out != '':
        silent_out = opts.silent_out
        outdir = os.path.dirname(silent_out)

    pssmCut = opts.pssmCut
    weight_file = opts.weights
    pssm_weight = opts.pssm_weight
    out_pdb = opts.outpdb


    protein_len = pose.total_residue()

    # =============================================================================
    #                              Run prefiltering
    # =============================================================================
    t0 = time.time()
    ## First check if rifres in recognition helix. If not, quit.
    rif_resis = []
    rec_helix = []
    motif = []
    info = pose.pdb_info()
    for res in range(1,pose.total_residue()+1):
        reslabel = info.get_reslabels(res)
        if 'RIFRES' in reslabel:
            rif_resis.append(res)
        if 'ss_RH' in reslabel:
            rec_helix.append(res)
        if 'MOTIF' in reslabel:
            motif.append(res)

    # I did this because the rec helix in 1L3L and 1PER topologies are annotated ss_H3
    if len(rec_helix) == 0:
        for res in range(1,pose.total_residue()+1):
            reslabel = info.get_reslabels(res)
            if 'ss_H3' in reslabel:
                rec_helix.append(res)

    rifres_in_rec_helix = False
    for rif_resi in rif_resis:
        if rif_resi in rec_helix:
            rifres_in_rec_helix = True

    motif_in_rec_helix = False
    if len(rec_helix) > 0 and len(motif) > 0:
        if (min(rec_helix) + 1 == min(motif)) or (max(rec_helix) -1 == max(motif)):
            motif_in_rec_helix = True

    if rifres_in_rec_helix == False and opts.require_rifres_in_rec_helix == 1:
        print(f'{out_pdb}: Rifres not in recognition helix. Quitting')
        return
    elif rifres_in_rec_helix == True and opts.require_rifres_in_rec_helix == 1:
        print('rifres is in recognition helix')
    elif motif_in_rec_helix == False and opts.require_motif_in_rec_helix == 1:
        print(f'{out_pdb}: Motif not in recognition helix. Quitting')
        return
    elif motif_in_rec_helix == True and opts.require_motif_in_rec_helix == 1:
        print('motif is in recognition helix')

    ##
    is_fast = True
    xml_prefilter = make_binder_design_xml(weight_file = weight_file, pssm_weight=pssm_weight, \
                                      pssmCut=pssmCut, pssm_f = opts.pssmFile, hbnet_restrictions=opts.hbnet_restrictions, is_fast = is_fast)
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_prefilter)
    task_relax.setup() # syntax check

    pre_pose = pose.clone()
    packed_pose = task_relax(pre_pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])
    df_scores['is_prefilter'] = True

    if opts.outpdb == '':
        out_tag = silent_tag + '_' + opts.rand_string

    if opts.pdb != '':
        out_tag = os.path.basename(opts.pdb)[:-len('.pdb')]

    df_scores['out_tag'] = out_tag

    df_scores['dPSSM'] = df_scores['pre_pssm_score'] - df_scores['post_pssm_score'] # lower better

    df_scores = hbond_score(pre_pose, out_tag, df_scores, opts)
    if opts.rboltz_in_predictor == 1:
        df_scores = calc_rboltz(pre_pose, df_scores)

    # ---------------------------------------------------------------------
    #              Preemption based on calibrated prefilters
    # ---------------------------------------------------------------------
    def get_first_line(f):
       with open(f,'r') as f_open:
           eq = [l for l in f_open][0]
       return eq

    prefilter_eq_file = opts.prefilter_eq_file
    prefilter_mle_cut_file = opts.prefilter_mle_cut_file
    if prefilter_eq_file is not None and prefilter_mle_cut_file is not None:
        eq = get_first_line(prefilter_eq_file)[1:-2]
        possible_features = df_scores.columns
        for f in possible_features:
            eq = eq.replace(f' {f} ', f' {df_scores[f].iloc[0]} ') # KEEP THE SPACES HERE!!

        EXP = np.exp
        log_prob = np.log10(-eval(eq)) # WHY is the equation being dumped with a - in front of?!
        log_prob_cut = float(get_first_line(prefilter_mle_cut_file))

        # Store relevant information
        df_scores['log_prob_cut'] = log_prob_cut
        df_scores['log_prob'] = log_prob
        for key in opts.__dict__:
            df_scores[key] = str(opts.__dict__[key])
        # df_scores.to_csv(out_csv_f)

        if log_prob<log_prob_cut:
            print(f'{out_pdb}: Did not pass prefilters. Quitting')
            dfs.append(df_scores)
            return dfs
        print(f"{out_pdb}: Passed prefilters. Continuing")

    # Add setting info to the score file
    for key in opts.__dict__:
        df_scores[key] = opts.__dict__[key]
    dfs.append(df_scores)
    t1 = time.time()
    print(f'prefiltering took {t1-t0}')

    # =============================================================================
    #                            Run full-scale design
    # =============================================================================

    print("Running full scale design")
    t0 = time.time()
    is_fast = False
    xml_full = make_binder_design_xml(weight_file = weight_file, pssm_weight=pssm_weight, pssmCut=pssmCut, pssm_f = opts.pssmFile, \
                                        hbnet_restrictions=opts.hbnet_restrictions, is_fast=is_fast)
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_full)
    task_relax.setup() # syntax check
    packed_pose = task_relax(pose)
    df_scores = pd.DataFrame.from_records([packed_pose.scores])
    df_scores['is_prefilter'] = False
    df_scores['out_tag'] = out_tag
    df_scores['dPSSM'] = df_scores['pre_pssm_score'] - df_scores['post_pssm_score'] # lower better

    # Add setting info to the pose and the score file
    for key in opts.__dict__:
        core.pose.add_comment(pose, str(key), str(opts.__dict__[key]))
        df_scores[key] = str(opts.__dict__[key])

    if opts.outpdb != '':
        pose.dump_pdb(out_pdb)
    elif opts.silent_out != '' and opts.silent_tag != '':
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

    # Calculate RotamerBoltzmann scores
    df_scores = calc_rboltz(pose, df_scores)

    dfs.append(df_scores)


    t1 = time.time()
    print(f'full-scale design for took {t1-t0}')

    return dfs

def id_generator(size=12, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

def main():
    t0 = time.time()
    #========================================================================
    # Build option parser
    #========================================================================
    parser = OptionParser(usage="usage: %prog [options] FILE", version="0.1")

    # Pose
    parser.add_option("--pdb", type="string", dest="pdb", metavar="STR", default = '', help="Path to input pdb file")
    parser.add_option("--silent_in", type="string", dest="silent_in", metavar="STR", default = '', help="Path to input silent file")
    parser.add_option("--silent_tag", type="string", dest="silent_tag", metavar="STR", default = '', help="Input silent tag")
    parser.add_option("--rand_string", type="string", dest="rand_string", metavar="STR", default = '', help="Random string")

    parser.add_option("--pssmFile", type="string", dest="pssmFile", metavar="STR", help="Path to pssm file")

    parser.add_option("--pssmCut", type="string", dest="pssmCut", metavar="STR", help="PSSM cutoff for amino acids")
    parser.add_option("--pssm_weight", type="string", dest="pssm_weight", metavar="STR", help="PSSM weight for amino acids")

    parser.add_option("--outpdb", type="string", dest="outpdb", metavar="STR", default = '', help="path to output pdb")
    parser.add_option("--silent_out", type="string", dest="silent_out", metavar="STR", default = '', help="path to output silent file")

    parser.add_option("--only_design_surface", type="string", dest="only_design_surface", default='1', metavar="STR", help="only design surface residues?")

    # restraints
    parser.add_option("--hbnet_restrictions", type="string", dest="hbnet_restrictions", default='', metavar="STR", help="Constain identities of all hbnet residues")

    # score function
    parser.add_option("--weights", type="string", dest="weights", metavar="STR", default = '', help="Path to score function weights")
    parser.add_option("--flags_file", type="string", dest="flags_file", metavar="STR", default = '', help="Path to score function flags file")

    # Prefilter
    parser.add_option("--require_rifres_in_rec_helix", type="int", dest="require_rifres_in_rec_helix", default=0, metavar="STR", help="1 = Require rifres in rec helix")
    parser.add_option("--require_motif_in_rec_helix", type="int", dest="require_motif_in_rec_helix", default=0, metavar="STR", help="1 = Require motif in rec helix")
    parser.add_option("--prefilter_eq_file", type="string", dest="prefilter_eq_file", metavar="STR", help="Sigmoid equation for prefiltering")
    parser.add_option("--prefilter_mle_cut_file", type="string", dest="prefilter_mle_cut_file", metavar="STR", help="MLE cutoff for prefiltering")

    # Metric settings
    parser.add_option("--rboltz_in_predictor", type="int", dest="rboltz_in_predictor", default=0, metavar="STR", help="1 = Calculate rboltz in predictor.")
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
        flags = '/home/cjg263/de_novo_dna/flags/flags_47D.460'

    init(f'-ignore_zero_occupancy {silent_in_init} {silent_out_init} -beta -output_virtual 1 -precompute_ig 1 -mute all \
           -holes:dalphaball /home/norn/software/DAlpahBall/DAlphaBall.gcc @{flags}')


    ## Load input pose or silent file
    if opts.pdb != '':
        in_pdb = opts.pdb
        pose = pyrosetta.pose_from_file(in_pdb)
    elif opts.silent_in != '':
        sfd = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
        sfd.read_file(opts.silent_in)
        if opts.silent_tag != '' and opts.silent_tag in sfd.tags():
            ss = sfd.get_structure(opts.silent_tag)
            pose = Pose()
            ss.fill_pose(pose)
    else:
        print('No specified input file. Aborting job.')
        sys. exit

    # define output and output directory
    if opts.outpdb != '':
        out_pdb = opts.outpdb
        outdir = os.path.dirname(out_pdb)
    elif opts.silent_out != '':
        silent_out = opts.silent_out
        outdir = os.path.dirname(silent_out)
    else:
        print('No specified output file. Aborting job.')
        sys. exit
    os.makedirs(outdir, exist_ok=True)

    # Define output csv file
    if opts.outpdb != '':
        out_csv_f = out_pdb[:-4] + '.csv'
    else:
        out_csv_f = silent_out.replace('.silent','.csv')

    if os.path.isfile(out_csv_f):
        csv_done = pd.read_csv(out_csv_f)
        csv_is_file = True
    else: csv_is_file = False

    dfs = []
    if ( opts.pdb != '' )  or ( opts.silent_in != '' and opts.silent_tag != '' ):
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
                dfs = run_pose(pose, opts, dfs, sfd_out, silent_tag)
                scores = pd.concat(dfs, axis=0, ignore_index=True)
                scores.to_csv(out_csv_f)

if __name__ == '__main__':
    main()
