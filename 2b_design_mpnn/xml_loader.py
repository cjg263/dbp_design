import pyrosetta

def fast_relax(protocols,weights_file,psipred_exe):
    xml_script = """
    <ROSETTASCRIPTS>
        # This is the relax script that Cameron adapted from the af2 superposition relax (originally written by Nate)
                <SCOREFXNS>
                        <ScoreFunction name="sfxn" weights="/home/cjg263/de_novo_dna/weights_files/beta16_opt47D.460_torsional.wts" />
                        <ScoreFunction name="sfxn_design" weights="/home/cjg263/de_novo_dna/weights_files/beta16_opt47D.460_torsional.wts" >
                                <Reweight scoretype="arg_cation_pi" weight="3" />
                        </ScoreFunction>

                        <ScoreFunction name="sfxn_design_fast" weights="/home/cjg263/de_novo_dna/weights_files/beta16_opt47D.460_torsional.wts">
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


                        <ScoreFunction name="sfxn_softish" weights="/home/cjg263/de_novo_dna/weights_files/beta16_opt47D.460_torsional.wts" >
                <Reweight scoretype="arg_cation_pi" weight="3"/>
                <Reweight scoretype="fa_rep" weight="0.15" />
                <Reweight scoretype="pro_close" weight="0"/>
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
        <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1" extrachi_cutoff="12"/>
                                <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
                                <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />

              <OperateOnResidueSubset name="restrict_to_chA" selector="chainB">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>

        <OperateOnResidueSubset name="restrict_to_interface" selector="Not_interface">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>

                                <OperateOnResidueSubset name="restrict_all_to_repacking" selector="chainA">
            <RestrictToRepackingRLT/>
        </OperateOnResidueSubset>

        <OperateOnResidueSubset name="restrict_target" selector="chainB">
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

                                <Ddg name="ddg"  threshold="-10" jump="1" repeats="3" repack="1" relax_mover="min" confidence="0" scorefxn="sfxn" extreme_value_removal="1" />

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

                                <SSPrediction name="pre_mismatch_probability" confidence="0" cmd="{psipred_exe}" use_probability="1" mismatch_probability="1" use_svm="0" />
                                <MoveBeforeFilter name="mismatch_probability" mover="chain1only" filter="pre_mismatch_probability" confidence="0" />



                                <SSShapeComplementarity name="ss_sc_pre" verbose="0" confidence="0" />

                                <MoveBeforeFilter name="ss_sc" mover="chain1only" filter="ss_sc_pre" confidence="0"/>
    </FILTERS>

    <JUMP_SELECTORS>
        <JumpIndex name="jump1" jump="1"/>
        <Not name="notjump1" selector="jump1"/>
    </JUMP_SELECTORS>

    <MOVE_MAP_FACTORIES>
        <MoveMapFactory name="mm_relax" bb="0" chi="0" jumps="0">
            <Backbone enable="True" residue_selector="chainA" />
            <Chi enable="True" residue_selector="chainA" />
            <Jumps enable="False" jump_selector="notjump1" />
            <Jumps enable="True" jump_selector="jump1" />
        </MoveMapFactory>

    </MOVE_MAP_FACTORIES>

    <MOVERS>
                        <PackRotamersMover name="pack_no_design" scorefxn="sfxn_design" task_operations="current,ex1_ex2aro,init,limitchi2,restrict_all_to_repacking,restrict_target"/>
                        <TaskAwareMinMover name="softish_min" scorefxn="sfxn_softish" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="init,restrict_all_to_repacking,restrict_target"/>
                        <TaskAwareMinMover name="hard_min" scorefxn="sfxn_design" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="init,restrict_all_to_repacking,restrict_target"/>

                <FastRelax name="FastRelax"
                                                                        scorefxn="sfxn_design"
                                                                        movemap_factory="mm_relax"
                                                                        repeats="1"
                                                                        batch="false"
                                                                        ramp_down_constraints="false"
                                                                        cartesian="false"
                                                                        bondangle="false"
                                                                        bondlength="false"
                                                                        min_type="dfpmin_armijo_nonmonotone"
                                                                        task_operations="init,current,restrict_to_chA,ex1_ex2"
                                                                        relaxscript="/home/nrbennet/protocols/relax_scripts/no_ref.rosettacon2018.beta_nov16_constrained.txt" >
      </FastRelax>

    </MOVERS>

    <APPLY_TO_POSE>
    </APPLY_TO_POSE>
    <PROTOCOLS>
        <Add mover="pack_no_design"/>
        <Add mover="softish_min"/>
        <Add mover="hard_min"/>
        <Add mover="FastRelax" />
        <Add filter_name="interface_buried_sasa" />
        <Add filter_name="ddg" />
        <Add filter_name="interface_sc" />
        <Add filter_name="score_per_res" />
        <Add filter_name="score_per_res_relax" />
        <Add filter="vbuns5.0_heavy_ball_1.1D" />
        <Add filter="sbuns5.0_heavy_ball_1.1D" />
        <Add filter="mismatch_probability" />
        <Add filter="sspred_overall" />
        <Add filter="contact_molecular_surface" />
        <Add filter="ss_sc" />
        <Add filter="total_sasa" />
        <Add filter="net_charge" />
        <Add filter="net_charge_over_sasa" />

    </PROTOCOLS>
</ROSETTASCRIPTS>
"""
    xml_fr = xml_script.format(weights_file=weights_file,psipred_exe=psipred_exe)
    objs = protocols.rosetta_scripts.XmlObjects.create_from_string(xml_script)
    pack_no_design = objs.get_mover('pack_no_design')
    softish_min = objs.get_mover('softish_min')
    hard_min = objs.get_mover('hard_min')
    fast_relax = objs.get_mover('FastRelax')

    ddg_filter = objs.get_filter( 'ddg' )
    if ( isinstance(ddg_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        ddg_filter = ddg_filter.subfilter()

    cms_filter = objs.get_filter( 'contact_molecular_surface' )
    if ( isinstance(cms_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        cms_filter = cms_filter.subfilter()

    vbuns_filter = objs.get_filter( 'vbuns5.0_heavy_ball_1.1D' )
    if ( isinstance(vbuns_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        vbuns_filter = vbuns_filter.subfilter()

    sbuns_filter = objs.get_filter( 'sbuns5.0_heavy_ball_1.1D' )
    if ( isinstance(sbuns_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        sbuns_filter = sbuns_filter.subfilter()

    net_charge_filter = objs.get_filter( 'net_charge' )
    if ( isinstance(net_charge_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        net_charge_filter = net_charge_filter.subfilter()

    net_charge_over_sasa_filter = objs.get_filter( 'net_charge_over_sasa' )
    if ( isinstance(net_charge_over_sasa_filter, pyrosetta.rosetta.protocols.filters.StochasticFilter) ):
        net_charge_over_sasa_filter = net_charge_over_sasa_filter.subfilter()
    
    return pack_no_design, softish_min, hard_min, fast_relax, ddg_filter, cms_filter, vbuns_filter, sbuns_filter, net_charge_filter, net_charge_over_sasa_filter

def fast_design_interface(protocols, weights_file, pssmFile, fixed_positions_dict):
    xml_script = """
    <ROSETTASCRIPTS>
                <SCOREFXNS>
                        <ScoreFunction name="sfxn" weights="{weights_file}" />
                        <ScoreFunction name="sfxn_design" weights="{weights_file}" >
                                <Reweight scoretype="arg_cation_pi" weight="3" />
                    <Reweight scoretype="atom_pair_constraint" weight="0.3"/>
                    <Reweight scoretype="res_type_constraint" weight="0.5"/>
                        </ScoreFunction>

                </SCOREFXNS>

        <TASKOPERATIONS>
            <SelectBySASA name="PR_monomer_core" mode="sc" state="monomer" probe_radius="2.2" core_asa="15" surface_asa="15" core="0" boundary="1" surface="1" verbose="0" />
        </TASKOPERATIONS>

        <RESIDUE_SELECTORS>
            <Chain name="chainA" chains="1"/>
            <Not name="chainB" selector="chainA"/>
            <Neighborhood name="interface_chA" selector="chainB" distance="9.0" />
            <Neighborhood name="interface_chB" selector="chainA" distance="9.0" />
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
            <ResidueName name="cys" residue_name3="CYS" />
            <Not name="not_cys" selector="cys"/>

            <ResidueName name="pro_and_gly_positions" residue_name3="PRO,GLY" />

            <And name="chainB_fixed" >
                <Or selectors="chainB_Not_interface,water" />
            </And>
            <And name="chainB_not_fixed" selectors="chainB">
                <Not selector="chainB_fixed"/>
            </And>

                                <ResiduePDBInfoHasLabel name="rec_helix" property="ss_RH" />
            <ResiduePDBInfoHasLabel name="motif_res" property="MOTIF" />
            <Or name="rec_helix_or_motif_res" selectors="rec_helix,motif_res" />

            <Index name="fixed_positions" resnums="{fixed_positions}"/>

        </RESIDUE_SELECTORS>

<TASKOPERATIONS>
            <SeqprofConsensus name="pssm_cutoff" filename="{pssm_f}" keep_native="1" min_aa_probability="-1" convert_scores_to_probabilities="0" probability_larger_than_current="0" debug="1" ignore_pose_profile_length_mismatch="1"/>
            <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
            <InitializeFromCommandline name="init" />
            <IncludeCurrent name="current" />
            <ExtraRotamersGeneric name="ex1_ex2" ex1="1" ex2="1" />
                                <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />

            <OperateOnResidueSubset name="prevent_fixed_positions" selector="fixed_positions">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>

                <OperateOnResidueSubset name="restrict_to_chA" selector="chainB">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="restrict_to_interface" selector="Not_interface">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="restrict_target" selector="chainB">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="no_new_cys" selector="not_cys">
                <RestrictAbsentCanonicalAASRLT aas="ADEFGHIKLMNPQRSTVWY"/>
            </OperateOnResidueSubset>

            <ProteinProteinInterfaceUpweighter name="upweight_interface" interface_weight="3" />

            <OperateOnResidueSubset name="restrict_PRO_GLY" selector="pro_and_gly_positions">
                            <PreventRepackingRLT/>
                  </OperateOnResidueSubset>

        </TASKOPERATIONS>

        <JUMP_SELECTORS>
            <JumpIndex name="jump1" jump="1"/>
            <Not name="notjump1" selector="jump1"/>
        </JUMP_SELECTORS>

        <MOVE_MAP_FACTORIES>
            <MoveMapFactory name="mm_design" bb="0" chi="0" jumps="0">
                <Backbone enable="False" residue_selector="chainA" />
                <Chi enable="True" residue_selector="chainA" />
                <Jumps enable="False" jump_selector="notjump1" />
                <Jumps enable="False" jump_selector="jump1" />
            </MoveMapFactory>
        </MOVE_MAP_FACTORIES>

 <MOVERS>
            <FavorSequenceProfile name="FSP" scaling="none" weight="1" pssm="{pssm_f}" scorefxns="sfxn_design"/>

            <AddConstraintsToCurrentConformationMover name="add_ca_csts" use_distance_cst="1" coord_dev="0.5" min_seq_sep="1" max_distance="9.0" cst_weight="1.0" CA_only="1" bb_only="0"/>

            <FastDesign name="FastDesign"
                       scorefxn="sfxn_design"
                       movemap_factory="mm_design"
                       repeats="1"
                       task_operations="init,current,limitchi2,ex1_ex2,restrict_to_interface,restrict_target,prevent_fixed_positions,no_new_cys,pssm_cutoff,upweight_interface,restrict_PRO_GLY"
                       batch="false"
                       ramp_down_constraints="false"
                       cartesian="False"
                       bondangle="false"
                       bondlength="false"
                       min_type="dfpmin_armijo_nonmonotone"
                       relaxscript="/home/bcov/sc/scaffold_comparison/relax_scripts/no_ref.rosettacon2018.beta_nov16.txt"/>

            <ClearConstraintsMover name="rm_csts" />

        </MOVERS>

        <APPLY_TO_POSE>
        </APPLY_TO_POSE>
        <PROTOCOLS>
            <Add mover="add_ca_csts"/>
            <Add mover="FSP"/>
            <Add mover="FastDesign" />
            <Add mover="rm_csts"/>

        </PROTOCOLS>
    </ROSETTASCRIPTS>
    """
    print(fixed_positions_dict)
    string_ints = [str(int) for int in fixed_positions_dict["tmp"]["A"]]
    fixed_positions = ','.join(string_ints)
    print(fixed_positions)
    xml_fd = xml_script.format(weights_file=weights_file, pssm_f=pssmFile, fixed_positions=fixed_positions)

    task_relax = protocols.rosetta_scripts.XmlObjects.create_from_string(xml_script)
    add_ca_csts = task_relax.get_mover("add_ca_csts")
    FSP = task_relax.get_mover("FSP")
    FastDesign = task_relax.get_mover("FastDesign")
    rm_csts = task_relax.get_mover("rm_csts")

    return task_relax, add_ca_csts, FSP, FastDesign, rm_csts

