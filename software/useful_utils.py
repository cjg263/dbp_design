import glob
import sys,os,json
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score




def collect_results(directory,out_pattern, combined_csv, combined_silent):
    # This code filters the out.csv files to only contain data for relaxed structures
    # This makes the next section faster since it only has to load the relaxed data into the dataframe.

    csv_fs = glob.glob(f'{directory}/{out_pattern}/out.csv')
    with open(csv_fs[0],'r') as f_in:
        column_line = f_in.readlines()[0].rstrip()
    total_predictions = 0
    passing_predictions = 0
    print(f"{len(csv_fs)} csvs were saved.")
    for csv in csv_fs:
        passed_csv = csv.replace('out.csv','prefilter.csv')
        with open(csv,'r') as f_in:
            with open(passed_csv,'w') as f_out:
                lines = [line.strip() for line in f_in.readlines()]
                try:
                    assert lines[0] == column_line # Trying to make sure columns are in correct order
                    f_out.write(lines[0]+'\n')
                    is_prefilter_col = lines[0].split(',').index('is_prefilter') if 'is_prefilter' in lines[0] else None
                    for line in lines[1:]:
                        total_predictions += 1
                        if line.split(',')[is_prefilter_col] == 'False':
                            passing_predictions += 1
                            f_out.write(line+'\n')
                except: continue
    print(f"{total_predictions} MPNN predictions were made, with {round(passing_predictions/total_predictions*100,2)}% ({passing_predictions}) passing prefilters.")
                       
    # This code collects data from passed.csv files. It is ugly but works.

    csv_fs = glob.glob(f'{directory}/{out_pattern}/out.csv')
    total_predictions = 0
    unexpected_length = 0

    out_csv = f'{directory}/{combined_csv}'
    with open(out_csv, 'w') as f_out:
        expected_col_lens = 0
        for csv in csv_fs[:]:
            with open(csv, 'r') as f_in:
                lines = f_in.readlines()
                try: columns = lines[0].split(',')
                except: continue
                #print(columns)
                if len(columns) > expected_col_lens:
                    expected_columns = lines[0]
                    expected_col_lens = len(columns)
        f_out.write(expected_columns)
        
        is_prefilter_column = columns.index('is_prefilter')

        for csv in csv_fs:
            with open(csv,'r') as f_in:
                lines = f_in.readlines()
                try: columns = lines[0].split(',')
                except: continue
                if len(columns) != expected_col_lens:
                    unexpected_length += 1
                    continue
                # print(csv, len(lines))
                for line in lines[1:]:
                    #line_cols = line.split(',')
                    if '"' in line:  # This became necessary because 'freeze_resis' column could have its own commas...I wonder if we've thrown other stuff out because of this.
                        line_cols_temp = line.split('"')
                        line_cols = line_cols_temp[0][:-1].split(',') + [line_cols_temp[1].replace(",","")] + line_cols_temp[2][1:].split(',')
                    else:
                        line_cols = line.split(',')
                    if len(line_cols) != expected_col_lens:
                        unexpected_length += 1
                        break
                    if line_cols[is_prefilter_column] == 'False':
                        total_predictions += 1
                        f_out.write(line)

    print(f"{total_predictions} relaxed models were produced. {unexpected_length} files had unexpected column length. ")
    
    # Save a silent file containing all relaxed structures

    silents = glob.glob(f'{directory}/{out_pattern}/out.*ilent')
    print(f'{len(silents)} silent files were produced.')
    silent_out = f'{directory}/{combined_silent}'
    with open(silent_out, 'w') as f_out:
        for silent in silents:
            with open(silent,'r') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    f_out.write(line)

                    
def collect_results_no_prefilter(directory,out_pattern, combined_csv, combined_silent):
    # This code filters the out.csv files to only contain data for relaxed structures
    # This makes the next section faster since it only has to load the relaxed data into the dataframe.

    csv_fs = glob.glob(f'{directory}/{out_pattern}/out.csv')
    total_predictions = 0
    passing_predictions = 0
    print(f"{len(csv_fs)} csvs were saved.")
                       
    # This code collects data from passed.csv files. It is ugly but works.

    csv_fs = glob.glob(f'{directory}/{out_pattern}/out.csv')
    total_predictions = 0
    unexpected_length = 0

    out_csv = f'{directory}/{combined_csv}'
    with open(out_csv, 'w') as f_out:
        expected_col_lens = 0
        for csv in csv_fs[:5]:
            with open(csv, 'r') as f_in:
                lines = f_in.readlines()
                columns = lines[0].split(',')
                if len(columns) > expected_col_lens:
                    expected_columns = lines[0]
                    expected_col_lens = len(columns)
        f_out.write(expected_columns)

        for csv in csv_fs:
            with open(csv,'r') as f_in:
                lines = f_in.readlines()
                try:
                    columns = lines[0].split(',')
                except: continue
                if len(columns) != expected_col_lens:
                    unexpected_length += 1
                    continue

                for line in lines[1:]:
                    total_predictions += 1
                    f_out.write(line)
    print(f"{total_predictions} relaxed models were produced. {unexpected_length} files had unexpected column length. ")
    
    # Save a silent file containing all relaxed structures

    silents = glob.glob(f'{directory}/{out_pattern}/out*.silent')
    print(f'{len(silents)} silent files were produced.')
    silent_out = f'{directory}/{combined_silent}'
    with open(silent_out, 'w') as f_out:
        for silent in silents:
            with open(silent,'r') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    f_out.write(line)
                    
                    

# This function filters the resulting outputs. You may play with filters to get a desired number of filtered outputs that look reasonable to you.
# You may change fraction not passing to allow some number of designs that are below the filter thresholds. 
# This is useful for testing if filters are helping in experiments.

def filter_designs(directory, df, threshold_dict, fraction_not_passing, seqid_cut, select_by='ddg_over_cms', minimize=True, use_current_clusters=False):

    terms_and_cuts = {}
    for j in threshold_dict.keys():
        if threshold_dict[j][1] == '>=' or threshold_dict[j][1] == '>':
            terms_and_cuts[j] = [threshold_dict[j][0],True]
        else:
            terms_and_cuts[j] = [threshold_dict[j][0],False]

    # ------------------------------------
    #        Find margins for probabilistic filtering
    # ------------------------------------
    for term in terms_and_cuts:
        higher_better = terms_and_cuts[term][1]
        df[term+'_rank'] = df[term].rank(ascending = False if higher_better else True).apply(np.floor)

    terms_and_prob_shapes = {}
    bad_percentile_rank = int(len(df)*0.98)
    print(f'{len(df)} designs passing xml prefilters\n')


    # Name of the score in pilot, cut value, higher better, name in predictor, is integer score-term
    print('filter hard_cut low_soft_cut_limit')
    print('----------------------------------')
    for term in terms_and_cuts:
        cut = terms_and_cuts[term][0]
        higher_better = terms_and_cuts[term][1]
        if higher_better:
            cut_rank = float(df[df[term]>=cut].sort_values(by = term+'_rank').tail(1)[term+'_rank'])
        else:
            cut_rank = float(df[df[term]<=cut].sort_values(by = term+'_rank').tail(1)[term+'_rank'])

        proposed_margin_rank = cut_rank + bad_percentile_rank
        cut_added_margin_rank = proposed_margin_rank if proposed_margin_rank<len(df) else len(df)    
        added_margin_cut = float(df[df[term+'_rank']<=cut_added_margin_rank].sort_values(by = term+'_rank').tail(1)[term])

        # For integer value features we will add at least +2
        is_integer = terms_and_cuts[term][-1]
        if is_integer:
            if np.abs(added_margin_cut-cut)<=2:
                added_margin_cut = cut-2 if higher_better else cut+2

        #Add info
        terms_and_prob_shapes[term] = [cut, added_margin_cut]

        print(term, cut, round(added_margin_cut,3))


    # Filter all the terms and print the thresholds
    print('\nHard cuts used to select orderable and setup the fuzzy cutoffs')
    print('--------------------------------------------------------------')
    ok_terms = []
    for pilot_term in terms_and_cuts:
        cut, good_high= terms_and_cuts[pilot_term]
        ok_term = pilot_term.replace("_pilot", "") + "_ok"
        if ( good_high ):
            df[ok_term] = df[pilot_term] >= cut
        else:
            df[ok_term] = df[pilot_term] <= cut

        ok_terms.append(ok_term)

        print("%30s: %6.2f"%(pilot_term, cut))

    # Print the pass rates for each term
    print("")
    print('number of designs passing hard cuts')
    df['orderable'] = True
    for ok_term in ok_terms:
        df['orderable'] &= df[ok_term]
        print("%30s: %5.0f%% pass-rate"%(ok_term.replace("_ok", ""), df[ok_term].sum() / len(df) * 100))

    # print the overall pass rate   
    subdf = df
    print('')
    print("Orderable: %i   -- %.2f%%"%(subdf['orderable'].sum(), (100*subdf['orderable'].sum() / len(subdf))))
    #n_orderable_plaits = subdf[subdf['is_plait']]['orderable'].sum()
    #print(f'Of these {n_orderable_plaits} plaits and {subdf["orderable"].sum() - n_orderable_plaits} NTF2s')

    # --------------------------- Plot and compare to natives
    relevant_features  = terms_and_cuts.keys()
    ncols = 3
    nrows = math.ceil(len(relevant_features) / ncols)
    (fig, axs) = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=[15,3*nrows]
    )
    axs = axs.reshape(-1)

    for (i, metric) in enumerate(terms_and_cuts):
        # Plot distributions
        #ax = sns.distplot(df[metric], ax=axs[i], color='blue', label='no filter', norm_hist=True)
        #if metric in df_natives.columns:
        #    sns.distplot(df_natives[metric], ax=axs[i], color='green', label='native', norm_hist=True)

        # plot the probability over the feature
        xs = sorted(list(set(df[metric].sample(frac=0.1))))
        max_score = terms_and_prob_shapes[metric][0]
        min_score =  terms_and_prob_shapes[metric][1]
        is_higher_better = terms_and_cuts[metric][1]

        if is_higher_better:
            prob = np.interp(xs, [min_score, max_score], [0.4, 1.0])
            df[metric+'_prob'] = np.power(np.interp(df[metric], [min_score, max_score], [0.4, 1.0]),3)
        else:
            prob = np.interp(xs, [max_score, min_score], [1.0, 0.4])
            df[metric+'_prob'] = np.power(np.interp(df[metric], [max_score, min_score], [1.0, 0.4]),3)

        prob = np.power(prob,3)
        ax2 = axs[i].twinx()
        #sns.lineplot(xs,prob, ax=ax2, color='red')

    # Compute logPs and get top x
    df['prob'] = 0
    for term in terms_and_cuts:
        df['prob'] = df['prob'] + df[term+'_prob']
    df_orderable = df[df['orderable']]

    DESIGNS_FILTERED_by_hard_cuts = len(df_orderable)
    DESIGNS_FILTERED_by_soft_cuts = math.ceil(fraction_not_passing*DESIGNS_FILTERED_by_hard_cuts)
    DESIGNS_FILTERED = DESIGNS_FILTERED_by_hard_cuts + DESIGNS_FILTERED_by_soft_cuts


    df_soft = df[~df['orderable']]
    df_soft = df_soft.sample(frac=DESIGNS_FILTERED_by_soft_cuts/len(df_soft), weights='prob', replace=False)
    sub = pd.concat([df_soft, df_orderable])
    
    if seqid_cut == 1:
        # Plot everything
        for (i, metric) in enumerate(threshold_dict.keys()):
            ax = sns.distplot(df[metric], ax=axs[i], color='b', label='no filter', norm_hist=True)
            ax = sns.distplot(sub[metric], ax=axs[i], color='red', label='filter', norm_hist=True)

        plt.tight_layout()
        plt.show()

        print('')
        print("Order proposal: %i "%(len(sub)))
        return sub
    
    else:
        ### Cluster by seqid
        # Step 1) make fasta file with silent_tag
        df_filtered = sub.copy()
        clustering_dir = f'{directory}/clustering'
        if use_current_clusters:
            fasta_file = 'tmp.fa'
            print(f'PATH=$PATH:/software/mmseqs2/bin; cd {clustering_dir}; mkdir tmp; mmseqs createdb {fasta_file} DB >> DB_mmseqs.out; mmseqs cluster --min-seq-id {seqid_cut} DB DB_clu tmp >> DB_mmseqs.out; mmseqs createtsv DB DB DB_clu DB_clu.tsv >> DB_mmseqs.out')
        else:
            ### Cluster by seqid
            # Step 1) make fasta file with silent_tag
            os.system(f'rm -r {clustering_dir}')
            os.makedirs(clustering_dir, exist_ok=True)
            df_filtered['fasta'] = '>' + df_filtered['tag'] + '\n' + df_filtered['sequence']
            fastas = list(df_filtered['fasta'])
            fasta_file = 'tmp.fa'
            with open(f'{clustering_dir}/{fasta_file}', 'w') as f_out:
                for fasta in fastas:
                    f_out.write(fasta + '\n')

            # Step 2) create DB and run DB clustering (and create .tsv file)
            # Cluster by 50% seqid
            seq_id_cmd = f'PATH=$PATH:/software/mmseqs2/bin; cd {clustering_dir}; mkdir tmp; mmseqs createdb {fasta_file} DB >> DB_mmseqs.out; mmseqs cluster --min-seq-id {seqid_cut} DB DB_clu tmp >> DB_mmseqs.out; mmseqs createtsv DB DB DB_clu DB_clu.tsv >> DB_mmseqs.out; '
            print(seq_id_cmd)
            os.system(seq_id_cmd)
            
        # Step 3) Load and merge the .tsv file and filter clusters by ddg (or do we pick randomly or by base score instead? tbd)
        cluster_df = pd.read_csv(f'{clustering_dir}/DB_clu.tsv', sep='\t',header=None)
        cluster_df.columns = ['cluster','tag']
        merged_for_clustering_df = pd.merge(df_filtered, cluster_df, how='inner',on='tag')

        # Select the best design from each cluster by ddg
        min_values = {}
        min_tags = {}

        for j, row in merged_for_clustering_df.iterrows():
            cluster = row['cluster']
            tag = row['tag']
            ddg = row[select_by]
            if minimize == True:
                try: 
                    if min_values[cluster] > ddg:
                        min_values[cluster] = ddg
                        min_clusters[cluster] = tag
                    elif min_values[cluster] < ddg:
                        continue
                except: 
                    min_values[cluster] = ddg
                    min_tags[cluster] = tag
            else:
                try:
                    if min_values[cluster] < ddg:
                        min_values[cluster] = ddg
                        min_clusters[cluster] = tag
                    elif min_values[cluster] > ddg:
                        continue
                except:
                    min_values[cluster] = ddg
                    min_tags[cluster] = tag
        min_ddg_list = list(min_tags.values())
        print(f'{len(min_ddg_list)} passing designs from unique clusters.')
        df_min = merged_for_clustering_df.loc[merged_for_clustering_df['tag'].isin(min_ddg_list)]
        
        # Plot everything
        for (i, metric) in enumerate(threshold_dict.keys()):
            ax = sns.distplot(df[metric], ax=axs[i], color='b', label='no filter', norm_hist=True)
            ax = sns.distplot(df_min[metric], ax=axs[i], color='red', label='filter', norm_hist=True)
#             if i == 0:
#                 plt.legend()

#         plt.legend()

        plt.tight_layout()
        plt.show()

        print('')
        print("Order proposal: %i "%(len(df_min)))

        return df_min


# This functions gets pdb annotation from DNA binder designs

def get_annotation(pdb):
    resis = []
    RH_resis = []
    TURN_resis = []
    AH_resis = []
    with open(pdb, 'r') as f_in:
        lines = f_in.readlines()
        for line in lines:
            if 'REMARK PDBinfo-LABEL:' in line and ('RH' in line or 'MOTIF' in line):
                resi = int(line[23:26].strip())
                RH_resis.append(resi)
            if 'REMARK PDBinfo-LABEL:' in line and 'TURN' in line:
                resi = int(line[23:26].strip())
                TURN_resis.append(resi)
            if 'REMARK PDBinfo-LABEL:' in line and 'AH' in line:
                resi = int(line[23:26].strip())
                AH_resis.append(resi)
            if line.startswith('ATOM') and line[21:23].strip() == 'A':
                resi = int(line[23:26].strip())
                if resi not in resis:
                    resis.append(resi)
        fix_resis = RH_resis + TURN_resis + AH_resis
    return resis, fix_resis, RH_resis, TURN_resis, AH_resis

# This function finds backbone-backbone hbond residues


def get_bb_phos_contacts(pose) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making backbone hydrogen bonds with DNA phosphate atoms.
    '''
    pose_hb = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose)
    pose_hb = pose.get_hbonds()
    bb_phos_resis = []
    for hbond in range(1,pose_hb.nhbonds()+1):
        hbond = pose_hb.hbond(hbond)
        donor_res = hbond.don_res()
        acceptor_res = hbond.acc_res()
        donor_hatm = hbond.don_hatm()
        acceptor_atm = hbond.acc_atm()
        don_atom_type = pose.residue(donor_res).atom_name(donor_hatm).strip(" ")
        acc_atom_type = pose.residue(acceptor_res).atom_name(acceptor_atm).strip(" ")
        if acc_atom_type in ["OP1","OP2","O5'","O3'"] and don_atom_type == 'H':
            bb_phos_resis.append(donor_res)
    print(f"n_backbone_phosphate_contacts: {len(bb_phos_resis)}")
    return bb_phos_resis


def count_hbonds_protein_dna(pose) :
    '''
    Takes in a pose and returns the amino acid positions of the residues making hydrogen bonds with DNA.
    '''
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

    aa_pos = []

    for residue in range(len(hbonds_acc_res)) :

        # if pose.residue(hbonds_acc_res[residue]) in ["OP1","OP2","O5'","O4'","O3'"]:
        #     continue

        don_atom_type = pose.residue(hbonds_don_res[residue]).atom_name(hbonds_don_hatm[residue]).strip(" ")
        acc_atom_type = pose.residue(hbonds_acc_res[residue]).atom_name(hbonds_acc_atm[residue]).strip(" ")
        
        if acc_atom_type in ["OP1","OP2","O5'","O4'","O3'"]:
            continue

        if pose.residue(hbonds_don_res[residue]).name().split(':')[0] in DNA_base_names :
            if not pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                if not int(hbonds_acc_res[residue]) in aa_pos:
                    aa_pos.append(int(hbonds_acc_res[residue]))
        else :
            if pose.residue(hbonds_acc_res[residue]).name().split(':')[0] in DNA_base_names :
                if not int(hbonds_don_res[residue]) in aa_pos:
                    aa_pos.append(int(hbonds_don_res[residue]))
    return aa_pos

# This function writes a pdb containing a protein chain.
def write_new_pdb(pdb,pdb_out):
    with open(pdb_out,'w') as f_out:
        with open(pdb, 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                if line.startswith('ATOM') and line[21:23].strip() == 'A':
                    f_out.write(line)
        return


# This function writes inpainting commands
def create_command(script,input_pdb,outpdb,contigs,n,topo_contigs='',topo_conf=0):
    if topo_contigs == '':
        cmd = f'python {script} \
                           --pdb {input_pdb} \
                           --contigs {contigs}\
                           --out {outpdb}_{n}\n'
        n += 1
    else:
        cmd = f'python {script} \
                           --pdb {input_pdb} \
                           --contigs {contigs}\
                           --topo_pdb {input_pdb} \
                           --topo_contigs {topo_contigs} \
                           --topo_conf {topo_conf} \
                           --out {outpdb}_{n}\n'
        n += 1
    return cmd, n

# function to check contig length
def check_len_fun(contigs,target_len):
    good_len = False
    check_len = sum([int(j) for j in contigs.split(',') if 'A' not in j]) + sum([int(j.split('-')[-1]) - int(j.split('-')[0][1:]) + 1 for j in contigs.split(',') if 'A' in j])
    if check_len == target_len:
        good_len = True
    return good_len

# function to check if topo contigs are in input pdb
def check_topo_range(resis,topo_range_min, topo_range_max):
    good_topo = False
    if topo_range_min in resis and topo_range_max in resis:
        good_topo = True
    return good_topo

def mutate_sequence(seq, mut_list):
    '''
    Creates a mutated sequence given a list of mutations
    in the form of ['A21V','G34W',...] (1-indexed).
    If the mutation doesn't match the sequence, this will
    report an error.
    '''
    seq = list(seq)
    for mut in mut_list:
        orig, position, new = mut[0], int(mut[1:-1]), mut[-1]
        assert orig == seq[position-1], "Original sequence does not match up with your stated mutation"
        seq[position-1] = new * 1
    return ''.join(seq)
