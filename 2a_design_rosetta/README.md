# Design of RIFdock outputs with Rosetta design. Recommended pipeline follows a similar logic as MPNN-based design. 

python {full_path_to_rosetta_design.py} --silent_in {silent} \
                                     --pssmFile {pssm_f} \ ### path to pssm file for given input scaffold
                                     --pssmCut {pssm_cut} \ ### choose between [0, -2.0]
                                     --pssm_weight 0.5 \
                                     --silent_out {outsilent} \
                                     --weights {weights_file} \ ### path to RM8B weights
                                     --rand_string {rand_string} \ ### random string to rename design output
                                     --require_rifres_in_rec_helix 1 \ ### ensure docks contain rifres in the recognition helix
                                     --rboltz_in_predictor 0 \ ### don't use this
                                     --hbond_energy_cut -0.5 \ ### specify the energy cutoff for counting hydrogen bonds
                                     --flags_file {flags_file} \ ### path to RM8B flags
                                     --n_per_silent 1' ### number of designs to sample within the silent file
