# MPNN sequence design

Running MPNN requires installation of several programs, including LigandMPNN, PyRosetta, PSI_pred, and PSI_blast. The paths to these programs must be encoded in the 'external_paths.txt' folder. 
The mpnn_design.py script automates the process of MPNN design and has several options including design of structures in silent files or a folder of pdbs. Additional options include the ability to generate many MPNN sequences and run FastDesign on the resulting outputs constrained by a PSSM from the MPNN designs. 

An apptainer containing the dependencies necessary for running `mpnn_design.py` may be built from mpnn_design.spec: `apptainer build mpnn_design.sif mpnn_design.spec` 

Options are also available to freeze resis labeled as hotspots in the pdb labels, freeze specific resis provided in a comma separated list, and skip designs that lack mainchain-phosphate hydrogen bonds or residues labeled with MOTIF or RIFRES in the recognition helix.

Design can be run in prefilter mode, which first checks if a design passes specified prefilters that are calibrated for their likelihood to pass final metric cutoffs. To run a design in prefilter calibration mode, use the following example:

DEMO: the following scripts can be run on a protein-DNA complex PDB, such as those included in the supplementary data files. Expected run time is approximately 10s - 2 min depending on settings (using Rosetta relax will take ~ 2 min per PDB). 

```
python  {full_path_to_mpnn_design.py} -silent {silent} \
                                -seq_per_struct 1 \  ### this option specifies the number of MPNN sequences to generate per input structure
                                -n_per_silent 1 \ ### In prefilter calibration, you should use this to randomly sample designs from rifdock silent file
                                -temperature 0.2 \ ### This specificies the temperature used in LigandMPNN design
                                -redesign_base_hbonds 1 \ ### In some cases, you want to turn off redesign of residues that form hydrogen bonds to bases, but not here.
                                -hbond_energy_cut -0.5 \ ### This specificies the energy cutoff for what counts as a hydrogen bond in filtering steps
                                -run_predictor 1 \ ### This specifies whether you should run a prefilter before full fast relax. If set to 0, the script will proceed straight to full relax after MPNN sequence design
                                -run_relax 1\n ### Setting this to 1 specificies that a full relax should be performed on the final structures. If set to 0, the script will save a structure with the MPNN sequence threaded onto the input design model.
```
Once you have generated 1000-10000 designs in prefilter calibration mode, you can follow the notebook 'prefilter_calibration.ipynb' to generate files that specify prefilter cutoffs.


Next, you we run design on the entire set of rifdock outputs using prefiltering to avoid full relax on bad docks. When paths to `prefilter_eq_file` and `prefilter_mle_cut_file` are specified, bad docks will be pre-empted from full relax.


```
python {full_path_to_mpnn_design.py} -silent {silent} \
                                -seq_per_struct 1 \
                                -temperature 0.2 \
                                -redesign_base_hbonds 1 \
                                -hbond_energy_cut -0.5 \
                                -prefilter_eq_file {prefilter_eq_file} \ ### specify the path to the generated prefilter_eq_file
                                -prefilter_mle_cut_file {prefilter_mle_cut_file} \ ### specificy the path to the generated prefilter_mle_cut_file
                                -run_predictor 1 \
                                -run_relax 1\n
```

You may iterate through MPNN design multiple steps to improve design metrics using a command similar to below. Evidence shows that this iteration between MPNN and Fast Relax can improve sequence design quality. Typically, one can do this with a lower MPNN temp.

```
python {full_path_to_mpnn_design.py} -silent {silent} \
                                -seq_per_struct 1 \
                                -temperature 0.1 \
                                -redesign_base_hbonds 1 \
                                -hbond_energy_cut -0.5 \
                                -run_relax 1\n`
```
