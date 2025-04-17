# dbp_design

# 1: steps for rifgen/rifdock. 
Follow instructions in 1_rifgen_rifdock. The pipeline will require installation of Rifdock: https://github.com/rifdock/rifdock. You will require a DNA structure with your target sequence, which can be generated with x3dna (https://x3dna.org) or other means. 

LigandMPNN and Rosetta steps can easily be demoed using a protein-DNA complex PDB, such as those in the corresponding data. 
Runtime for a single PDB on a standard computer ranges from 10s (LigandMPNN, no Rosetta relax) to 2 min (Rosetta design or LigandMPNN + Rosetta Relax)

# 2a: steps for interface design with rosetta (not recommended)
# 2b: steps for interface design with LigandMPNN (recommended)

We recommend the following end-to-end pipeline, which are detailed in the companion notebook dna_interface_design.ipynb. 
1. RIFgen/RIFdock
2. LigandMPNN Prefilter Calibration
3. LigandMPNN Stage 1 + Filtering
4. Loop inpainting diversification (or partial RFdiffusion)
5. LigandMPNN Stage 2 + Filtering
6. LigandMPNN Stage 3 (Ligandmpnn-Rosetta FastRelax recycling) (optional)
7. AlphaFold2 Filtering

The directory `db_design_mpnn` contains information for running the MPNN design script and creation of a Singularity apptainer with necessary dependencies. All of the steps above which use LigandMPNN can be substituted for Rosetta-based design using the scripts in 2a_rosetta_design. However, we strongly recommend using LigandMPNN design. 

We have also included a Jupyter notebook used to generate figures in the manuscript and to demonstrate how data analysis was performed. "Glasscock_et al_figure_generation.ipynb"
