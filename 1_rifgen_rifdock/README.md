# Protein-DNA docking with RIFgen + RIFdock

Running this pipeline requires pre-installation of RIFdock, which itself requires Rosetta. Follow instructions at https://github.com/rifdock/rifdock. Additionally, silent tools (https://github.com/bcov77/silent_tools) should be installed to manage PDB silent files:

```
cd ../software
git clone https://github.com/bcov77/silent_tools
cd silent_tools
echo "PATH=\$PATH:$(pwd)"
```

Running RIFgen requires an input pdb of the target DNA, such as `CGCACCGACTCACG.pdb`, and further inputs of protein residues to use as hotspots, which are provided in the hotspots folder. You will also require the unzipped scaffold pdbs:

```
cd ./scaffold_pdbs; unzip scaffold.pdbs; cd ../
```

The contained jupyter notebook walks through the steps in RIFgen/RIFdock for DBPs, although some details are specific to the IPD computing cluster. This notebook requires a python environment with the following packages: sys, os, time, glob, random, shutil, subprocess, math, re, numpy, pandas.

RIFgen is controlled by many command-line flags, which we have placed into the file `rifgen.flag` as an example. 

RIFgen is run with the following command format:
```
/software/rifdock/rifgen @rifgen.flag > rifgen.log 2>&1
```

To run docking, RIFdock uses the initial inputs, the RIFgen outputs, as well as additional protein scaffold structures, provided as pdbs.
We have provided another file, `rifdock.flag`, to contain an example of the command-line options we used for RIFdock.
RIFdock is run with the following command format:
```
/software/rifdock/rif_dock_test @rifdock.flag -scaffolds [scaffold1.pdb] [scaffold2.pdb] > rifdock.log 2>&1
```
