Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
rsync -a --no-o --no-g /usr/local/cuda-11.7/lib/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
rsync -a --no-o --no-g /usr/local/cuda-11.7/lib64/ $APPTAINER_ROOTFS/usr/lib/x86_64-linux-gnu/
rsync -a --no-o --no-g /usr/local/cuda-11.7/bin/ $APPTAINER_ROOTFS/usr/bin/

%files
/etc/localtime
/etc/apt/sources.list

%post
# Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks for IPD
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

# apt
apt-get update
apt-get install -y python3-pip python-is-python3

# pip packages
pip install \
  ipython \
  pandas \
  numpy \
  matplotlib \
  ipykernel \
  seaborn \
  biopython \
  ml-collections \
  torch

pip install \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  ipython \
  pandas \
  numpy \
  matplotlib \
  ipykernel \
  seaborn \
  biopython \
  mock \
  jax[cuda12_pip] \
  jaxlib \
  dm-haiku \
  dm-tree \
  ml-collections \
  tensorflow \
  prody

# PyRosetta
pip install pyrosetta -f https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.linux.wheel/

# Clean up
apt-get clean
pip cache purge

%runscript
exec python "$@"

%help
PyTorch/Tensorflow environment for PPI design.
Author: Nate Bennett
