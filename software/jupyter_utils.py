import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os
import getpass
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def make_submit_file(cmds='commands', env='', submitfile='submit.sh', queue='cpu', timeout='3:00:00', group_size=1, ntasks=None, logsfolder='logs', cpus=1, mem=None, gres=None, mem_per_cpu=None):
    username = getpass.getuser()
    logsfolder = f'/net/scratch/{username}/{logsfolder}_{id_generator()}'

    os.makedirs(logsfolder, exist_ok=True)
    n_jobs = sum(1 for line in open(cmds))
    groups = int(np.ceil(float(n_jobs)/group_size))
    
    mem_spec = 'mem' if mem_per_cpu is None else 'mem-per-cpu'
    mem_req = mem if mem_per_cpu is None else mem_per_cpu
    both_mem_specs_defined = (mem is not None) and (mem_per_cpu is not None)
    assert not both_mem_specs_defined # You can only require memory in using one of the options
    
    task_str = f'#SBATCH -n {ntasks}' if ntasks is not None else ''
    gpu_mem = f'#SBATCH --gres={gres}' if gres is not None else ''
    
    submit_file_str = \
    f"""#!/bin/bash
#SBATCH -p {queue}
#SBATCH -c {cpus}
#SBATCH -t {timeout}
#SBATCH -N 1
#SBATCH --{mem_spec}={mem_req}
#SBATCH -a 1-{groups}
#SBATCH -o {logsfolder}/out.%a
#SBATCH -e {logsfolder}/out.%a
{task_str}
{gpu_mem}

GROUP_SIZE={group_size}
{env}
for I in $(seq 1 $GROUP_SIZE)
do
    J=$(($SLURM_ARRAY_TASK_ID * $GROUP_SIZE + $I - $GROUP_SIZE))
    CMD=$(sed -n "${{J}}p" {cmds} )
    echo "${{CMD}}" | bash
done
"""

    with open(submitfile,'w') as f_out:
        f_out.write(submit_file_str)
            

def make_dist_plots(df, relevant_features, df2=None):
    ncols = 3
    nrows = math.ceil(len(relevant_features) / ncols)
    (fig, axs) = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=[15,3*nrows]
    )
    axs = axs.reshape(-1)

    for (i, metric) in enumerate(relevant_features):
        is_int_arr = np.array_equal(df[metric], df[metric].astype(int))
        if is_int_arr:
            c = Counter(df[metric])
            c2 = Counter(df2[metric])
            
            sns.barplot(x=list(c.keys()), y=list(c.values()), ax=axs[i], color='grey', alpha=0.5)
            sns.barplot(x=list(c2.keys()), y=list(c2.values()), ax=axs[i], color='blue')
            axs[i].set_xlabel(metric)
        else:
            sns.distplot(df[metric], ax=axs[i], color='grey')
            sns.distplot(df2[metric], ax=axs[i], color='blue')

    sns.despine()
    plt.tight_layout()
    plt.show()

