import subprocess
import os
import tempfile

i = 0
seed = 0
for n_latents_env in [1, 2, 5, 10, 20, 50]:
    for inducing_points_env in [10, 20, 50, 100, 200, 500]:
        nodelist = i % 7 + 1
        i += 1
        sbatch = f'''#!/bin/bash
#SBATCH --job-name={seed}_{n_latents_env}_{inducing_points_env}  # Give your experiment a name
#SBATCH --mail-type=END  # Type of email notification: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=thhe@cs.aau.dk
#SBATCH --output=/nfs/home/cs.aau.dk/cp68wp/slurm-output/run_python-%j.out  # Redirect the output stream to this file (%j is the jobid)
#SBATCH --error=/nfs/home/cs.aau.dk/cp68wp/slurm-output/run_python-%j.err   # Redirect the error stream to this file (%j is the jobid)
#SBATCH --partition=rome  # Which partitions may your job be scheduled on
#SBATCH --nodelist=rome0{nodelist}
#SBATCH --mem=48G  # Memory limit that slurm allocates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --time=1:00:00  # (Optional) time limit in dd:hh:mm:ss format. Make sure to keep an eye on your jobs (using 'squeue -u $(whoami)') anyways.

U=cp68wp@cs.aau.dk
PD=/nfs/home/cs.aau.dk/cp68wp

# Activate python virtual environment
source ${{PD}}/venv/bin/activate

cd ${{PD}}/Biogeography/

mkdir -p /nfs/home/cs.aau.dk/cp68wp/res_kybernetika/saved/env_{seed}_{n_latents_env}_{inducing_points_env}

##########################
# Run your python script #
##########################

# Maybe you need to copy a result file back to ${{PD}}
python -m EcoGP.train --config config_dirmul --n_latents_env {n_latents_env} --n_inducing_points_env {inducing_points_env} --seed {seed} --save_model_path /nfs/home>
'''

        # Write and submit
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
            f.write(sbatch)
            script_path = f.name

        subprocess.run(
            ["sbatch", script_path],
            text=True,
        )
        os.remove(script_path)

