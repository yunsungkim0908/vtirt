import os
import json

from vtirt.const import CONFIG_DIR, RUN_DIR

def gen_script(n_q, data_no, n_run):
    run_name = f'hmc_Q{n_q}_no{data_no}_run{n_run}'
    config = {
        "exp": {
            "name": "ExpHMC",
            "model": "dirt_2pl",
            "data": "Wiener2PLDataset"
        },
        "agent": {
            "num_chains": 1,
            "num_warmup": 100,
            "num_samples": 200
        },
        "data": {
            "num_train": 1000,
            "num_valid": 2000,
            "num_ques": n_q,
            "traj_len": n_q,
            "overwrite": False,
            "data_no": data_no
        }
    }
    config_path = os.path.join(CONFIG_DIR, 'hmc', f'{run_name}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    run_dir = os.path.join(RUN_DIR, 'hmc')
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    sbatch_script = f"""
#!/bin/bash
#
#SBATCH --job-name {run_name}
#
#SBATCH --time=7-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=piech
#SBATCH --account=piech
#SBATCH --mem=10G
#SBATCH --output={run_dir}/{run_name}.log

cd /piech/u/yunsung/vtirt/vtirt
python exp/hmc.py configs/hmc/{run_name}.json
    """.strip()
    filename = os.path.join(run_dir, f'{run_name}.sh')
    with open(filename, 'w') as f:
        f.write(sbatch_script)

if __name__=='__main__':
    for n_q in [40, 80, 160, 320]:
        gen_script(n_q, 0, 0)
