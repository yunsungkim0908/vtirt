import os
import json

from vtirt.const import CONFIG_DIR, RUN_DIR

def gen_config(n_q, data_no, n_run):
    run_name = f'Q{n_q}_no{data_no}_run{n_run}'
    config= {
        "exp": {
            "name": "ExpTSKIRT",
            "data": "Wiener2PLDataset"
        },
        "data": {
            "num_train": 1000,
            "num_valid": 2000,
            "num_ques": n_q,
            "traj_len": n_q,
            "overwrite": True,
            "data_no": data_no
        },
        "valid": {
            "loader": {
                "batch_size": 512,
                "num_workers": 1
            }
        }
    }
    config_path = os.path.join(CONFIG_DIR, 'tskirt', f'{run_name}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def gen_script():
    run_dir = os.path.join(RUN_DIR, 'tskirt')
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    sbatch_script = f"""
#!/bin/bash
#
#SBATCH --job-name tskirt
#
#SBATCH --time=7-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=piech
#SBATCH --account=piech
#SBATCH --mem=10G
#SBATCH --output={run_dir}/tskirt.log

cd /piech/u/yunsung/vtirt/vtirt
for f in configs/tskirt/*.json
do
     python exp/tskirt.py $f
done
    """.strip()
    filename = os.path.join(run_dir, 'tskirt.sh')
    with open(filename, 'w') as f:
        f.write(sbatch_script)

if __name__=='__main__':
    for n_q in [40, 80, 160, 320]:
        gen_config(n_q, 0, 0)

    gen_script()
