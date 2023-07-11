import os
import json

from vtirt.const import CONFIG_DIR, RUN_DIR

def gen_script(model, n_q, data_no, n_run):
    run_name = f'{model}_Q{n_q}_no{data_no}_run{n_run}'
    config={
        "exp": {
            "name": "ExpSVI",
            "model": model,
            "data": "Wiener2PLDataset"
        },
        "agent": {
            "hidden_dim": 8
        },
        "data": {
            "num_train": 1000,
            "num_valid": 2000,
            "num_ques": n_q,
            "traj_len": n_q,
            "overwrite": False,
            "data_no": data_no
        },
        "train": {
            "num_epochs": 300,
            "loader": {
                "batch_size": 32,
                "num_workers": 1
            }
        },
        "valid": {
            "loader": {
                "batch_size": 512,
                "num_workers": 1
            }
        },
        "optim": {
            "lr": 3e-3
        }
    }
    config_path = os.path.join(CONFIG_DIR, 'vtirt', f'{run_name}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    run_dir = os.path.join(RUN_DIR, 'vtirt')
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    sbatch_script = f"""
#!/bin/bash
#
#SBATCH --job-name vtirt_{run_name}
#
#SBATCH --time=7-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=piech
#SBATCH --account=piech
#SBATCH --mem=10G
#SBATCH --output={run_dir}/{run_name}.log

cd /piech/u/yunsung/vtirt/vtirt
python exp/svi.py configs/vtirt/{run_name}.json
    """.strip()
    filename = os.path.join(run_dir, f'{run_name}.sh')
    with open(filename, 'w') as f:
        f.write(sbatch_script)

if __name__=='__main__':
    for model in ['VTIRTSingleKC', 'VTIRTSingleKCIndep', 'VIBOSingleKC']:
        for n_q in [40, 80, 160, 320]:
            gen_script(model, n_q, 0, 0)

