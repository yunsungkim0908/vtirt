import argparse
import os
import json
from dotmap import DotMap
from pprint import pprint

from vtirt.exp.svi import ExpSVI
from vtirt.exp.tskirt import ExpTSKIRT
from vtirt.exp.hmc import ExpHMC
from vtirt.const import OUT_DIR

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'

    config['path'] = args.config_path
    config = DotMap(config)

    base_dirname = config.path.split('configs/')[-1].split('.json')[0]
    out_dirname = os.path.join(OUT_DIR, base_dirname)

    exp = globals()[config.exp.name](config, device, out_dirname)
    exp.run()
