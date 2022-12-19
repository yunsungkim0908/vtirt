import argparse
import json
from dotmap import DotMap
from pprint import pprint

from vtirt.exp.svi import ExpSVI

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)
        pprint(config)

    config = DotMap(config)
    exp = globals()[config.exp.name](config)
    exp.run()
