#!/usr/bin/env python3
import numpy as np
import argparse
import yaml

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Collection dataset')
    parser.add_argument('--config', type=str, default="config/sample/check_operability/point_mass.yml",
                        help='config path e.g. config/sample/check_operability/point_mass.yml')
    args = parser.parse_args()

    with open(args.config) as _file:
        cfg = yaml.safe_load(_file)

if __name__ == "__main__":
    main()
