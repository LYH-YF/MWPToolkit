# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/07/31 15:06:33
# @File: run_mwptoolkit.py


import argparse
import sys
import os

from mwptoolkit.quick_start import run_toolkit

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GTS', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='math23k', help='name of datasets')
    parser.add_argument('--task_type', '-t', type=str, default='single_equation', help='name of tasks')

    args, _ = parser.parse_known_args()
    config_dict = {}

    #config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    #run_toolkit()
    run_toolkit(args.model, args.dataset, args.task_type, config_dict)
