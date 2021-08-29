import argparse
import sys
import os

from ray import tune

from mwptoolkit.hyper_search import hyper_search_process
from mwptoolkit.utils.utils import read_json_data


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GTS', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='math23k', help='name of datasets')
    parser.add_argument('--task_type', '-t', type=str, default='single_equation', help='name of tasks')
    parser.add_argument('--search_parameter', '-s', type=str, action='append', default=[])
    parser.add_argument('--search_file','-f',type=str,default=None)

    args, _ = parser.parse_known_args()
    config_dict = {}
    parameter_dict = {}
    if args.search_file != None:
        search_parameter=read_json_data(args.search_file)
        for parameter in search_parameter:
            value = parameter.split('=')
            space = eval(value[1])
            if isinstance(space,list):
                parameter_dict[value[0]] = tune.grid_search(space)
            elif isinstance(space,tuple) and len(space)==2:
                if space[0]==0 and space[1]==1:
                    parameter_dict[value[0]] = tune.uniform(space[0],space[1])
                else:
                    parameter_dict[value[0]] = tune.loguniform(space[0],space[1])
            else:
                parameter_dict[value[0]] = space
    search_parameter = args.search_parameter
    for parameter in search_parameter:
        value = parameter.split('=')
        space = eval(value[1])
        if isinstance(space,list):
            parameter_dict[value[0]] = tune.grid_search(space)
        elif isinstance(space,tuple) and len(space)==2:
            if space[0]==0 and space==1:
                parameter_dict[value[0]] = tune.uniform(space[0],space[1])
            else:
                parameter_dict[value[0]] = tune.loguniform(space[0],space[1])
        else:
            parameter_dict[value[0]] = space

    hyper_search_process(args.model, args.dataset, args.task_type, parameter_dict, config_dict)
