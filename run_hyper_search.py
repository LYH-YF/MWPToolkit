import argparse
import json

from ray import tune

from mwptoolkit.hyper_search import hyper_search_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GTS', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='math23k', help='name of datasets')
    parser.add_argument('--task_type', '-t', type=str, default='single_equation', help='name of tasks')
    parser.add_argument('--search_parameter', '-s', type=str, action='append', default=[])
    #parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    config_dict = {}
    parameter_dict = {}
    search_parameter = args.search_parameter
    for parameter in search_parameter:
        value = parameter.split('=')
        parameter_dict[value[0]] = tune.grid_search(json.loads(value[1]))

    hyper_search_process(args.model, args.dataset, args.task_type, parameter_dict, config_dict)
