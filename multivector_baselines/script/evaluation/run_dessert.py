import os

if __name__ == '__main__':
    config_l = {
        'dbg': {
            # 'dataset_l': ['lotte-lifestyle', 'lotte', 'msmacro'],
            'dataset_l': ['quora'],
            # 'build_index_parameter_l': [{'n_table': 32}, {'n_table': 64}, {'n_table': 128}, {'n_table': 256}, ]
            # 'build_index_parameter_l': [{'n_table': 16}, {'n_table': 32}, {'n_table': 64} ]
            'build_index_parameter_l': [{'n_table': 64}]
        },
        'local': {
            'dataset_l': ['lotte-500-gnd'],
            'build_index_parameter_l': [{'n_table': 32}]
        }
    }
    host_name = 'local'
    config = config_l[host_name]
    dataset_l = config['dataset_l']
    build_index_parameter_l = config['build_index_parameter_l']
    for dataset in dataset_l:
        for build_index_parameter in build_index_parameter_l:
            cmd = f'python3 eval_dessert.py --host_name {host_name} --dataset {dataset} --n_table {build_index_parameter["n_table"]} '
            os.system(cmd)
