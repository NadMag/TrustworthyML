import argparse
import os
import subprocess
import numpy as np

def parse_args():
    description_str = 'Generate Slurm file for running experiments.'
    parser = argparse.ArgumentParser(description=description_str,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', '-f',
                        dest="file_path",
                        metavar='FILE',
                        help =  'path to the run file',
                        default='./run_ae.py')
    parser.add_argument('--run', action='store_true', dest='should_run')
    parser.add_argument('--run_locally', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--name', default='workshop', type=str)
    parser.add_argument('--config',  '-c',
                        dest="config_path",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='./configs/vanila_colab.yaml')
    parser.set_defaults(run_locally=False)
    parser.set_defaults(gpu=True)
    return parser.parse_args()


def generate_job_file(args):
    test_name = args.name
    base_path = os.getcwd()
    output_path = os.path.join(base_path, "slurm/logs", f"{test_name}.out")
    err_path = os.path.join(base_path, "slurm/logs", f"{test_name}.err")
    slurm_lines = ["#!/bin/sh",
                   f"#SBATCH --job-name={test_name}",
                   f"#SBATCH --output={output_path}",
                   f"#SBATCH --error={err_path}",
                   f"#SBATCH --partition=studentkillable",
                   "#SBATCH --time=24:00:00",
                   f"#SBATCH --gpus={1 if args.gpu else 0}",
                   #f"#SBATCH --cpus-per-task={args.process_num}",
                   # f"#SBATCH --mem=2000"
                   #"#SBATCH --mem-per-cpu=8000"
                   ]
    # change this line to run your script
    slurm_lines.append(f"python {args.file_path} -c {args.config_path}")

    out_dir = os.path.join(base_path, "slurm", 'jobs')
    log_dir = os.path.join(base_path, 'slurm', "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{test_name}.slurm")

    with open(file_path, "w") as slurm_file:
        slurm_file.write('\n'.join(slurm_lines) + '\n')
    return file_path


if __name__ == '__main__':
    """ Run jobs on slurm """
    args = parse_args()
    slurm_path = generate_job_file(args)
    if args.should_run == True or args.run_locally == True:
        command = "sh" if args.run_locally == True else "sbatch"
        subprocess.run([command, slurm_path])