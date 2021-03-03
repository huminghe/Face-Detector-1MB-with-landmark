# !/usr/bin/python
# -*- coding: utf-8 -*-


import argparse
import os
from pathlib import Path
from multiprocessing import Pool


def execute_predict(path, out, gpu_id):
    name = path.name
    command = 'python get_videos.py '
    command += ' --dataset_folder ' + str(path)
    command += ' --save_folder ' + str(os.path.join(out, name))
    command += ' --device ' + str(gpu_id)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_path",
                        default="/data/flamingo/recordfiles",
                        type=str,
                        help="source data path")
    parser.add_argument("--output_path",
                        default="/data/huminghe/test/face_test",
                        type=str,
                        help="output path")
    parser.add_argument("--pool_size",
                        default=4,
                        type=int,
                        help="the max number of processes to allow at one time")
    parser.add_argument("--model_path",
                        default="./weights/RBF_Final.pth",
                        type=str,
                        help="model path")
    parser.add_argument("--model_config",
                        default="RFB",
                        type=str,
                        help="model type")
    parser.add_argument("--gpu_id",
                        default=2,
                        type=int,
                        help="gpu id")

    main_args = parser.parse_args()

    pool = Pool(main_args.pool_size)

    for path in Path(main_args.source_data_path).glob('*'):
        pool.apply_async(execute_predict, args=(path, main_args.output_path, main_args.gpu_id))

    pool.close()
    pool.join()
    print("all job done.")
