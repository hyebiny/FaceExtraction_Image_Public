#!/usr/bin/env bash
# CJENM-test CelebA-HQ-WO-test test_benchmark_02


root='/experiments/2023-09-22-01-28-04/test_images/test_benchmark_02'
python evaluation.py --pred-dir '.'$root \
                --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

root='/experiments/2023-09-22-09-41-27/test_images/test_benchmark_02'
python evaluation.py --pred-dir '.'$root \
                --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

root='/experiments/2023-09-22-09-42-01/test_images/test_benchmark_02'
python evaluation.py --pred-dir '.'$root \
                --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/2023-09-04-11-11-46/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/2023-09-04-11-12-14/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/2023-09-06-00-02-57/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/mgmatting/2023-08-30-17-26-33/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'
# root='/experiments/mgmatting/2023-08-31-09-56-43/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'


