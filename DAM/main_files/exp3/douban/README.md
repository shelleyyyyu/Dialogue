- Parameters: douban_A_C_01_01_09

1) threshold 0.1 0.9
2) threshold 0.2 0.8
3) threshold 0.3 0.7
4) threshold 0.4 0.6
5) threshold 0.5 0.5


CUDA_VISIBLE_DEVICES=0 nohup python -u 3.1.9_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_01_09.log 2>&1&
CUDA_VISIBLE_DEVICES=1 nohup python -u 3.2.8_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_02_08.log 2>&1&
CUDA_VISIBLE_DEVICES=2 nohup python -u 3.3.7_douban_A_C_01_03_07.py > ./log_douban/train_douban_A_C_01_01_09_03_07.log 2>&1&
CUDA_VISIBLE_DEVICES=3 nohup python -u 3.4.6_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_04_06.log 2>&1&
CUDA_VISIBLE_DEVICES=4 nohup python -u 3.5.5_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_05_05.log 2>&1&