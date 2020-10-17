- Parameters: douban_A_C_01_01_09

1) validation_data_amt 0.9
2) validation_data_amt 0.8
3) validation_data_amt 0.7
4) validation_data_amt 0.6
5) validation_data_amt 0.5
6) validation_data_amt 0.4
7) validation_data_amt 0.3
8) validation_data_amt 0.2
9) validation_data_amt 0.1

CUDA_VISIBLE_DEVICES=0,1 nohup python -u 3.9_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_09.log 2>&1&
CUDA_VISIBLE_DEVICES=2,5 nohup python -u 3.8_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_08.log 2>&1&
CUDA_VISIBLE_DEVICES=7,6 nohup python -u 3.7_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_07.log 2>&1&
CUDA_VISIBLE_DEVICES=4 nohup python -u 3.6_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_06.log 2>&1&

CUDA_VISIBLE_DEVICES=4 nohup python -u 3.5_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_05.log 2>&1&
CUDA_VISIBLE_DEVICES=0 nohup python -u 3.4_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_04.log 2>&1&
CUDA_VISIBLE_DEVICES=1 nohup python -u 3.3_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_03.log 2>&1&
CUDA_VISIBLE_DEVICES=2 nohup python -u 3.2_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_02.log 2>&1&
CUDA_VISIBLE_DEVICES=3 nohup python -u 3.1_douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09_01.log 2>&1&
