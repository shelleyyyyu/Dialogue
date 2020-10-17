- Parameters: tieba_A_D100_05_01_08


- Parameters: tieba_A_D100_05_01_08

1) validation_data_amt 0.9
2) validation_data_amt 0.8
3) validation_data_amt 0.7
4) validation_data_amt 0.6
5) validation_data_amt 0.5
6) validation_data_amt 0.4
7) validation_data_amt 0.3
8) validation_data_amt 0.2
9) validation_data_amt 0.1

CUDA_VISIBLE_DEVICES=0,1 nohup python -u 6.9_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_09.log 2>&1&
CUDA_VISIBLE_DEVICES=2,5 nohup python -u 6.8_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_08.log 2>&1&
CUDA_VISIBLE_DEVICES=7,6 nohup python -u 6.7_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_07.log 2>&1&
CUDA_VISIBLE_DEVICES=3 nohup python -u 6.6_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_06.log 2>&1&
CUDA_VISIBLE_DEVICES=4 nohup python -u 6.5_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_05.log 2>&1&
CUDA_VISIBLE_DEVICES=0 nohup python -u 6.4_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_04.log 2>&1&
CUDA_VISIBLE_DEVICES=1 nohup python -u 6.3_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_03.log 2>&1&
CUDA_VISIBLE_DEVICES=2 nohup python -u 6.2_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_02.log 2>&1&
CUDA_VISIBLE_DEVICES=3 nohup python -u 6.1_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_01.log 2>&1&
