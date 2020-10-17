- Parameters: tieba_A_D100_05_01_08

1) threshold 0.1 0.9
2) threshold 0.2 0.8
3) threshold 0.3 0.7
4) threshold 0.4 0.6
5) threshold 0.5 0.5


CUDA_VISIBLE_DEVICES=0 nohup python -u 6.1.9_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_01_09.log 2>&1&
CUDA_VISIBLE_DEVICES=1 nohup python -u 6.2.8_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_02_08.log 2>&1&
CUDA_VISIBLE_DEVICES=2 nohup python -u 6.3.7_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_03_07.log 2>&1&
CUDA_VISIBLE_DEVICES=3 nohup python -u 6.4.6_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_04_06.log 2>&1&
CUDA_VISIBLE_DEVICES=4 nohup python -u 6.5.5_tieba_A_D100_05_01_08.py > ./log_douban/train_tieba_A_D100_05_01_08_05_05.log 2>&1&