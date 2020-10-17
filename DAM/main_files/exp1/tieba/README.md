- Our Framework
Calibration Pretrain steps \t Matching Pretrain steps \t Joint Training Epoch \t Others \t Min \t Max
					
1) tieba_A_D100_05_015_09:	1	1	5	A+D100_0.5	0.15 	0.9
2) tieba_A_D100_05_02_09 :	1	1	5	A+D100_0.5	0.2	    0.9
3) tieba_A_D100_05_025_09:	1	1	5	A+D100_0.5  0.25 	0.9
4) tieba_A_B100_01_085   :  1	1	5	A+B_100	    0.1	    0.85
5) tieba_A_D100_05_01_085:	1	1	5	A+D100_0.5	0.1	    0.85
6) tieba_A_D100_05_01_08 :	1	1	5	A+D100_0.5	0.1	    0.8


CUDA_VISIBLE_DEVICES=0,2 nohup python -u 1_tieba_A_D100_05_015_09.py > ./log_tieba/train_tieba_A_D100_05_015_09.log 2>&1&

[NOT RUN YET]
CUDA_VISIBLE_DEVICES=1 nohup python -u 2_tieba_A_D100_05_02_09.py > ./log_tieba/train_tieba_A_D100_05_02_09.log 2>&1&
CUDA_VISIBLE_DEVICES=2 nohup python -u 3_tieba_A_D100_05_025_09.py > ./log_tieba/train_tieba_A_D100_05_025_09.log 2>&1&
CUDA_VISIBLE_DEVICES=3 nohup python -u 4_tieba_A_B100_01_085.py > ./log_tieba/train_tieba_A_B100_01_085.log 2>&1&
CUDA_VISIBLE_DEVICES=4 nohup python -u 5_tieba_A_D100_05_01_085.py > ./log_tieba/train_tieba_A_D100_05_01_085.log 2>&1&
CUDA_VISIBLE_DEVICES=0 nohup python -u 6_tieba_A_D100_05_01_08.py > ./log_tieba/train_tieba_A_D100_05_01_08.log 2>&1&
