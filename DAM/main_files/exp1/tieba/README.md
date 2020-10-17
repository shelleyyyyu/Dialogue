- Our Framework
Calibration Pretrain steps \t Matching Pretrain steps \t Joint Training Epoch \t Others \t Min \t Max
					
1) tieba_A_D100_05_015_09:	1	1	5	A+D100_0.5	0.15 	0.9
2) tieba_A_D100_05_02_09 :	1	1	5	A+D100_0.5	0.2	    0.9
3) tieba_A_D100_05_025_09:	1	1	5	A+D100_0.5  0.25 	0.9
4) tieba_A_B100_01_085   :  1	1	5	A+B_100	    0.1	    0.85
5) tieba_A_D100_05_01_085:	1	1	5	A+D100_0.5	0.1	    0.85
6) tieba_A_D100_05_01_08 :	1	1	5	A+D100_0.5	0.1	    0.8


CUDA_VISIBLE_DEVICES=5 nohup python -u tieba_A_D100_05_015_09.py > ./log_tieba/train_tieba_A_D100_05_015_09.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u tieba_A_D100_05_02_09.py > ./log_tieba/train_tieba_A_D100_05_02_09.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u tieba_A_D100_05_025_09.py > ./log_tieba/train_tieba_A_D100_05_025_09.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u tieba_A_B100_01_085.py > ./log_tieba/train_tieba_A_B100_01_085.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u tieba_A_D100_05_01_085.py > ./log_tieba/train_tieba_A_D100_05_01_085.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u tieba_A_D100_05_01_08.py > ./log_tieba/train_tieba_A_D100_05_01_08.log 2>&1&
