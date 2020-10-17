- Our Framework
Calibration Pretrain steps \t Matching Pretrain steps \t Joint Training Epoch \t Others \t Min \t Max
					
1)	douban_A_D100_05_015_09: 	1	1	5	A+D100_0.5	0.15 	0.9
2) 	douban_A_D100_05_025_09:	1	1	5	A+D100_0.5	0.25 	0.9
3) 	douban_A_C_01_01_09    :	1	1	5	A+C_0.1	    0.1	    0.9

CUDA_VISIBLE_DEVICES=5 nohup python -u douban_A_D100_05_015_09.py > ./log_douban/train_douban_A_D100_05_015_09.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u douban_A_D100_05_025_09.py > ./log_douban/train_douban_A_D100_05_025_09.log 2>&1&
CUDA_VISIBLE_DEVICES=5 nohup python -u douban_A_C_01_01_09.py > ./log_douban/train_douban_A_C_01_01_09.log 2>&1&