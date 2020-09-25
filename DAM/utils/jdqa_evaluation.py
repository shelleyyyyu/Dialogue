from sklearn.metrics import accuracy_score, roc_auc_score
import sys

def cal_acc_auc(fname):
    with open (fname, 'r', encoding='utf-8') as file:
        result_arr = file.readlines()
        prob_1_list, predict_label_list, truth_label_list = [], [], []
        for result in result_arr:
            prob_1 = float(result.split('\t')[0])
            prob_1_list.append(prob_1)
            predict_label_list.append(1 if prob_1 > 0.5 else 0)
            truth_label = result.split('\t')[1]
            truth_label_list.append(int(truth_label))

    fname += '.processed'

    with open (fname, 'w', encoding='utf-8') as file:
        for i in range(len(result_arr)):
            file.write(str(prob_1_list[i]) + '\t' + str(predict_label_list[i]) + '\t' + str(truth_label_list[i]) + '\t' + str(truth_label_list[i] == predict_label_list[i]) + '\n')


    print('Accuracy: %.3f' %accuracy_score(predict_label_list, truth_label_list))
    print('AUC Score: %.3f' %roc_auc_score(truth_label_list, prob_1_list))

if __name__ == '__main__':
    fname = sys.argv[1]
    cal_acc_auc(fname)
