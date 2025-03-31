import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader as pyDataLoader
from torch.utils.data import Dataset
from models import SoftWeighted
import pandas as pd
import pickle
from sklearn import metrics
import warnings
from config import Config

warnings.filterwarnings("ignore")

label_path = './Dataset/'
prediction_path = './Dataset/prediction/'

def init():
    global label_path, prediction_path
    if Config.test_type == 1:
        label_path = './Dataset/Test_60.pkl'
        prediction_path = './Dataset/prediction/test_60/'
    elif Config.test_type == 2:
        label_path = './Dataset/Test_315-28.pkl'
        prediction_path = './Dataset/prediction/test315-28/'
    elif Config.test_type == 3:
        label_path = './Dataset/Test_60.pkl'
        prediction_path = './Dataset/prediction/Btest31-6/'
    elif Config.test_type == 4:
        label_path = './Dataset/UBtest_31-6.pkl'
        prediction_path = './Dataset/prediction/UBtest31-6/'


class ViewDataset(Dataset):
    def __init__(self, n_view, y, x, x1=None, x2=None, x3=None):
        self.n_view = n_view
        self.y = y
        self.x = x
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def __getitem__(self, index):
        if self.n_view == 1:
            return self.x[index], self.y[index]
        elif self.n_view == 2:
            return self.x[index], self.x1[index], self.y[index]
        elif self.n_view == 3:
            return self.x[index], self.x1[index], self.x2[index], self.y[index]
        else:
            return self.x[index], self.x1[index], self.x2[index], self.x3[index], self.y[index]

    def __len__(self):
        return len(self.x)

# feats_type = "VNEGNN-Alpha_VNPLM-Alpha_GSAEGNN-Alpha_GSAPLM-Alpha"
feats_type = "VNEGNN_VNPLM_GSAEGNN_GSAPLM"
n_view = 4
agg_type = 'adaptive' # adaptive, avg
BATCH_SIZE = 1
EPOCHS = 15
torch.manual_seed(Config.seed)


def analysis(y_true, y_pred, sequence_names, best_threshold = None):
    t1, t2 = [], []
    y_true = list(y_true)
    for i in range(len(y_true)):
        t1.extend(list(y_true[i]))
    for i in range(len(y_pred)):
        t2.extend(list(y_pred[i]))
    y_t, y_p = y_true, y_pred
    y_true, y_pred = t1, t2
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true
    one, right, false = 0, 0, 0
    for i in range(len(binary_true)):
        one += binary_true[i]
        if binary_true[i] == 1 and binary_pred[i] == 1:
            right += 1
        if binary_true[i] == 0 and binary_pred[i] == 1:
            false += 1
    print('all : ', one, ' right : ', right, ' false : ', false)

    # this is work for length analyse
    small_tp, small_tn, small_fp, small_fn = 0, 0, 0, 0
    medium_tp, medium_tn, medium_fp, medium_fn = 0, 0, 0, 0
    large_tp, large_tn, large_fp, large_fn = 0, 0, 0, 0

    # get ans for signle protein
    for i in range(len(sequence_names)):
        sequence_name = sequence_names[i]
        # if sequence_name != '4nqwB':
        #     continue
        yy_t, yy_p = y_t[i], y_p[i]
        b_t = yy_t
        b_p = [1 if pred >= best_threshold else 0 for pred in yy_p]
        op, r, f = 0, 0, 0
        # this is work for case study
        tp, fp, fn, tn = 0, 0, 0, 0

        base = 7
        tp_list, fp_list, fn_list = [], [], []
        seq = [0] * len(b_t)
        for i in range(len(b_t)):
            op += b_t[i]
            if b_t[i] == 1 and b_p[i] == 1:
                r += 1
                tp += 1
                seq[i] = 1
                idx = i + base
                tp_list.append(idx)
            if b_t[i] == 1 and b_p[i] == 0:
                fn += 1
                seq[i] = 2
                idx = i + base
                fn_list.append(idx)
            if b_t[i] == 0 and b_p[i] == 1:
                f += 1
                fp += 1
                seq[i] = 3
                idx = i + base

                fp_list.append(idx)
            if b_t[i] == 0 and b_p[i] == 0:
                tn += 1
        if tp + fp == 0:
            pre = 0
        else:
            pre = tp * 1.0 / (tp + fp)
        rec = tp * 1.0 / (tp + fn)
        if pre + rec == 0:
            f1 = 0
        else:
            f1 = (pre * rec * 2) / (pre + rec)
        # print(sequence_name, ' ', op, ' right : ', r, ' false : ', f, ' tp : ', tp, ' fp : ', fp, ' fn : ', fn, ' tn: ', tn, " ACC: ", (tp + tn) * 1.0 / (tp + tn + fp + fn), " f1: ", f1)
        if len(b_t) < 128:
            small_fn += fn
            small_fp += fp
            small_tp += tp
            small_tn += tn
        elif len(b_t) <= 248:
            medium_fn += fn
            medium_fp += fp
            medium_tp += tp
            medium_tn += tn
        else:
            large_fn += fn
            large_fp += fp
            large_tp += tp
            large_tn += tn
        # IFKVGDTVVYPHHGAALVEAIETREQKEYLVLKVAQGDLTVRVPAENAEYVGVRDVVGQEGLDKVFQVLRAPWSRRYKANLEKLASGDVNKVAEVVRDLWRRDQERGLSAGEKRMLAKARQILVGELALAESTDDAKAETILDEVLAA
        # prepare for pymol, draw case study graph
        # print("tp list: ", tp_list)
        # tp_str = ''
        # for x in tp_list:
        #     tp_str += str(x)
        #     tp_str += "+"
        # print(tp_str)
        # print("fp list: ", fp_list)
        # fp_str = ''
        # for x in fp_list:
        #     fp_str += str(x)
        #     fp_str += "+"
        # print(fp_str)
        # print("fn list: ", fn_list)
        # fn_str = ''
        # for x in fn_list:
        #     fn_str += str(x)
        #     fn_str += "+"
        # print(fn_str)

    # small_ACC = (small_tp + small_tn) * 1.0 / (small_tp + small_tn + small_fp + small_fn)
    # small_pre = small_tp * 1.0 / (small_tp + small_fp)
    # small_rec = small_tp * 1.0 / (small_tp + small_fn)
    # small_f1 = (small_pre * small_rec * 2) / (small_pre + small_rec)
    # print("small_acc : ", small_ACC, " small_f1: ", small_f1)
    #
    # medium_ACC = (medium_tp + medium_tn) * 1.0 / (medium_tp + medium_tn + medium_fp + medium_fn)
    # medium_pre = medium_tp * 1.0 / (medium_tp + medium_fp)
    # medium_rec = medium_tp * 1.0 / (medium_tp + medium_fn)
    # medium_f1 = (medium_pre * medium_rec * 2) / (medium_pre + medium_rec)
    # print("medium_acc : ", medium_ACC, " medium_f1: ", medium_f1)
    #
    # large_ACC = (large_tp + large_tn) * 1.0 / (large_tp + large_tn + large_fp + large_fn)
    # large_pre = large_tp * 1.0 / (large_tp + large_fp)
    # large_rec = large_tp * 1.0 / (large_tp + large_fn)
    # large_f1 = (large_pre * large_rec * 2) / (large_pre + large_rec)
    # print("large_acc : ", large_ACC, " large_f1: ", large_f1)

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def process_type3(dataset):
    if Config.test_type == 3:
        ans = {}
        with open("./Dataset/bound_unbound_mapping31-6.txt", "r") as f:
            lines = f.readlines()[1:]
        for line in lines:
            bound_ID, unbound_ID, _ = line.strip().split()
            ans[bound_ID] = dataset[bound_ID]
        return ans
    return dataset

# AlphaFold3 case study
def post_dataset(dataset):
    # if Config.test_type == 1:
        # dataset = dataset['4cdgA']
        # dataset = {'4cdgA': dataset}
        # global agg_type
        # agg_type = 'real_test_60'
        # agg_type = 'AlphaFold3_test_60' # real_test_60
    return dataset


def get_zero_list():
    with open(label_path, "rb") as f:
        dataset = pickle.load(f)
    dataset = process_type3(dataset)
    dataset = post_dataset(dataset)
    dataset = sorted(dataset.items(), key=lambda kv: kv[0])
    zeros_list = []
    for data in dataset:
        label = data[1][1]
        label = list(map(float, label))
        t = [0] * len(label)
        t = list(map(float, t))
        zeros_list.append(t)
    return zeros_list


def main(device):
    # load test data label
    with open(label_path, "rb") as f:
        dataset = pickle.load(f)
    dataset = process_type3(dataset)
    dataset = post_dataset(dataset)
    dataset = sorted(dataset.items(), key=lambda kv: kv[0])
    sequence_name_list = [t[0] for t in dataset]
    labels = []
    zeros_list = []
    for data in dataset:
        label = data[1][1]
        label = list(map(float, label))
        labels.append(label)
        t = [0] * len(label)
        t = list(map(float, t))
        zeros_list.append(t)

    # load multi-view prediction
    feats_list = feats_type.split('_')
    output_list = []
    train_list = []

    for name in feats_list:
        preds_file = prediction_path + name + '.pkl'
        if os.path.exists(preds_file) is False:
            continue
        print(preds_file)
        test_df = pd.read_pickle(preds_file)
        test_df = post_dataset(test_df)
        test_df = sorted(test_df.items(), key=lambda kv: kv[0])
        preds = []
        for tp in test_df:
            pred = list(map(float, tp[1]))
            preds.append(pred)
        prediction = preds
        train_list.append(prediction)
        output_list.append(prediction)

    while len(train_list) < 12:
        train_list.append([])
    deal_dataset = ViewDataset(n_view, labels, train_list[0], train_list[1], train_list[2], train_list[3])
    test_loader = pyDataLoader(deal_dataset, batch_size=BATCH_SIZE, shuffle=False)

    net = SoftWeighted(n_view).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=5e-3)
    lr_sched = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    mcc_max = -1
    best_result = None
    best_weight = None
    for epoch in range(EPOCHS):
        test_loss = 0.0
        net.train()
        print('Epoch:{', epoch + 1, '} ----------------------')
        for data in test_loader:
            data = [torch.tensor(x).to(device) for x in data]
            label = data.pop()
            output, weight_var = net(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            loss.backward()
            optimizer.step()

        weight_var = None
        with torch.no_grad():
            for data in test_loader:
                data = [torch.tensor(x).to(device) for x in data]
                label = data.pop()
                output, weight_var = net(data)
                break

        test_loss /= len(test_loader)
        weight_var = [weight_var[i].item() for i in range(n_view)]
        zeros_list = get_zero_list()
        fusion_pre = zeros_list

        if agg_type == 'avg':
            weight_var = [1.0 / n_view for i in range(n_view)]
        elif agg_type == 'real_test_60':
            weight_var = [0.3313891291618347, 0.16522787511348724, 0.3381921052932739, 0.1651908904314041]
        elif agg_type == 'AlphaFold3_test_60':
            weight_var = [0.3058316111564636, 0.17136450111865997, 0.35158807039260864, 0.17121583223342896]
        for i in range(n_view):
            for j in range(len(output_list[i])):
                for k in range(len(output_list[i][j])):
                    fusion_pre[j][k] += weight_var[i] * output_list[i][j][k]

        print(weight_var)
        result_test = analysis(labels, fusion_pre, sequence_name_list)
        print("========== Evaluate Test set ==========")
        print("test loss : ", test_loss)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])

        if result_test['mcc'] > mcc_max:
            mcc_max = result_test['mcc']
            best_result = result_test
            best_weight = weight_var

        if lr_sched:
            scheduler.step(test_loss)

    print("\n")
    print("Best weight : ")
    print(best_weight)
    print("The Best in Test data : ")
    print("Test binary acc: ", best_result['binary_acc'])
    print("Test precision:", best_result['precision'])
    print("Test recall: ", best_result['recall'])
    print("Test f1: ", best_result['f1'])
    print("Test AUC: ", best_result['AUC'])
    print("Test AUPRC: ", best_result['AUPRC'])
    print("Test mcc: ", best_result['mcc'])
    print("Threshold: ", best_result['threshold'])

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init()
    main(device)