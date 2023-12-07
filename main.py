# -*- coding: utf-8 -*-

# 1. 本文件为训练模型的主文件，可用于训练MLP、LSTM、Transformer等模型


from __future__ import unicode_literals, print_function, division
import time
import math
import torch.nn as nn
from torch import optim
from dataPre import *
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from import_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='设置训练用seed，可用于ensemble')
parser.add_argument('--epoch', default=50, type=int, help='训练的epochs')
parser.add_argument('--model', default='MLP', type=str, help='选择模型')
parser.add_argument('--y_label', default='ADE', type=str, help='输出模式/诊断目标：[ADE, FDE, all_fut, dis, OOD_all, OOD_history, OOD_fut]')
parser.add_argument('--data', default='SinD_new_no_all_stop_train_with_scaler', type=str, help='选择训练数据')
parser.add_argument('--weight_decay', default=0, type=float, help='设置权重正则化')
parser.add_argument('--augment', default=False, action='store_true', help='是否用数据增强')
parser.add_argument('--data_source', default='data', type=str, help='选择数据来源 data, data_CV, data_GRIP_seed_30, data_MATP')
# parser.add_argument('--scaler', default=False, action='store_true', help='是否用缩放后的数据')
# parser.add_argument('--cuda_id', default='0', type=str)
args = parser.parse_args()
print('基本配置：', args)

def setup_seed(seed): # 参考:https://www.jianshu.com/p/945103d49655
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if args.seed > 0:# 设置随机数种子，但仅在需要时设置
    setup_seed(args.seed)

#### 导入模型
NNPred = import_model(args)

# main_path = './data' # if args.weight_decay == 0 else f'./data/checkpoints_weight_decay_{args.weight_decay}'

main_path = f'./{args.data_source}'

if args.augment:
    main_path = main_path+'/checkpoints_augment'
if args.weight_decay > 0:
    main_path = main_path + f'/checkpoints_weight_decay_{args.weight_decay}'

if args.seed > 0:
    main_path = main_path+f'/ensemble'

ckpt_path = f'{main_path}/checkpoints_for_{args.data}/{args.y_label}/MLP_add_vel/{args.model}_{args.y_label}'


if args.seed > 0:
    ckpt_path = ckpt_path+f'/seed{args.seed}'

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
writer = SummaryWriter(log_dir=os.path.join(ckpt_path, 'log'))
iters = 0

data_file = f'./{args.data_source}/data_with_scaler_new/{args.data}.pkl' if '_with_scaler' in args.data else f'./{args.data_source}/{args.data}.pkl' # TODO:此处进行了数据源修改
Training_generator, Test, Valid, WholeSet = get_dataloader(data_file, y_label=args.y_label, augment=args.augment)

print('训练数据：', data_file)
print('保存模型：', ckpt_path)

### 保存基本配置——超参数
with open(os.path.join(ckpt_path, 'config.json'), 'w') as conf_json:
    cfg = vars(args)
    json.dump(cfg, conf_json)

# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')

def compute_ADE(pred, gt):
    error = ((((pred - gt)**2).sum(-1))**0.5).mean(-1)
    error = error.mean()
    return error

def compute_dis_loss(error_pred, error_label, pra_error_order=2):
    x2y2 = torch.abs(error_pred - error_label) ** pra_error_order
    error = x2y2.mean()
    return error

def run_trainval(encoder, n_epoch, print_every=1000, plot_every=1, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    # plot_loss_total = 0  # Reset every plot_every
    # criterion = nn.SmoothL1Loss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, 1.0, gamma=0.99) # TODO
    criterion = nn.MSELoss(reduction='mean')
    pltcount = 0
    prtcount = 0
    # cp = 0
    best_loss = np.inf
    # for epoch in tqdm(range(1, n_epoch + 1)):
    for epoch in range(1, n_epoch + 1):
        # 模型训练
        # print('#'*50, '训练阶段')
        encoder.train()
        for iteration, (history_batch, pred_batch, error_labels) in enumerate(tqdm(Training_generator)):
            if history_batch.shape[0] != BatchSize:
                continue
            pltcount = pltcount + 1
            prtcount = prtcount + 1
            # encoder.zero_grad()
            encoder_optimizer.zero_grad()

            history_batch = history_batch.to(device)
            pred_batch = pred_batch.to(device)
            error_labels = error_labels.to(device)

            if 'OOD' in args.y_label:
                gt_batch = error_labels
                if args.y_label == 'OOD_history':
                    input_batch = history_batch
                elif args.y_label == 'OOD_fut':
                    input_batch = error_labels
                elif args.y_label == 'OOD_all':
                    input_batch = torch.cat([history_batch, gt_batch], dim=1)

                if args.model in OOD_list:
                    predY = encoder(input_batch)
                else:
                    predY = encoder(history_batch, gt_batch)
                    predY = predY.view(128, -1, 2)
            # elif args.model in Seq2Seq_list:
            #     predY = encoder(history_batch, pred_batch)
            else:
                predY = encoder(history_batch, pred_batch)

            if args.y_label == 'all_fut':
                loss = compute_ADE(predY.view(128, -1, 2), error_labels)
            elif 'dis' in args.y_label:
                loss = compute_dis_loss(predY, error_labels, pra_error_order=1)
            elif args.y_label in ['ADE', 'FDE']:
                loss = criterion(predY, error_labels).to(device)
            if 'OOD' in args.y_label:
                loss = compute_ADE(predY, input_batch)
            # elif ('OOD' in args.y_label) & (args.model in OOD_list):
            #     loss = compute_ADE(predY, history_batch)
            # elif ('OOD' in args.y_label) & (args.model not in OOD_list):
            #     loss = compute_ADE(predY.view(128,-1,2), history_batch)

            loss.backward()
            encoder_optimizer.step()

            ls = loss.detach().item()
            print_loss_total += ls
            # plot_loss_total += ls

            global iters
            writer.add_scalar('train/MSE Loss', ls, iters)
            iters += 1

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / prtcount
            print_loss_total = 0
            prtcount = 0
            print('Time: %s  epoch: (%d %d%%)  training MSE loss: %f' % (timeSince(start, epoch / n_epoch),
                                       epoch, epoch / n_epoch * 100, print_loss_avg))

        # 模型验证
        encoder.eval()
        val_loss = 0
        val_num = 0
        with torch.no_grad():
            for iteration, (history_batch, pred_batch, error_labels) in enumerate(Valid):
                if history_batch.shape[0] != BatchSize:
                    continue
                history_batch = history_batch.to(device)
                pred_batch = pred_batch.to(device)
                error_labels = error_labels.to(device)

                if 'OOD' in args.y_label:
                    gt_batch = error_labels
                    if args.y_label == 'OOD_history':
                        input_batch = history_batch
                    elif args.y_label == 'OOD_fut':
                        input_batch = gt_batch
                    elif args.y_label == 'OOD_all':
                        input_batch = torch.cat([history_batch, gt_batch], dim=1)

                    if args.model in OOD_list:
                        predY = encoder(input_batch)
                    else:
                        predY = encoder(history_batch, gt_batch)
                        predY = predY.view(128, -1, 2)
                else:
                    predY = encoder(history_batch, pred_batch)

                if args.y_label == 'all_fut':
                    loss = compute_ADE(predY.view(128, -1, 2), error_labels)
                elif 'dis' in args.y_label:
                    loss = compute_dis_loss(predY, error_labels, pra_error_order=1)
                elif args.y_label in ['ADE', 'FDE']:
                    loss = criterion(predY, error_labels).to(device)
                if 'OOD' in args.y_label:
                    loss = compute_ADE(predY, input_batch)
                # elif ('OOD' in args.y_label) & (args.model in OOD_list):
                #     loss = compute_ADE(predY, history_batch)
                # elif ('OOD' in args.y_label) & (args.model not in OOD_list):
                #     loss = compute_ADE(predY.view(128, -1, 2), history_batch)

                ls = loss.item()
                val_loss += ls
                val_num += 1

            cur_val_loss = val_loss/val_num
            writer.add_scalar('val/MSE Loss', cur_val_loss, iters)

        if cur_val_loss < best_loss:
            best_loss = cur_val_loss
            torch.save(encoder.state_dict(), f'{ckpt_path}/best_model.pth.tar')
            torch.save(encoder.state_dict(), f'{ckpt_path}/checkpoint_{epoch}.pth.tar')
            print('#'*20, '第{}次循环更新模型,当前验证集损失: {:.4f}'.format(epoch, best_loss))

        scheduler.step()

        # if epoch % 50 == 1:
        #     # cp = cp + 1
        #     torch.save(encoder.state_dict(), f'{ckpt_path}/checkpoint_{epoch}.pth.tar')

        # if epoch % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / pltcount
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0
        #     pltcount = 0
    return plot_losses

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

if __name__ == '__main__':
    train_iter = iter(Training_generator)
    input_xy, pred_xy, error_label = train_iter.next()
    print(input_xy.shape)
    hidden_size = 256
    if args.y_label in ['all_fut', 'OOD_fut']:
        output_size = pred_xy.shape[1] * 2
    elif 'dis' in args.y_label:
        output_size = pred_xy.shape[1]
    elif args.y_label in ['OOD_history']:
        output_size = input_xy.shape[1] * 2
    elif args.y_label in ['OOD_all']:
        output_size = input_xy.shape[1] * 2 + pred_xy.shape[1] * 2
    else:
        output_size = 1

    Prednet = NNPred(2, output_size=output_size, hidden_size=hidden_size, batch_size=BatchSize)

    print(device)

    # if path.exists("checkpoint.pth.tar"):
    #     Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)

    plot_losses = run_trainval(Prednet, args.epoch, print_every=2)
    torch.save(Prednet.state_dict(), f'{ckpt_path}/checkpoint_{args.epoch}.pth.tar')
    writer.close()

