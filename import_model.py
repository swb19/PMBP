
MLP_variety_list = [f'MLP_{i}' for i in range(1, 19)] + ['MLP_1_relu', 'MLP_1_no_dropout'] + ['MATP_MLP_1'] + [
    'MLP', 'MLP_dropout', 'MLP_history', 'MLP_pred']

MLP_add_vel_list = [f'MLP_add_vel_{i}' for i in range(1, 17)] + [
    'MATP_MLP_add_vel_1', 'MATP_MLP_add_vel_7'] + ['MLP_add_vel_7_his', 'MLP_add_vel_7_fut', 'MLP_add_vel_7_vel', 'MLP_add_vel_7_acc']
LSTM_add_vel_list = [f'LSTM_add_vel_{i}' for i in range(1, 3)] + ['MATP_LSTM_add_vel_1']
Conv1D_add_vel_list = [f'Conv1D_add_vel_{i}' for i in range(1, 3)] + [f'MATP_Conv1D_add_vel_{i}' for i in range(1, 2)]

Conv1D_variety_list = [f'Conv1D_{i}' for i in range(1, 8)] + ['MATP_Conv1D_1'] + ['densenet_1']
LSTM_variety_list = [f'LSTM_{i}' for i in range(1,14)] + ['MATP_LSTM_1', 'MATP_LSTM_6'] + ['LSTM']
MLP_LSTM_list = ['MLP', 'MLP_dropout', 'LSTM', 'MLP_history', 'MLP_pred'] + MLP_variety_list \
                + Conv1D_variety_list + LSTM_variety_list + MLP_add_vel_list + LSTM_add_vel_list + Conv1D_add_vel_list

Transformer_variety_list = [f'Transformer_{i}' for i in range(1, 8)] + [
    'MATP_Transformer_1'] + ['Transformer', 'Transformer2', 'Transformer_history']
Transformer_add_vel_list = [f'Transformer_add_vel_{i}' for i in range(1,18)] + ['MATP_Transformer_add_vel_7']
Transformer_list = ['Transformer', 'Transformer_history', 'Transformer2'] + Transformer_variety_list + Transformer_add_vel_list

Seq2Seq_list = [f'Seq2Seq_{i}' for i in range(1,5)]

OOD_list = ['LstmAutoEncoder', 'LstmFcAutoEncoder']


def import_model(args):
    if args.model == 'MLP':
        from model.MLP import NNPred
    elif args.model == 'MLP_dropout':
        from model.MLP_dropout import NNPred
    elif args.model == 'LSTM':
        from model.LSTM import NNPred
    elif args.model == 'MLP_history':
        from model.MLP_history import NNPred
    elif args.model == 'MLP_pred':
        from model.MLP_pred import NNPred
    elif args.model == 'Transformer':
        from model.Transformer import NNPred
    elif args.model == 'Transformer_history':
        from model.Transformer_history import NNPred
    elif args.model == 'Transformer2':
        from model.Transformer2 import NNPred
    elif args.model == 'LstmAutoEncoder':
        from model_ood.LstmAutoEncoder import LstmFcAutoEncoder
    elif args.model == 'LstmFcAutoEncoder':
        from model_ood.LstmFcAutoEncoder import LstmFcAutoEncoder
    elif args.model == 'MLP_1':
        from model.MLP_variety.MLP_1 import NNPred
    elif args.model == 'MLP_2':
        from model.MLP_variety.MLP_2 import NNPred
    elif args.model == 'MLP_3':
        from model.MLP_variety.MLP_3 import NNPred
    elif args.model == 'MLP_4':
        from model.MLP_variety.MLP_4 import NNPred
    elif args.model == 'MLP_5':
        from model.MLP_variety.MLP_5 import NNPred
    elif args.model == 'MLP_6':
        from model.MLP_variety.MLP_6 import NNPred
    elif args.model == 'MLP_7':
        from model.MLP_variety.MLP_7 import NNPred
    elif args.model == 'MLP_8':
        from model.MLP_variety.MLP_8 import NNPred
    elif args.model == 'MLP_9':
        from model.MLP_variety.MLP_9 import NNPred
    elif args.model == 'MLP_10':
        from model.MLP_variety.MLP_10 import NNPred
    elif args.model == 'MLP_11':
        from model.MLP_variety.MLP_11 import NNPred
    elif args.model == 'MLP_12':
        from model.MLP_variety.MLP_12 import NNPred
    elif args.model == 'MLP_1_no_dropout':
        from model.MLP_variety.MLP_1_no_dropout import NNPred
    elif args.model == 'MLP_13':
        from model.MLP_variety.MLP_13 import NNPred
    elif args.model == 'MLP_14':
        from model.MLP_variety.MLP_14 import NNPred
    elif args.model == 'MLP_15':
        from model.MLP_variety.MLP_15 import NNPred
    elif args.model == 'MLP_16':
        from model.MLP_variety.MLP_16 import NNPred
    elif args.model == 'MLP_17':
        from model.MLP_variety.MLP_17 import NNPred
    elif args.model == 'MLP_18':
        from model.MLP_variety.MLP_18 import NNPred
    elif args.model == 'LSTM_1':
        from model.LSTM_variety.LSTM_1 import NNPred
    elif args.model == 'LSTM_2':
        from model.LSTM_variety.LSTM_2 import NNPred
    elif args.model == 'LSTM_3':
        from model.LSTM_variety.LSTM_3 import NNPred
    elif args.model == 'LSTM_4':
        from model.LSTM_variety.LSTM_4 import NNPred
    elif args.model == 'LSTM_5':
        from model.LSTM_variety.LSTM_5 import NNPred
    elif args.model == 'LSTM_6':
        from model.LSTM_variety.LSTM_6 import NNPred
    elif args.model == 'LSTM_7':
        from model.LSTM_variety.LSTM_7 import NNPred
    elif args.model == 'LSTM_8':
        from model.LSTM_variety.LSTM_8 import NNPred
    elif args.model == 'LSTM_9':
        from model.LSTM_variety.LSTM_9 import NNPred
    elif args.model == 'LSTM_10':
        from model.LSTM_variety.LSTM_10 import NNPred
    elif args.model == 'LSTM_11':
        from model.LSTM_variety.LSTM_11 import NNPred
    elif args.model == 'LSTM_12':
        from model.LSTM_variety.LSTM_12 import NNPred
    elif args.model == 'LSTM_12':
        from model.LSTM_variety.LSTM_12 import NNPred
    elif args.model == 'Conv1D_1':
        from model.Conv1D_variety.Conv1D_1 import NNPred
    elif args.model == 'Conv1D_2':
        from model.Conv1D_variety.Conv1D_2 import NNPred
    elif args.model == 'Conv1D_3':
        from model.Conv1D_variety.Conv1D_3 import NNPred
    elif args.model == 'Conv1D_4':
        from model.Conv1D_variety.Conv1D_4 import NNPred
    elif args.model == 'Conv1D_5':
        from model.Conv1D_variety.Conv1D_5 import NNPred
    elif args.model == 'Conv1D_6':
        from model.Conv1D_variety.Conv1D_6 import NNPred
    elif args.model == 'Conv1D_7':
        from model.Conv1D_variety.Conv1D_7 import NNPred
    elif args.model == 'Transformer_1':
        from model.Transformer_variety.Transformer_1 import NNPred
    elif args.model == 'Transformer_2':
        from model.Transformer_variety.Transformer_2 import NNPred
    elif args.model == 'Transformer_3':
        from model.Transformer_variety.Transformer_3 import NNPred
    elif args.model == 'Transformer_4':
        from model.Transformer_variety.Transformer_4 import NNPred
    elif args.model == 'Transformer_5':
        from model.Transformer_variety.Transformer_5 import NNPred
    elif args.model == 'Transformer_6':
        from model.Transformer_variety.Transformer_6 import NNPred
    elif args.model == 'Transformer_7':
        from model.Transformer_variety.Transformer_7 import NNPred
    elif args.model == 'MLP_add_vel_1':
        from model.MLP_add_vel.MLP_add_vel_1 import NNPred
    elif args.model == 'MLP_add_vel_2':
        from model.MLP_add_vel.MLP_add_vel_2 import NNPred
    elif args.model == 'MLP_add_vel_3':
        from model.MLP_add_vel.MLP_add_vel_3 import NNPred
    elif args.model == 'MLP_add_vel_4':
        from model.MLP_add_vel.MLP_add_vel_4 import NNPred
    elif args.model == 'MLP_add_vel_5':
        from model.MLP_add_vel.MLP_add_vel_5 import NNPred
    elif args.model == 'MLP_add_vel_6':
        from model.MLP_add_vel.MLP_add_vel_6 import NNPred
    elif args.model == 'MLP_add_vel_7':
        from model.MLP_add_vel.MLP_add_vel_7 import NNPred
    elif args.model == 'MLP_add_vel_7_acc':
        from model.MLP_add_vel.MLP_add_vel_7_acc import NNPred
    elif args.model == 'MLP_add_vel_7_vel':
        from model.MLP_add_vel.MLP_add_vel_7_vel import NNPred
    elif args.model == 'MLP_add_vel_8':
        from model.MLP_add_vel.MLP_add_vel_8 import NNPred
    elif args.model == 'MLP_add_vel_9':
        from model.MLP_add_vel.MLP_add_vel_9 import NNPred
    elif args.model == 'MLP_add_vel_10':
        from model.MLP_add_vel.MLP_add_vel_10 import NNPred
    elif args.model == 'MLP_add_vel_11':
        from model.MLP_add_vel.MLP_add_vel_11 import NNPred
    elif args.model == 'MLP_add_vel_12':
        from model.MLP_add_vel.MLP_add_vel_12 import NNPred
    elif args.model == 'MLP_add_vel_13':
        from model.MLP_add_vel.MLP_add_vel_13 import NNPred
    elif args.model == 'MLP_add_vel_14':
        from model.MLP_add_vel.MLP_add_vel_14 import NNPred
    elif args.model == 'MLP_add_vel_15':
        from model.MLP_add_vel.MLP_add_vel_15 import NNPred
    elif args.model == 'MLP_add_vel_16':
        from model.MLP_add_vel.MLP_add_vel_16 import NNPred
    elif args.model == 'LSTM_add_vel_1':
        from model.MLP_add_vel.LSTM_add_vel_1 import NNPred
    elif args.model == 'LSTM_add_vel_2':
        from model.MLP_add_vel.LSTM_add_vel_2 import NNPred
    elif args.model == 'Conv1D_add_vel_1':
        from model.MLP_add_vel.Conv1D_add_vel_1 import NNPred
    elif args.model == 'Conv1D_add_vel_2':
        from model.MLP_add_vel.Conv1D_add_vel_2 import NNPred
    elif args.model == 'MATP_MLP_1':
        from model.MATP_model.MLP_1 import NNPred
    elif args.model == 'MATP_LSTM_1':
        from model.MATP_model.LSTM_1 import NNPred
    elif args.model == 'MATP_LSTM_6':
        from model.MATP_model.LSTM_6 import NNPred
    elif args.model == 'MATP_Conv1D_1':
        from model.MATP_model.Conv1D_1 import NNPred
    elif args.model == 'MATP_Transformer_1':
        from model.MATP_model.Transformer_1 import NNPred
    elif args.model == 'MATP_MLP_add_vel_1':
        from model.MATP_model.MLP_add_vel_1 import NNPred
    elif args.model == 'MATP_MLP_add_vel_7':
        from model.MATP_model.MLP_add_vel_7 import NNPred
    elif args.model == 'MATP_Transformer_add_vel_7':
        from model.MATP_model.Transformer_add_vel_7 import NNPred
    elif args.model == 'MATP_Conv1D_add_vel_1':
        from model.MATP_model.Conv1D_add_vel_1 import NNPred
    elif args.model == 'MATP_LSTM_add_vel_1':
        from model.MATP_model.LSTM_add_vel_1 import NNPred

    elif args.model == 'Transformer_add_vel_1':
        from model.Transformer_add_vel.Transformer_add_vel_1 import NNPred
    elif args.model == 'Transformer_add_vel_2':
        from model.Transformer_add_vel.Transformer_add_vel_2 import NNPred
    elif args.model == 'Transformer_add_vel_3':
        from model.Transformer_add_vel.Transformer_add_vel_3 import NNPred
    elif args.model == 'Transformer_add_vel_4':
        from model.Transformer_add_vel.Transformer_add_vel_4 import NNPred
    elif args.model == 'Transformer_add_vel_5':
        from model.Transformer_add_vel.Transformer_add_vel_5 import NNPred
    elif args.model == 'Transformer_add_vel_6':
        from model.Transformer_add_vel.Transformer_add_vel_6 import NNPred
    elif args.model == 'Transformer_add_vel_7':
        from model.Transformer_add_vel.Transformer_add_vel_7 import NNPred
    elif args.model == 'Transformer_add_vel_8':
        from model.Transformer_add_vel.Transformer_add_vel_8 import NNPred
    elif args.model == 'Transformer_add_vel_9':
        from model.Transformer_add_vel.Transformer_add_vel_9 import NNPred
    elif args.model == 'Transformer_add_vel_10':
        from model.Transformer_add_vel.Transformer_add_vel_10 import NNPred
    elif args.model == 'Transformer_add_vel_11':
        from model.Transformer_add_vel.Transformer_add_vel_11 import NNPred
    elif args.model == 'Transformer_add_vel_12':
        from model.Transformer_add_vel.Transformer_add_vel_12 import NNPred
    elif args.model == 'Transformer_add_vel_13':
        from model.Transformer_add_vel.Transformer_add_vel_13 import NNPred
    elif args.model == 'Transformer_add_vel_14':
        from model.Transformer_add_vel.Transformer_add_vel_14 import NNPred
    elif args.model == 'Transformer_add_vel_15':
        from model.Transformer_add_vel.Transformer_add_vel_15 import NNPred
    elif args.model == 'Transformer_add_vel_16':
        from model.Transformer_add_vel.Transformer_add_vel_16 import NNPred
    elif args.model == 'Transformer_add_vel_17':
        from model.Transformer_add_vel.Transformer_add_vel_17 import NNPred

    elif args.model == 'MLP_add_vel_7_his':
        from model.MLP_add_vel.MLP_add_vel_7_his import NNPred
    elif args.model == 'MLP_add_vel_7_fut':
        from model.MLP_add_vel.MLP_add_vel_7_fut import NNPred
    elif args.model == 'Seq2Seq_1':
        from model.Seq2Seq_variety.Seq2Seq_1 import NNPred
    elif args.model == 'Seq2Seq_2':
        from model.Seq2Seq_variety.Seq2Seq_2 import NNPred
    elif args.model == 'Seq2Seq_3':
        from model.Seq2Seq_variety.Seq2Seq_3 import NNPred
    elif args.model == 'Seq2Seq_4':
        from model.Seq2Seq_variety.Seq2Seq_4 import NNPred

    elif args.model == 'densenet_1':
        from model.DenseNet.densenet_1 import NNPred

    if args.model in  ['LstmAutoEncoder', 'LstmFcAutoEncoder']:
        return LstmFcAutoEncoder
    else:
        return NNPred