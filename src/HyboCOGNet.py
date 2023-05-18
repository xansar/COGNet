import torch
import torch.distributed as dist

import argparse
import numpy as np
import dill
import time
from hybo_modules.optim import RiemannianAdam, RiemannianSGD
from torch.utils import data
import os
import torch.nn.functional as F
import random
from collections import defaultdict
import wandb

from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace

import sys

sys.path.append("..")
from hybo_modules import HyboCOGNet
from hypergraph_construction import construct_graphs
from util import llprint, ddi_rate_score, get_n_params, output_flatten
from recommend import eval, test

torch.manual_seed(1203)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

model_name = 'HyboCOGNet'
resume_path = f'./saved/{model_name}/Epoch_43_JA_0.5047_DDI_0.0863.model'

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--debug', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lamda', type=float, default=0., help='weight decay')
parser.add_argument('--n_layers', type=int, default=1, help='num of layers')
parser.add_argument('--beta', type=float, default=0., help='contrastive learning loss factor')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')
parser.add_argument('--ddp', action='store_true', default=False,
                    help='whether to use ddp')

args = parser.parse_args()

if args.ddp:
    dist.init_process_group(backend='nccl')


def init_wandb(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MLHC",
        group=args.model_name,

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "weight_decay": args.lamda,
            "architecture": args.model_name,
            "dataset": "MIMIC-III",
            "epochs": 50,
            "beta": args.beta,
            "batch_size": args.batch_size,
            "emb_dim": args.emb_dim,
            "seed": 1203
        },

        name=f'{args.model_name}_lr_{args.lr}_beta_{args.beta}_lamda_{args.lamda}',

        # dir
        dir='./saved'
    )


def log_and_eval(total_loss_lst, pred_loss_lst, con_loss_lst, model, eval_dataloader, voc_size, epoch, device, TOKENS,
                 args, tic, history, best_ja, best_epoch):
    print(
        f'\nLoss: {np.mean(total_loss_lst):.4f}\tPred Loss: {np.mean(pred_loss_lst):.4f}\tCon Loss: {np.mean(con_loss_lst):.4f}')
    tic2 = time.time()
    model.cache_embedding_for_eval()  # 获取图卷积embedding
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, voc_size, epoch, device,
                                                              TOKENS, args)
    print('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

    history['ja'].append(ja)
    history['ddi_rate'].append(ddi_rate)
    history['avg_p'].append(avg_p)
    history['avg_r'].append(avg_r)
    history['avg_f1'].append(avg_f1)
    history['prauc'].append(prauc)
    history['med'].append(avg_med)
    history['total_loss'].append(np.mean(total_loss_lst))
    history['pred_loss'].append(np.mean(pred_loss_lst))
    history['con_loss'].append(np.mean(con_loss_lst))

    wandb.log({
        'ja': ja,
        'ddi_rate': ddi_rate,
        'avg_p': avg_p,
        'avg_r': avg_r,
        'avg_f1': avg_f1,
        'prauc': prauc,
        'med': avg_med,
        'total_loss': np.mean(total_loss_lst),
        'pred_loss': np.mean(pred_loss_lst),
        'con_loss': np.mean(con_loss_lst)
    })

    if epoch >= 5:
        print('ddi: {}, Med: {}, Ja: {}, F1: {}'.format(
            np.mean(history['ddi_rate'][-5:]),
            np.mean(history['med'][-5:]),
            np.mean(history['ja'][-5:]),
            np.mean(history['avg_f1'][-5:]),
            np.mean(history['prauc'][-5:]),
            np.mean(history['total_loss'][-5:]),
            np.mean(history['pred_loss'][-5:]),
            np.mean(history['con_loss'][-5:])
        ))
    torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
                                                     'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja,
                                                                                                ddi_rate)),
                                        'wb'))

    if best_ja < ja:
        best_epoch = epoch
        best_ja = ja
        wandb.save(f"best_{args.model_name}.h5")

    print('best_epoch: {}'.format(best_epoch))

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))
    return best_ja, best_epoch


def main(args):
    # load data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'
    cache_pth = '../data/graphs.pkl'

    # ehr_adj_path = '../data/weighted_ehr_adj_final.pkl'
    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    ddi_mask_path = '../data/ddi_mask_H.pkl'
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1

    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x: med_count[x])
            data[i][j][2] = cur_medications

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    H, A = construct_graphs(
        cache_pth=cache_pth,
        data_train=data_train,
        n_diag=voc_size[0] + 3,  # +3是考虑pad token
        n_pro=voc_size[1] + 3,
        n_drug=voc_size[2] + 3
    )

    if args.debug:
        print('=' * 20 + 'DEBUG' + '=' * 20)
        data_train = data_train[:100]
        data_eval = data_eval[:10]
    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_eval)
    test_dataset = mimic_data(data_test)

    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        model = HyboCOGNet(voc_size, ehr_adj, ddi_adj, ddi_mask_H, H, A, emb_dim=args.emb_dim, n_layers=args.n_layers,
                           device=device)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train,
                                      shuffle=False, pin_memory=True, sampler=train_sampler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False,
                                     pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False,
                                     pin_memory=True)

        model = model.cuda(local_rank)
        torch.cuda.set_device(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True,
                                                          broadcast_buffers=False)
        model.to(device=device)
        if dist.get_rank() == 0:
            print(f"Diag num:{len(diag_voc.idx2word)}")
            print(f"Proc num:{len(pro_voc.idx2word)}")
            print(f"Med num:{len(med_voc.idx2word)}")
            print('parameters', get_n_params(model))

            if not os.path.exists(os.path.join("saved", model_name)):
                os.makedirs(os.path.join("saved", model_name))
            if not args.Test:
                init_wandb(args)

    else:
        print(f"Diag num:{len(diag_voc.idx2word)}")
        print(f"Proc num:{len(pro_voc.idx2word)}")
        print(f"Med num:{len(med_voc.idx2word)}")

        if not os.path.exists(os.path.join("saved", model_name)):
            os.makedirs(os.path.join("saved", model_name))

        if not args.Test:
            init_wandb(args)

        device = torch.device(f'cuda:{1}')
        model = HyboCOGNet(voc_size, ehr_adj, ddi_adj, ddi_mask_H, H, A, emb_dim=args.emb_dim, n_layers=args.n_layers,
                           device=device)
        model.to(device=device)
        print('parameters', get_n_params(model))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train,
                                      shuffle=True, pin_memory=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False,
                                     pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=False,
                                     pin_memory=True)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        model.cache_embedding_for_eval()
        smm_record, ja, prauc, precision, recall, f1, med_num = test(model, test_dataloader, diag_voc, pro_voc, med_voc,
                                                                     voc_size, 0, device, TOKENS, ddi_adj, args)
        result = []
        for _ in range(10):
            data_num = len(ja)
            final_length = int(0.8 * data_num)
            idx_list = list(range(data_num))
            random.shuffle(idx_list)
            idx_list = idx_list[:final_length]
            avg_ja = np.mean([ja[i] for i in idx_list])
            avg_prauc = np.mean([prauc[i] for i in idx_list])
            avg_precision = np.mean([precision[i] for i in idx_list])
            avg_recall = np.mean([recall[i] for i in idx_list])
            avg_f1 = np.mean([f1[i] for i in idx_list])
            avg_med = np.mean([med_num[i] for i in idx_list])
            cur_smm_record = [smm_record[i] for i in idx_list]
            ddi_rate = ddi_rate_score(cur_smm_record, path='../data/ddi_A_final.pkl')
            result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
            llprint(
                '\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                    ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)
        print('test time: {}'.format(time.time() - tic))
        return

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.lamda, stabilize=10)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 50
    if args.debug:
        EPOCH = 3

    for epoch in range(EPOCH):
        if args.ddp:
            if dist.get_rank() == 0:
                train_sampler.set_epoch(epoch)

        tic = time.time()
        if args.ddp:
            if dist.get_rank() == 0:
                print('\nepoch {} --------------------------'.format(epoch))
        else:
            print('\nepoch {} --------------------------'.format(epoch))

        model.train()
        pred_loss_lst = []
        con_loss_lst = []
        total_loss_lst = []
        for idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
            dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)
            # media_output = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix,
                                            # m_mask_matrix,
                                            # seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                                            # dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            output_logits, con_loss = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix,
                                            m_mask_matrix,
                                            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                                            dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix,
                                                 voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
            pred_loss = F.nll_loss(predictions, labels.long())

            loss = pred_loss
            # loss = pred_loss + args.beta * con_loss

            loss.backward()
            optimizer.step()
            pred_loss_lst.append(pred_loss.item())
            con_loss_lst.append(con_loss.item())
            total_loss_lst.append(loss.item())
            llprint('\rtraining step: {} / {}'.format(idx, len(train_dataloader)))

        if args.ddp:
            if dist.get_rank() == 0:
                best_epoch, best_ja = log_and_eval(total_loss_lst, pred_loss_lst, con_loss_lst, model.module,
                                                   eval_dataloader, voc_size, epoch,
                                                   device, TOKENS, args, tic, history, best_ja, best_epoch)
        else:
            best_epoch, best_ja = log_and_eval(total_loss_lst, pred_loss_lst, con_loss_lst, model,
                                               eval_dataloader, voc_size, epoch,
                                               device, TOKENS, args, tic, history, best_ja, best_epoch)

    if args.ddp:
        if dist.get_rank() == 0:
            wandb.finish()
    else:
        wandb.finish()


if __name__ == '__main__':
    main(args)
