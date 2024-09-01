from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib import pyplot as plt

from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from utils import *
from model import *

torch.cuda.is_available()


class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()


##################################################################
##################################################################
def attn_map(attn, subfix):
    attn = attn.cpu().detach().numpy()
    plt.close('all')
    ax_1 = plt.gca()
    im_1 = ax_1.matshow(attn)
    plt.colorbar(im_1)
    plt.savefig(f'{des_dir}ECCTr_{subfix}.png', bbox_inches='tight')


def train(model, device, train_loader, optimizer, epoch, LR, flag=False):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = cum_mloss = 0
    t = time.time()
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(train_loader):
        z_pred = model(magnitude.to(device), syndrome.to(device), flag)
        loss, x_pred = model.loss(-z_pred, bin_to_sign(x).to(device), y.to(device))
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]

    if flag:
        for idx in range(8):
            attn_map(model.decoder.layers[-1].self_attn.attn[0, idx], f'LDPC_121_80_e{str(epoch)}_AttnL{idx + 1}')
    print(
        f'Training epoch {epoch}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} Train Time: {time.time() - t :.2f}s')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################

def test(model, device, test_loader_list, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                # z_mul = (y * bin_to_sign(x))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(-z_pred, bin_to_sign(x).to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                ebno = EbNo_range_test[ii]
                if min_FER > 0:
                    if ebno < 4:
                        if cum_count > 1e4 and test_fer > min_FER:
                            break
                    else:
                        if cum_count >= 1e8:
                            # print(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                            break
                        elif cum_count > 1e5 and test_fer > min_FER:
                            # print(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                            break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            print(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}, Total samples = {cum_count}')

    return test_loss_list, test_loss_ber_list, test_loss_fer_list


##################################################################
##################################################################
##################################################################


def main(args):
    code = args.code
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #################################
    model = ECC_Transformer(args, dropout=0).to(device)
    # model = torch.load(f'{des_dir}BCH__Code_n_127_k_64__12_12_10_44_DRF/best_model')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    EbNo_range_test = range(1, 8)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(ECC_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=True),
                                  batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers)
                            for ii in range(len(std_test))]
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # if epoch in [100, 300, 500, 900, 1500]:
        #  flag = True
        # else:
        # flag = False
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], flag=False)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
        if epoch in [1000, args.epochs]:
            test(model, device, test_dataloader_list, EbNo_range_test)


##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='PolarCode',
                        choices=['BCH', 'PolarCode', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=32)
    parser.add_argument('--code_n', type=int, default=64)
    parser.add_argument('--standardize', default=True)  # #action='store_true')

    # model args
    parser.add_argument('--N_dec', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument("-f", required=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(args.gpus)
    set_seed(args.seed)
    ####################################################################

    print(args.standardize)


    class Code():
        pass


    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    print(args.code_type)
    G, H = Get_Generator_and_Parity(code, standard_form=args.standardize)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ####################################################################
    model_dir = os.path.join(des_dir,
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '__' + datetime.now().strftime(
                                 "%d_%m_%H_%M_DRF"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args)



