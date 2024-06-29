from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../..")
import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from Modules.StaticDAG.NRIMLP import MLPEncoder, MLPDecoder
from Modules.StaticDAG.NRICNN import CNNEncoder
from Modules.StaticDAG.NRIRNN import RNNDecoder
from Modules.StaticDAG.NRISimulationDecoder import SimulationDecoder
from tqdm import tqdm
from Modules.LoadData.load_data_ADNI import get_dataloader
dataloader = get_dataloader(batch_size = 1, parent=3)

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=90,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=1,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=187,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

args, _ = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

for data, id, group in tqdm(dataloader):
    if args.encoder == 'mlp':
        encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                            args.edge_types,
                            args.encoder_dropout, args.factor)
    elif args.encoder == 'cnn':
        encoder = CNNEncoder(args.dims, args.encoder_hidden,
                            args.edge_types,
                            args.encoder_dropout, args.factor)

    if args.decoder == 'mlp':
        decoder = MLPDecoder(n_in_node=args.dims,
                            edge_types=args.edge_types,
                            msg_hid=args.decoder_hidden,
                            msg_out=args.decoder_hidden,
                            n_hid=args.decoder_hidden,
                            do_prob=args.decoder_dropout,
                            skip_first=args.skip_first)
    elif args.decoder == 'rnn':
        decoder = RNNDecoder(n_in_node=args.dims,
                            edge_types=args.edge_types,
                            n_hid=args.decoder_hidden,
                            do_prob=args.decoder_dropout,
                            skip_first=args.skip_first)
        
    if args.load_folder:
        encoder_file = os.path.join(args.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(args.load_folder, 'decoder.pt')
        decoder.load_state_dict(torch.load(decoder_file))

        args.save_folder = False

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                        lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                    gamma=args.gamma)



    if args.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()


    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        scheduler.step()


        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)

        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 100,
                                burn_in=True,
                                burn_in_steps=args.timesteps - args.prediction_steps)
        else:
            output = decoder(data, edges, rel_rec, rel_send,
                                args.prediction_steps)
        target = data.permute(0,2,1).unsqueeze(3)[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                                args.edge_types)
        loss = loss_nll + loss_kl
        loss.backward()
        optimizer.step()
    causality = np.zeros([args.num_atoms,args.num_atoms])
    temp1 = np.where(off_diag)[0]
    temp2 = np.where(off_diag)[1]
    for i in range(args.num_atoms*args.num_atoms-args.num_atoms):
        causality[temp1[i],temp2[i]] = edges.cpu().detach().numpy()[0,i,0]
    folder = "ECNs_results/" + id[0] + "_" + group[0] + ".npy"
    np.save(folder, causality)




end_time = time.time()
print(f"time:{end_time - start_time}seconds")
