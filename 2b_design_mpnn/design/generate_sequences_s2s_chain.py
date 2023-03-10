from __future__ import print_function
import json, time, os, sys, glob
import shutil
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import os.path

from opt_einsum import contract

# get the program directory
from sys import argv
prog_dir = os.path.dirname(sys.argv[0])
sys.path.append(prog_dir)

# Load external paths
path_file = prog_dir + '/external_paths.txt'
with open(path_file,'r') as f_in:
    paths_dict = {}
    for line in f_in.readlines():
        key = line.split(',')[0]
        path = line.split(',')[1].rstrip()
        paths_dict[key] = path
sys.path.append(paths_dict['mpnn_path'])
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq
from protein_mpnn_utils import tied_featurize, StructureDataset, StructureDatasetPDB #, ProteinMPNN

global DECODING_ORDER

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def set_default_args( seq_per_target, omit_AAs=['X'], bias_AA_dict=None, decoding_order='forward' ):

    global DECODING_ORDER
    DECODING_ORDER = decoding_order

    retval = {}
    retval['BATCH_COPIES'] = min( 1, seq_per_target )
    retval['NUM_BATCHES'] = seq_per_target // retval['BATCH_COPIES']
    retval['temperature'] = 0.1

    omit_AAs_list = omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    retval['omit_AAs_np'] = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
        for n, AA in enumerate(alphabet):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]
    retval['bias_AAs_np'] = bias_AAs_np

    return retval
#


def generate_sequences( model, feature_dict, arg_dict, masked_chains, visible_chains, args, chain_id_dict=None, fixed_positions_dict=None, pssm_dict=None ):
    seqs_scores = []

    with torch.no_grad():

        batch_clones = [copy.deepcopy( feature_dict ) for i in range(arg_dict['BATCH_COPIES'])]
        batch_clones = batch_clones[0]

        chain_dict=None

        Z, Z_m, Z_t, X, X_m, Y, Y_m, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_position_dict=fixed_positions_dict, pssm_dict=pssm_dict) #omit_AA_dict, tied_positions_dict, 
        
        pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float()

       # randn_1 = torch.randn(chain_M.shape).to(device)
       # log_probs = model(X, S, lengths, mask, chain_encoding_all, chain_M*chain_M_pos, randn_1)
       # mask_for_loss = mask*chain_M*chain_M_pos
       # scores = _scores(S, log_probs, mask_for_loss)
       # native_score = scores.cpu().data.numpy()

        for j in range(arg_dict['NUM_BATCHES']):
            randn_2 = torch.randn(chain_M.shape).to(device)
            S_sample = model.sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=arg_dict['temperature'], omit_AAs_np=arg_dict['omit_AAs_np'], bias_AAs_np=arg_dict['bias_AAs_np'], chain_M_pos=chain_M_pos, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag))["S"].to(device)

            # Compute scores
            log_probs = model(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S_sample, chain_M*chain_M_pos, chain_encoding_all, residue_idx, mask)
            mask_for_loss = mask*chain_M
            scores = _scores(S_sample, log_probs, mask_for_loss)
            scores = scores.cpu().data.numpy()

            for b_ix in range( arg_dict['BATCH_COPIES'] ):
                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                score = scores[b_ix]

                seqs_scores.append( (seq,score) )

    return seqs_scores







def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq'] 
                name = entry['name']

                # Convert raw coords to np arrays
                #for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
                    else:
                        discard_count['too_long'] += 1
                else:
                    print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            print('discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class StructureDatasetPDB():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    
class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
            
            
            
# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.reshape(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V 


class DecLayerJ(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,-1,h_E.size(-2),-1) #the only difference
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V



class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E



class ProteinFeatures(nn.Module):
    def __init__(self, node_features, edge_features, num_positional_embeddings=16,
        num_rbf=16, num_rbf_sc=8,top_k=30, augment_eps=0., num_chain_embeddings=16, device=None, side_residue_num=32, atom_context_num=25):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_rbf_side = num_rbf_sc
        self.atom_context_num = atom_context_num
        self.num_positional_embeddings = num_positional_embeddings
        self.side_residue_num = side_residue_num
        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.node_project_down = nn.Linear(5*num_rbf+105, node_features, bias=False)
        #self.node_embedding = nn.Linear(atom_context_num*32, node_features, bias=False) #NOT USED
        
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.edge_embedding_s = nn.Linear(num_rbf_sc*31*5, edge_features, bias=False)


        self.j_nodes = nn.Linear(105, node_features, bias=False)
        self.j_edges = nn.Linear(num_rbf, node_features, bias=False)


        self.norm_j_edges = nn.LayerNorm(node_features)
        self.norm_j_nodes = nn.LayerNorm(node_features)

        #element_dict = {"C": 0, "N": 1, "O": 2, "P": 3, "S": 4}
        #dna_rna_atom_types = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7"]
        #atom_list = [
        #'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        #'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        #'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        #'CZ3', 'NZ']
        self.DNA_RNA_types = torch.tensor([3, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 2, 1, 0], device=device)
        self.side_chain_atom_types = torch.tensor([1, 0, 0, 2, 0, 0, 0, 0, 2, 2, 4, 0, 0, 0, 1, 1, 2, 2, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0, 0, 0, 1], device=device)
       
    def _dist(self, X, mask, top_k_sample=True, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf_side(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf_side
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _get_rbf_side(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf_side(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, X_m, Y, Y_m, Z, Z_m, Z_t, mask, residue_idx, chain_encoding_all):
        """ Featurize coordinates as an attributed graph """
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            Y = Y + self.augment_eps * torch.randn_like(Y)        
            Z = Z + self.augment_eps * torch.randn_like(Z)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        #N, Ca, C, O, Cb - are five atoms representing a residue

        #Get neighbors
        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)


        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb

        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C

        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O



        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_encoding_all[:, :, None] - chain_encoding_all[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset, E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)



        E_idx_sub = E_idx[:,:,:self.side_residue_num] #[B, L, 15]
        side_L = E_idx_sub.shape[2]
        RBF_sidechain = []
        R_m = gather_nodes(X_m[:,:,5:], E_idx_sub) #[B, L, K, 31]
        X_sidechain = X[:,:,5:,:].view(X.shape[0], X.shape[1], -1)
        R = gather_nodes(X_sidechain, E_idx_sub).view(X.shape[0], X.shape[1], side_L, -1, 3) #[B, L, 15, 9, 3]
        Y_t = self.DNA_RNA_types[None,None,None,:].repeat(Y.shape[0], Y.shape[1], Y.shape[2], 1) #[B, L, 10, 26]
        R_t = self.side_chain_atom_types[None,None,None,5:].repeat(X.shape[0], X.shape[1], side_L, 1) #[B, L, 25, 46]
        #R - [B, L, 15, 31, 3]
        #R_m - [B, L, 15, 31]
        #R_t - [B, L, 15, 31]

        #Y - [B, L, 10, 26, 3]
        #Y_m  - [B, L, 10, 26]
        #Y_t - [B, L, 10, 26]

        R = R.view(X.shape[0], X.shape[1], -1, 3)
        R_m = R_m.view(X.shape[0], X.shape[1], -1)
        R_t = R_t.view(X.shape[0], X.shape[1], -1)
        
        Y = Y.view(X.shape[0], X.shape[1], -1, 3)
        Y_m = Y_m.view(X.shape[0], X.shape[1], -1)
        Y_t = Y_t.view(X.shape[0], X.shape[1], -1)
        
        Y = torch.cat([Y, Z[:,None,:,:].repeat(1,Y.shape[1], 1, 1)], -2)
        Y_m = torch.cat([Y_m, Z_m[:,None,:].repeat(1, Y.shape[1], 1)], -1)
        Y_t = torch.cat([Y_t, Z_t[:,None,:].repeat(1, Y.shape[1], 1)], -1)
        
        J = torch.cat((R, Y), 2) #[B, L, atoms, 3]
        J_m = torch.cat((R_m, Y_m), 2) #[B, L, atoms]
        J_t = torch.cat((R_t, Y_t), 2) #[B, L, atoms]



        Cb_J_distances = torch.sqrt(torch.sum((Cb[:,:,None,:] - J)**2,-1) + 1e-6) #[B, L, num_atoms]
        mask_J = mask[:,:,None]*J_m
        Cb_J_distances_adjusted = Cb_J_distances*mask_J+(1. - mask_J)*10000.0
        D_J, E_idx_J = torch.topk(Cb_J_distances_adjusted, self.atom_context_num, dim=-1, largest=False) #pick 25 closest atoms
        
        mask_far_atoms = (D_J < 20.0).float() #[B, L, K]
        J_picked = torch.gather(J, 2, E_idx_J[:,:,:,None].repeat(1,1,1,3)) #[B, L, 50, 3]
        J_t_picked = torch.gather(J_t, 2, E_idx_J) #[B, L, 50]
        J_m_picked = torch.gather(mask_J, 2, E_idx_J) #[B, L, 50]
        J_t_1hot = torch.nn.functional.one_hot(J_t_picked, 105) #N, C, O, P, S #[B, L, 50, 4]


        J_edges = self._rbf(torch.sqrt(torch.sum((J_picked[:,:,:,None,:] - J_picked[:,:,None,:,:])**2,-1) + 1e-6)) #[B, L, 50, 50, num_bins]


        RBF_DNA = []

        D_N_J = self._rbf(torch.sqrt(torch.sum((N[:,:,None,:] - J_picked)**2,-1) + 1e-6)) #[B, L, 50, num_bins]
        D_Ca_J = self._rbf(torch.sqrt(torch.sum((Ca[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_C_J = self._rbf(torch.sqrt(torch.sum((C[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_O_J = self._rbf(torch.sqrt(torch.sum((O[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_Cb_J = self._rbf(torch.sqrt(torch.sum((Cb[:,:,None,:] - J_picked)**2,-1) + 1e-6))

        D_all = torch.cat((D_N_J, D_Ca_J, D_C_J, D_O_J, D_Cb_J, J_t_1hot), dim=-1) #[B,L,25,5*num_bins+5]
        #D_all = torch.cat((D_N_J, D_Ca_J, D_C_J, D_O_J, D_Cb_J), dim=-1) #[B,L,25,5*num_bins+5]
        D_all = D_all*J_m_picked[:,:,:,None]*mask_far_atoms[:,:,:,None]
        V = self.node_project_down(D_all) #[B, L, 50, 32] 
        #V = V.view(X.shape[0], X.shape[1], -1) #[B, L, 50*32]
        #V = self.node_embedding(V)
        V = self.norm_nodes(V)


        J_node_mask = J_m_picked*mask_far_atoms


        J_edges =  self.j_edges(J_edges)
        J_nodes = self.j_nodes(J_t_1hot.float())

        J_edges = self.norm_j_edges(J_edges)
        J_nodes = self.norm_j_nodes(J_nodes)


        return V, E, E_idx, J_node_mask, J_nodes, J_edges



class ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, device=None):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, device=device)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)
        

        self.W_nodes_j = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_edges_j = nn.Linear(hidden_dim, hidden_dim, bias=True)


        self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_C_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)





        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])


        self.context_encoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(2)
        ])
      
        self.j_context_encoder_layers = nn.ModuleList([
            DecLayerJ(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(2)
        ])



        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, X, X_m, Y, Y_m, Z, Z_m, Z_t, randn, S, chain_mask, chain_encoding_all, residue_idx, mask):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        chain_M_ = chain_mask*mask
        X_m = X_m * (1.-chain_M_[:,:,None])
        V, E, E_idx, J_m, J_nodes, J_edges = self.features(X, X_m, Y, Y_m, Z, Z_m, Z_t, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)


        h_V_C = self.W_c(h_V)
        J_m_edges = J_m[:,:,:,None]*J_m[:,:,None,:]
        J_nodes = self.W_nodes_j(J_nodes)
        J_edges = self.W_edges_j(J_edges)
        for i in range(len(self.context_encoder_layers)):
            J_nodes = self.j_context_encoder_layers[i](J_nodes, J_edges, J_m, J_m_edges)
            h_E_context_cat = torch.cat([h_E_context, J_nodes], -1)
            h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, mask, J_m)


        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))


        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_mask = chain_mask*mask #update chain_M to include missing regions
        decoding_order = torch.argsort(torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs



    def sample(self, X, X_m, Y, Y_m, Z, Z_m, Z_t, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None):

        device=X.device

        chain_M_ = chain_mask*chain_M_pos*mask
        X_m = X_m * (1.-chain_M_[:,:,None])


        V, E, E_idx, J_m, J_nodes, J_edges = self.features(X, X_m, Y, Y_m, Z, Z_m, Z_t, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_V_C = self.W_c(h_V)
        J_m_edges = J_m[:,:,:,None]*J_m[:,:,None,:]
        J_nodes = self.W_nodes_j(J_nodes)
        J_edges = self.W_edges_j(J_edges)
        for i in range(len(self.context_encoder_layers)):
            J_nodes = self.j_context_encoder_layers[i](J_nodes, J_edges, J_m, J_m_edges)
            h_E_context_cat = torch.cat([h_E_context, J_nodes], -1)
            h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, mask, J_m)


        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort(torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        t_order = torch.argmax(permutation_matrix_reverse,axis=-1)
        #chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None


        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = t_order[:,t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B]
            if (chain_mask_gathered==0).all():
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(
                    h_V_t, h_ESV_t, mask_V=torch.gather(mask, 1, t[:,None])[:,0,]
                ))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:,None])[:,0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:,None,None].repeat(1,1,pssm_bias.shape[-1]))[:,0]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:,None, None].repeat(1,1,pssm_log_odds_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*pssm_log_odds_mask_gathered+1e-8
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_gathered[:,:,None,]*probs[:,None,:]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S, "probs": all_probs}
        return output_dict


    def tied_sample(self, X, X_m, Y, Y_m, Z, Z_m, Z_t, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, tied_pos=None, tied_beta=None):


        device=X.device

        chain_M_ = chain_mask*chain_M_pos*mask
        # Prepare node and edge embeddings
        X_m = X_m * (1.-chain_M_[:,:,None])


        V, E, E_idx, J_m, J_nodes, J_edges = self.features(X, X_m, Y, Y_m, Z, Z_m, Z_t, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_V_C = self.W_c(h_V)
        J_m_edges = J_m[:,:,:,None]*J_m[:,:,None,:]
        J_nodes = self.W_nodes_j(J_nodes)
        J_edges = self.W_edges_j(J_edges)
        for i in range(len(self.context_encoder_layers)):
            J_nodes = self.j_context_encoder_layers[i](J_nodes, J_edges, J_m, J_m_edges)
            h_E_context_cat = torch.cat([h_E_context, J_nodes], -1)
            h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, mask, J_m)


        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.dropout(h_V_C))


        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort(torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(X.shape[0],1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            done_flag = False
            for t in t_list:
                if (chain_mask[:,t]==0).all():
                    S_t = S_true[:,t]
                    for t in t_list:
                        h_S[:,t,:] = self.W_s(S_t)
                        S[:,t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:,t:t+1,:]
                    h_E_t = h_E[:,t:t+1,:,:]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:,t:t+1,:,:]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:,t:t+1,:]
                        h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1][:,t,:] = layer(h_V_t, h_ESV_t, mask_V=mask[:,t:t+1]).squeeze(1)
                    h_V_t = h_V_stack[-1][:,t,:]
                    logits += tied_beta[t]*(self.W_out(h_V_t) / temperature)/len(t_list)
            if done_flag:
                pass
            else:
                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:,t]
                    pssm_bias_gathered = pssm_bias[:,t]
                    probs = (1-pssm_multi*pssm_coef_gathered[:,None])*probs + pssm_multi*pssm_coef_gathered[:,None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:,t]
                    probs_masked = probs*pssm_log_odds_mask_gathered+1e-8
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:,t]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                for t in t_list:
                    h_S[:,t,:] = self.W_s(S_t_repeat)
                    S[:,t] = S_t_repeat
                    all_probs[:,t,:] = probs.float()
        output_dict = {"S": S, "probs": all_probs}
        return output_dict

