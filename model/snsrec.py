import torch
import torch.nn as nn
import torch.nn.functional as F

from data.temporal_sets_data_loader1 import TemporalSetsInput
from model.attention import Attention, AttentionPool
from model.time_encoding import PositionalEncoding, TimestampEncoding
from model.functions import itm, stm, pad_sequence, time_encode, set_embedding, din
from model.co_transformer import CDTE, PDTE, DualTransformer, CoTransformerLayer, CoTransformerLayer2, CoTransformerLayer3, CoTransformer


class SNSRec(nn.Module):

    def __init__(self,
                 items_total: int,
                 embed_dim: int,
                 time_encoding: str,
                 set_embed_method: str,
                 num_set_embeds: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 itm_temperature: float=1.0,
                 stm_temperature: float=1.0,
                 dropout: float=0.1,
                 set_embed_dropout: float = 0.1,
                 attn_output: int=0,       
                 co_existence: int=0,
                 coex_scores: int=0        
                 ): 
    
        super(SNSRec, self).__init__()
        self.items_total = items_total  
        self.embed_dim = embed_dim
        self.co_emb_dim = 64
        self.item_embed = nn.Embedding(num_embeddings=items_total, embedding_dim=embed_dim) 
        if co_existence:
            self.coex_item_embed = nn.Embedding(num_embeddings=items_total, embedding_dim=self.co_emb_dim)  #only use 64 dim
            # transformer
            self.coex_transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.co_emb_dim,
                                                                                                  nhead=num_transformer_heads,
                                                                                                  dropout=dropout),
                                                         num_layers=num_transformer_layers)
            self.coex_item_attn = Attention(embed_dim=self.co_emb_dim, temperature=itm_temperature)
            
        # time encoding
        self.time_encoding_method = time_encoding
        if time_encoding == 'none':
            self.time_encoding = None
        elif time_encoding == 'positional':
            self.time_encoding = PositionalEncoding(embed_dim=embed_dim)
        elif time_encoding == 'timestamp': ##’timestamp‘ in dsntsp.josn
            self.time_encoding = TimestampEncoding(embed_dim=embed_dim)
        else:
            raise NotImplementedError()

        # set embedding
        self.set_embed_method = set_embed_method
        if set_embed_method == 'attn_pool':
            self.set_embed = AttentionPool(embed_dim, num_queries=num_set_embeds, dropout=set_embed_dropout)
        else:
            raise NotImplementedError()

        self.co_transformer = CoTransformer(layer=CoTransformerLayer(embed_dim, num_transformer_heads, dropout=dropout), num_layers=num_transformer_layers)

        self.item_attn = Attention(embed_dim=embed_dim, temperature=itm_temperature)
        self.set_attn = Attention(embed_dim=embed_dim, temperature=stm_temperature)

        self.items_bias = torch.zeros(items_total).cuda(1) 

        if attn_output:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.Sigmoid()
            )
        else:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

        self.attn_output = attn_output
        self.co_existence = co_existence
        self.coex_scores = coex_scores
        
        self.w1 = 0.2
        self.w2 = 0.1
        self.w3 = 0.1

    def forward(self, input_batch: TemporalSetsInput,  model_mode = 'train', return_fusion_weights: bool=False):
        
        items_seqs = input_batch.get_items() 
        sets_seqs = input_batch.get_sets()  
        item_times_seqs = input_batch.get_item_times()
        set_times_seqs = input_batch.get_set_times()
        
        targets = input_batch.get_set_targets()
        
        items_seqs_emb = [self.item_embed(items) for items in items_seqs]
        padded_items_seqs, items_padding_mask = pad_sequence(items_seqs_emb) 
        padded_item_times_seqs, _ = pad_sequence(item_times_seqs) 

        padded_items_seqs = time_encode(self.time_encoding_method, self.time_encoding, padded_items_seqs, padded_item_times_seqs)

        sets_seqs_embs = [[self.item_embed(user_set) for user_set in sets] for sets in sets_seqs]
        set_embed_seqs = set_embedding(sets_seqs_embs, self.set_embed)
        padded_set_embed_seqs, sets_padding_mask = pad_sequence(set_embed_seqs)
        padded_set_times_seqs, _ = pad_sequence(set_times_seqs)
        padded_set_embed_seqs = time_encode(self.time_encoding_method, self.time_encoding, padded_set_embed_seqs, padded_set_times_seqs)

        padded_items_seqs = torch.transpose(padded_items_seqs, 0, 1)
        padded_set_embed_seqs = torch.transpose(padded_set_embed_seqs, 0, 1)
        items_output, sets_output = self.co_transformer(padded_items_seqs, padded_set_embed_seqs, items_padding_mask, sets_padding_mask)
        items_output = torch.transpose(items_output, 0, 1)
        sets_output = torch.transpose(sets_output, 0, 1)

        if self.co_existence:
            coex_items_seqs = [self.coex_item_embed(items) for items in items_seqs]
            coex_padded_items_seqs, coex_padding_mask = pad_sequence(coex_items_seqs)

            coex_transformer_input = torch.transpose(coex_padded_items_seqs, 0, 1)
            coex_transformer_output = self.coex_transformer_encoder(coex_transformer_input, src_key_padding_mask=coex_padding_mask)
            coex_items_output = torch.transpose(coex_transformer_output, 0, 1)
            
            coex_items_embed = din(coex_items_output, coex_padding_mask, self.coex_item_embed.weight, self.coex_item_attn)
        if self.attn_output:

            item_user_embed = din(items_output, items_padding_mask, self.item_embed.weight, self.item_attn)
            set_user_embed = din(sets_output, sets_padding_mask, self.item_embed.weight, self.set_attn)
            assert item_user_embed.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            assert set_user_embed.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])

            item_weight = self.item_embed.weight.unsqueeze(0).expand(input_batch.batch_size, -1, -1)
            assert item_weight.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            tmp = torch.cat([item_user_embed, set_user_embed, item_weight], dim=-1)
            assert tmp.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim * 3])
            gate = self.gate_net(tmp)
            assert gate.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            ones = gate.new_ones((input_batch.batch_size, self.items_total, self.embed_dim), dtype=torch.float)

            if return_fusion_weights:
                return ones - gate, gate
            user_embed = (ones - gate) * item_user_embed + gate * set_user_embed
            assert user_embed.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            scores = (user_embed * self.item_embed.weight).sum(-1) + self.items_bias 
            
        else: 
            item_user_embed = items_output.mean(dim=-2)
            set_user_embed = sets_output.mean(dim=-2)

            assert item_user_embed.shape == torch.Size([input_batch.batch_size, self.embed_dim])
            assert set_user_embed.shape == torch.Size([input_batch.batch_size, self.embed_dim])

            tmp = torch.cat([item_user_embed, set_user_embed], dim=-1)
            assert tmp.shape == torch.Size([input_batch.batch_size, self.embed_dim * 2])
            gate = self.gate_net(tmp)
            assert gate.shape == torch.Size([input_batch.batch_size, self.embed_dim])
            ones = gate.new_ones((input_batch.batch_size, self.embed_dim), dtype=torch.float)
            user_embed = (ones - gate) * item_user_embed + gate * set_user_embed
            assert user_embed.shape == torch.Size([input_batch.batch_size, self.embed_dim])
            
            if model_mode == 'evaluate' or model_mode == 'test':
                scores = F.linear(user_embed, self.item_embed.weight, bias=self.items_bias)
                assert scores.shape == torch.Size([input_batch.batch_size, self.items_total])
                
                if self.co_existence and self.coex_scores:
                    co_scores = torch.bmm(coex_items_embed, torch.transpose(coex_items_embed, 1,2)).mean(2)
                    scores = (1-self.w1)*scores + self.w1*co_scores   
                    return scores
               
                if self.co_existence and not self.coex_scores:
                    return scores, coex_items_embed
                return scores

            elif model_mode == 'train':
                
                index = 0   
                sets_lh = []
                negatives = input_batch.get_set_negatives()
                k_kernels = input_batch.get_k_kernels()
                
                for pre_sets in sets_seqs:
                    
                    if self.co_existence:
                        edge_weight_emb = coex_items_embed[index]
                    else:
                        edge_weight_emb = self.item_embed.weight
                    
                    set_qualities = []
                    for pset in pre_sets:
                        pre_score = F.linear(user_embed[index]*self.w3, self.item_embed(pset)*self.w3, bias=self.items_bias[pset]*self.w3).mean(0, keepdim=True)
                        normed_item_sim = torch.matmul(edge_weight_emb[pset]*self.w2, edge_weight_emb[pset].t()*self.w2)
                        pre_edge_weight = torch.mean(normed_item_sim, dim=[0,1])
                        set_qualities.append((1-self.w1)*pre_score + self.w1*pre_edge_weight)  
                        
                    for tset in targets[index]:
                        tar_score_weight = F.linear(user_embed[index]*self.w3, self.item_embed(tset)*self.w3, bias=self.items_bias[tset]*self.w3).mean(0, keepdim=True)
                        normed_tar_sim = torch.matmul(edge_weight_emb[tset]*self.w2, edge_weight_emb[tset].t()*self.w2)
                        tar_edge_weight = torch.mean(normed_tar_sim, dim=[0,1])
                        set_qualities.append((1-self.w1)*tar_score_weight + self.w1*tar_edge_weight)  

                    for nset in negatives[index]:
                        neg_score_weight = F.linear(user_embed[index]*self.w3, self.item_embed(nset)*self.w3, bias=self.items_bias[nset]*self.w3).mean(0, keepdim=True)
                        normed_neg_sim = torch.matmul(edge_weight_emb[nset]*self.w2, edge_weight_emb[nset].t()*self.w2)
                        neg_edge_weight = torch.mean(normed_neg_sim, dim=[0,1])
                        set_qualities.append((1-self.w1)*neg_score_weight + self.w1*neg_edge_weight)
                    
                    all_set_qualities = torch.cat(set_qualities, 0) 
                    all_set_qualities = torch.exp(all_set_qualities)
                    selected_set_qualities = all_set_qualities[:len(pre_sets) + len(targets[index])]
                    neg_set_qualities = all_set_qualities[len(pre_sets) + len(targets[index]):]

                    all_set_q_diag = torch.diag_embed(all_set_qualities)
                    selected_set_q_diag = torch.diag_embed(selected_set_qualities)
                    neg_set_q_diag = torch.diag_embed(neg_set_qualities)
                    
                    all_set_diagI = torch.diag_embed(torch.FloatTensor([1e-4]*len(pre_sets) + [1]*(len(targets[index]) + len(negatives[index])))) 
                    selected_set_diagI = torch.diag_embed(torch.FloatTensor([1e-4]*(len(pre_sets) + len(targets[index]))))
                    neg_set_diagI = torch.diag_embed(torch.FloatTensor([1e-4]*len(negatives[index])))
                    
                    all_set_k_kernel = k_kernels[index] 
                    selected_set_k_kernel = all_set_k_kernel[:len(pre_sets)+len(targets[index]), :len(pre_sets)+len(targets[index])]
                    neg_set_k_kernel = all_set_k_kernel[len(pre_sets)+len(targets[index]):, len(pre_sets)+len(targets[index]):]
                    
                    all_set_l_kernel = torch.mm(torch.mm(all_set_q_diag, all_set_k_kernel), all_set_q_diag)
                    selected_set_l_kernel = torch.mm(torch.mm(selected_set_q_diag, selected_set_k_kernel), selected_set_q_diag)
                    neg_set_l_kernel = torch.mm(torch.mm(neg_set_q_diag, neg_set_k_kernel), neg_set_q_diag)
    
                    selected_det = torch.det(selected_set_l_kernel.cpu() + selected_set_diagI).cuda()  
                    all_det =  torch.det(all_set_l_kernel.cpu() + all_set_diagI).cuda()
                    neg_det = torch.det(neg_set_l_kernel.cpu() + neg_set_diagI).cuda()
                    
                    if selected_det/all_det > 0 and (1-neg_det/all_det) > 0 and neg_det > 0: 
                        set_lh = torch.log(selected_det/all_det) + torch.log(1-neg_det/all_det)  
                        sets_lh.append(set_lh)
                    index += 1
                    
                if len(sets_lh) > 0:
                    sdpp_loss = -torch.mean(torch.stack(sets_lh))
                    return sdpp_loss
                else:
                    return 0
                
        return scores