import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from config import *
from tokenization import tokenize_aa

CDS_CHUNK_LENGTH = 170


def trunc_protein_seq(seq):
    seq = seq.lower()
    return [
        seq[i:i + CDS_CHUNK_LENGTH]
        for i in range(0, len(seq), CDS_CHUNK_LENGTH)
    ] or [seq]


class CDS_(nn.Module):
    def __init__(self, encoder, decoder, prot_pad_idx, cds_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prot_pad_idx = prot_pad_idx
        self.cds_pad_idx = cds_pad_idx
        self.device = device

    def make_prot_mask(self, prot):
        return (prot != self.prot_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_cds_mask(self, cds):
        cds_pad_mask = (cds != self.cds_pad_idx).unsqueeze(1).unsqueeze(2)
        cds_len = cds.shape[1]
        cds_sub_mask = torch.tril(torch.ones((cds_len, cds_len), device=self.device)).bool()
        cds_mask = cds_pad_mask & cds_sub_mask
        return cds_mask
    # 注：这里源码应该有笔误，prot 应对应的是 protein_input，cds 应对应的是 cds_input
    # def forward(self, prot, cds):
    #     protein_mask = self.make_prot_mask(protein_input)
    #     cds_mask = self.make_cds_mask(cds_input)
    #     encoded_protein = self.encoder(protein_input, protein_mask)
    #     decoded_output, attn_weights = self.decoder(
    #         cds_input, encoded_protein, cds_mask, protein_mask
    #     )
    #     return decoded_output, attn_weights

    # 修正后的函数
    def forward(self, prot, cds):
        protein_mask = self.make_prot_mask(prot)
        cds_mask = self.make_cds_mask(cds)
        encoded_protein = self.encoder(prot, protein_mask)
        decoded_output, attn_weights = self.decoder(cds, encoded_protein, cds_mask, protein_mask)
        return decoded_output, attn_weights

    @staticmethod
    def _stoi(vocab, token):
        if hasattr(vocab, "stoi"):
            return vocab.stoi[token]
        return vocab[token]

    @staticmethod
    def _itos(vocab, index):
        return vocab.itos[index]

    @torch.no_grad()
    def sampling(self, enc_prot, prot_mask, tokens, cds_vocab, device, SEED):
        torch.manual_seed(SEED)

        input_ids = torch.LongTensor([[self._stoi(cds_vocab, init_token)]]).to(device)
        generated_seq = []
        model_score = 0.0

        for cur_length in range(1, len(tokens)):
            cds_mask = self.make_cds_mask(input_ids)
            output, _ = self.decoder(input_ids, enc_prot, cds_mask, prot_mask)

            current_source_token = tokens[cur_length]
            if current_source_token == eos_token:
                break

            logits_last = output[:, -1, :].squeeze(0)
            normalized_probs = F.softmax(logits_last, dim=-1)
            sharpened_probs = normalized_probs.pow(2.3)
            sharpened_probs = sharpened_probs / sharpened_probs.sum()
            pred_token = torch.multinomial(sharpened_probs, 1).item()
            current_target_token = self._itos(cds_vocab, pred_token)

            if current_target_token != eos_token:
                if current_target_token not in codon_dict[current_source_token]:
                    current_target_token = codon_freq[current_source_token][0]
                    pred_token = self._stoi(cds_vocab, current_target_token)

            generated_seq.append(current_target_token)
            model_score += torch.log(normalized_probs[pred_token]).item()

            next_token = torch.LongTensor([[pred_token]]).to(device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return generated_seq, model_score

    @torch.no_grad()
    def gen(self, protein_seq, prot_vocab, cds_vocab, device):
        """Generate CDS using the same sampling flow as the closed Cython CDS.gen."""
        self.eval()

        SEED = random.randint(1, 200)
        generated_seqs = []
        final_modelscore = 0.0

        for seq in trunc_protein_seq(protein_seq):
            # 1. 蛋白质序列转为小写并分词（prot_vocab 的键是小写字母）
            protein_tokens = tokenize_aa(seq)
            tokens = [init_token] + protein_tokens + [eos_token]

            # 2. 构建蛋白质索引列表：<sos> + 氨基酸索引 + <eos>
            prot_indexes = [self._stoi(prot_vocab, token) for token in tokens]

            # 3. 转为 tensor 并编码蛋白质
            prot_tensor = torch.LongTensor(prot_indexes).unsqueeze(0).to(device)
            prot_mask = self.make_prot_mask(prot_tensor)
            enc_prot = self.encoder(prot_tensor, prot_mask)

            generated_seq, model_score = self.sampling(
                enc_prot, prot_mask, tokens, cds_vocab, device, SEED
            )
            generated_seqs.extend(generated_seq)
            final_modelscore += model_score

        final_seq = ''.join(generated_seqs).upper()
        final_naturalness = math.exp(final_modelscore / len(generated_seqs))

        print(f'\nGenerated CDS & Naturalness')
        print(f'{final_seq} {final_naturalness:.2f}\n')

        return None
        
