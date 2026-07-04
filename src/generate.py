#!/usr/bin/env python3
# GEMORNA 模型生成脚本 - 用于生成 CDS（编码序列）和 UTR（非翻译区）序列

# 导入必要的库
import pickle  # 用于加载词汇表文件
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import torch  # PyTorch 深度学习框架
import sys  # 系统相关功能
import argparse  # 命令行参数解析
import platform  # 检测操作系统平台

# 根据操作系统类型导入相应的共享模块
if platform.system() == "Darwin":  # macOS 系统
    from shared.libg2m import *
elif platform.system() == "Linux":  # Linux 系统
    from shared.mod_xzr01 import *
else:
    raise RuntimeError("Unsupported OS")  # 不支持的操作系统

# 导入配置和模型定义
from config import *  # 导入配置参数
from models.gemorna_cds import *  # CDS 模型（编码序列生成模型）
from models.gemorna_utr import *  # UTR 模型（非翻译区生成模型）

def has_noncanonical(protein_seq):
    """
    检查蛋白质序列中是否包含非标准氨基酸字符
    
    参数:
        protein_seq: 蛋白质序列字符串
    
    返回:
        bool: 如果包含非标准氨基酸返回 True，否则返回 False
    """
    canonical = set("ACDEFGHIKLMNPQRSTVWY*")  # 20 种标准氨基酸 + 终止密码子
    return any(residue not in canonical for residue in protein_seq.upper())

def main(args):
    """
    主函数：根据指定模式生成 CDS 或 UTR 序列
    
    参数:
        args: 命令行参数对象，包含模式、检查点路径、蛋白质序列和 UTR 长度等信息
    """
    # 解析命令行参数
    mode = args.mode  # 运行模式：'cds'（编码序列）、'5utr'（5'非翻译区）或 '3utr'（3'非翻译区）
    ckpt_path = args.ckpt_path  # 预训练模型的检查点文件路径
    protein_seq = args.protein_seq  # 输入的蛋白质序列（仅用于 CDS 模式）
    utr_len = args.utr_length  # UTR 序列的长度类别（short/medium/long）
    
    # 检测并设置计算设备（优先使用 GPU，如不可用则使用 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CDS 模式：根据蛋白质序列生成对应的编码 DNA 序列
    if mode == 'cds':
        # 验证必须提供蛋白质序列
        if protein_seq is None:
            raise ValueError(f'Please provide the protein sequence when using mode {mode}')

        # 检查蛋白质序列是否包含非标准氨基酸
        if has_noncanonical(protein_seq):
            raise ValueError(f'The input protein sequence contains non-canonical amino acid characters.')

        # 定义词汇表文件路径
        prot_vocab_path = './vocab/prot_vocab.pkl'  # 蛋白质词汇表路径
        cds_vocab_path = './vocab/cds_vocab.pkl'  # CDS（DNA）词汇表路径

        # 加载蛋白质和 CDS 词汇表
        with open(prot_vocab_path, 'rb') as f:
            prot_vocab = pickle.load(f)  # 蛋白质氨基酸到索引的映射
        with open(cds_vocab_path, 'rb') as f:
            cds_vocab = pickle.load(f)  # DNA 密码子到索引的映射

        # 加载 CDS 模型配置
        # 来源于 shared.mod_xzr01,GEMORNA_CDS_Config(input_dim=29, output_dim=347, hidden_dim=128, num_layers=12, num_heads=8, ff_dim=256, dropout=0.1, cnn_kernel_size=3, cnn_padding=1, prot_pad_idx=1, cds_pad_idx=1)
        model_config = GEMORNA_CDS_Config()

        # 构建编码器（Encoder）：将蛋白质序列编码为隐藏表示
        enc = Encoder(
            input_dim=model_config.input_dim,  # 输入维度（蛋白质词汇表大小）=29
            hid_dim=model_config.hidden_dim,  # 隐藏层维度=128
            n_layers=model_config.num_layers,  # Transformer 层数=12
            n_heads=model_config.num_heads,  # 多头注意力机制的头数=8
            pf_dim=model_config.ff_dim,  # 前馈网络维度=256
            dropout=model_config.dropout,  # Dropout 比率用于正则化=0.1
            cnn_kernel_size=model_config.cnn_kernel_size,  # CNN 卷积核大小=3
            cnn_padding=model_config.cnn_padding,  # CNN 填充大小=1
            device=device  # 计算设备
        )

        # 构建解码器（Decoder）：根据编码表示生成 CDS 序列
        dec = Decoder(
            output_dim=model_config.output_dim,  # 输出维度（CDS 词汇表大小）=347
            hid_dim=model_config.hidden_dim,  # 隐藏层维度=128
            n_layers=model_config.num_layers,  # Transformer 层数=12
            n_heads=model_config.num_heads,  # 多头注意力机制的头数=8
            pf_dim=model_config.ff_dim,  # 前馈网络维度=256
            dropout=model_config.dropout,  # Dropout 比率=0.1
            device=device  # 计算设备
        )

        # 组装完整的 CDS 生成模型（编码器-解码器架构）
        # CDS 函数来源于 shared.mod_xzr01
        model = CDS(enc, dec, model_config.prot_pad_idx, model_config.cds_pad_idx, device)

        # 加载预训练模型权重
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)  # 将模型移到指定设备
        model.eval()  # 设置为评估模式（关闭 dropout 等训练特性）
        
        # 执行序列生成：根据输入的蛋白质序列生成对应的 CDS
        model.gen(protein_seq, prot_vocab, cds_vocab, device)

    # UTR 模式：生成 5'UTR 或 3'UTR 序列
    else:
        # 根据不同的 UTR 类型加载对应的配置和词汇表
        if mode == '5utr':
            # 5'UTR 模型配置
            # GEMORNA_5UTR_Config(block_size=768, vocab_size=512, n_layer=12, n_head=12, n_embd=144, dropout=0.1, bias=True)
            model_config = GEMORNA_5UTR_Config()
            # 5'UTR 词汇表
            # {'<sos>': 1, 'ACA': 2, 'CAG': 3, 'CGG': 4, 'GAA': 5, 'GGG': 6, 'AUU': 7, 'GCG': 8, 'AGC': 9, 'UGG': 10, 'UCG': 11, 'GAC': 12, 'CAA': 13, 'CCU': 14, 'GAG': 15, 'AAG': 16, 'ACC': 17, 'CCC': 18, 'GUC': 19, '<eos>': 20, 'AAC': 21, 'AAA': 22, 'CUG': 23, 'AUA': 24, 'CCA': 25, 'GCA': 26, 'UCU': 27, 'UGU': 28, 'UUG': 29, 'UUU': 30, 'UCC': 31, 'UCA': 32, 'AAU': 33, 'CAU': 34, 'CAC': 35, 'CGU': 36, 'UAU': 37, 'GAU': 38, 'GUG': 39, 'AGU': 40, 'CGA': 41, 'GGC': 42, 'GGU': 43, 'GGA': 44, 'AGG': 45, 'AGA': 46, 'ACU': 47, 'UAC': 48, 'UUC': 49, 'AUC': 50, 'CGC': 51, 'GCU': 52, 'CCG': 53, 'UUA': 54, 'GUA': 55, 'CUA': 56, 'GCC': 57, 'CUC': 58, 'CUU': 59, 'GUU': 60, 'AUG': 61, 'UAG': 62, 'UGC': 63, 'UGA': 64, 'ACG': 65, 'UAA': 66, 'GGN': 67, 'UNN': 68, 'CNN': 69, 'CUN': 70, 'ANN': 71, 'GAN': 72, 'AAN': 73, 'GNN': 74, 'AGN': 75, 'UUN': 76, 'UGN': 77, 'CAN': 78, 'GCN': 79, 'CGN': 80, 'UCN': 81, 'CCN': 82, 'UAN': 83, 'AUN': 84, 'GUN': 85, 'ACN': 86, 'NNC': 87, 'NCC': 88, 'NGG': 89, 'NGC': 90, 'GNG': 91, 'CNG': 92, 'CNA': 93, 'NCA': 94, 'NCU': 95, 'NAG': 96, 'NNN': 97, 'NNU': 98, 'NUU': 99, 'NGN': 100, 'NGA': 101, 'NNA': 102, 'NAA': 103, 'NNG': 104, 'NCG': 105, 'GNC': 106, 'ANC': 107, 'NUC': 108, 'CNU': 109, 'GNA': 110, 'NAC': 111, 'ANU': 112, 'CNC': 113, 'GNU': 114, 'NGU': 115, 'UNC': 116, 'UNU': 117, 'NAU': 118, 'NUG': 119, 'ANA': 120, 'UNG': 121, 'ANG': 122, 'NUA': 123, 'CRG': 124, 'SCA': 125, 'AMU': 126, 'KCY': 127, 'MGC': 128, 'RCC': 129, 'GYU': 130, 'RAC': 131, 'RGA': 132, 'GCS': 133, 'AAM': 134, 'CGY': 135, 'CVU': 136, 'AGK': 137, 'GKA': 138, 'AGR': 139, 'SUG': 140, 'UNA': 141, 'CMA': 142, 'CCK': 143, 'RGC': 144, 'YAA': 145, 'RGU': 146, 'ACY': 147, 'GGS': 148, 'ASC': 149, 'UYC': 150, 'YCU': 151, 'ABG': 152, 'UUD': 153, 'RAA': 154, 'CYC': 155, 'KCC': 156, 'YCG': 157, 'CCW': 158, 'GGR': 159, 'AAR': 160, 'RUC': 161, 'GUR': 162, 'KUU': 163, 'RCU': 164, 'GGK': 165, 'AYC': 166, 'GRG': 167, 'GUK': 168, 'UKA': 169, 'UCY': 170, 'YCC': 171, 'CCR': 172, 'UUS': 173, 'CYU': 174, 'AYG': 175, 'YGG': 176, 'USU': 177, 'AUY': 178, 'UYG': 179, 'SGA': 180, 'YUG': 181, 'GAY': 182, 'YGC': 183, 'MUG': 184, 'ARC': 185, 'GMC': 186, 'URG': 187, 'SCC': 188, 'GCY': 189, 'MGU': 190, 'CGR': 191, 'GGY': 192, 'GYC': 193, 'CRU': 194, 'YUC': 195, 'CCY': 196, 'AKG': 197, 'CSU': 198, 'ARG': 199, 'GAK': 200, 'GAR': 201, 'ARA': 202, 'GSC': 203, 'GRC': 204, 'RGG': 205, 'RCG': 206, 'CWC': 207, 'WCU': 208, 'CMC': 209, 'ARU': 210, 'AGY': 211, 'ACR': 212, 'GCW': 213, 'CYG': 214, 'UKG': 215, 'YUU': 216, 'YAU': 217, 'AMC': 218, 'MGG': 219, 'GCR': 220, 'GCM': 221, 'SCU': 222, 'YGA': 223, 'ACS': 224, 'GUY': 225, 'CAR': 226, 'WAC': 227, 'SUA': 228, 'UMG': 229, 'CUS': 230, 'GMG': 231, 'GGM': 232, 'CUR': 233, 'UCR': 234, 'AKU': 235, 'CGM': 236, 'CKC': 237, 'GWA': 238, 'UYA': 239, 'RAU': 240, 'SGG': 241, 'CCS': 242, 'MAC': 243, 'CAY': 244, 'MCA': 245, 'AYA': 246, 'CSG': 247, 'UGM': 248, 'UYU': 249, 'GYA': 250, 'YAC': 251, 'GCK': 252, 'CYA': 253, 'YGU': 254, 'AKC': 255, 'ASU': 256, 'UAY': 257, 'CRC': 258, 'KAC': 259, 'GYG': 260, 'URU': 261, 'UUM': 262, 'MGA': 263, 'CWU': 264, 'AUS': 265, 'UCM': 266, 'GAM': 267, 'GAS': 268, 'YUA': 269, 'WGC': 270, 'GRA': 271, 'RAG': 272, 'USN': 273, 'AMA': 274, 'ASG': 275, 'ACK': 276, 'UCK': 277, 'UGR': 278, 'GUM': 279, 'KAA': 280, 'GUS': 281, 'URC': 282, 'YCA': 283, 'UUR': 284, 'GRM': 285, 'UCW': 286, 'UGY': 287, 'GKG': 288, 'GKU': 289, 'CRA': 290, 'CSC': 291, 'AGW': 292, 'AAW': 293, 'RUA': 294, 'UCS': 295, 'CSA': 296, 'AGS': 297, 'MCG': 298, 'YAG': 299, 'SUC': 300, 'GRY': 301, 'RYU': 302, 'RUG': 303, 'SAG': 304, 'MUU': 305, 'CCM': 306, 'MCU': 307, 'RUU': 308, 'UUK': 309, 'RCA': 310, 'CAS': 311, 'GWG': 312, 'CWS': 313, 'KGC': 314, 'WCG': 315, 'GMU': 316, 'WCC': 317, 'CUY': 318, 'KGA': 319, 'KGU': 320, 'GMR': 321, 'MGN': 322, 'AMG': 323, 'GSG': 324, 'SGC': 325, 'KCA': 326, 'CWG': 327, 'AUW': 328, 'URA': 329, 'AAY': 330, 'GMA': 331, 'MAG': 332, 'SAA': 333, 'AYU': 334, 'UKK': 335, 'KGG': 336, 'MCC': 337, 'SAN': 338, 'ASA': 339, 'UAS': 340, 'SAU': 341, 'RNN': 342, 'MUA': 343, 'CMG': 344, 'CKG': 345, 'WCA': 346, 'RYC': 347, 'KUA': 348, 'UUY': 349, 'SAC': 350, 'CGS': 351, 'UGS': 352, 'UKY': 353, 'WAA': 354, 'ACM': 355, 'KUC': 356, 'UYY': 357, 'UWU': 358, 'CUK': 359, 'ACW': 360, 'CWA': 361, 'WGG': 362, 'SGU': 363, 'AAK': 364, 'WUA': 365, 'ARN': 366, 'GRU': 367, 'UAR': 368, 'UKC': 369, 'AWU': 370, 'UMC': 371, 'WAG': 372, 'UKU': 373, 'KUG': 374, 'UGK': 375, 'CAK': 376, 'MUC': 377, 'UAM': 378, 'MAU': 379, 'CKU': 380, 'UMU': 381, 'GWC': 382, 'KCG': 383, 'URY': 384, 'CYR': 385, 'AGM': 386, 'CSK': 387, 'GKC': 388, 'MNN': 389, 'WUC': 390, 'SUU': 391, 'USA': 392, 'MAA': 393, 'WUU': 394, 'KCU': 395, 'USC': 396, 'UMA': 397, 'UUW': 398, 'CUM': 399, 'GUW': 400, 'GSA': 401, 'GWU': 402, 'KAU': 403, 'SCG': 404, 'YKG': 405, 'UGW': 406, 'WSG': 407, 'WGA': 408, 'CAM': 409, 'USK': 410, 'USG': 411, 'AWG': 412, 'UAW': 413, 'UWC': 414, 'GYR': 415, 'WAU': 416, 'CWK': 417, 'WKU': 418, 'SSU': 419, 'AUR': 420, 'KAG': 421, 'UMK': 422, 'AAS': 423, 'CMU': 424, 'GWR': 425, 'GAW': 426, 'RCN': 427, 'AKA': 428, 'GMW': 429, 'MWG': 430, 'MCN': 431, 'CUW': 432, 'GGW': 433, 'CKA': 434, 'SGN': 435, 'KMC': 436, 'CYY': 437, 'WGN': 438, 'CGK': 439, 'YMG': 440, 'CYM': 441, 'UWA': 442, 'SWC': 443, 'GSR': 444, 'WGU': 445, 'UAK': 446, 'CYN': 447, 'GRR': 448, 'YCN': 449, 'AWC': 450, 'AYN': 451, 'VUA': 452, 'NCN': 453, 'DNN': 454, 'NCB': 455, 'NAN': 456, 'UWG': 457, 'DUG': 458, 'VAB': 459, 'WWG': 460, 'GSU': 461, 'CAW': 462, 'UYW': 463, 'WUG': 464, 'ASN': 465, 'RGN': 466, 'AUK': 467, 'CGW': 468, 'YRC': 469, 'AWA': 470, 'CKS': 471, 'SGS': 472}
            vocab = five_prime_utr_vocab

        elif mode == '3utr':
            # 3'UTR 模型配置
            # GEMORNA_3UTR_Config(block_size=1024, vocab_size=448, n_layer=12, n_head=12, n_embd=288, dropout=0.1, bias=True)
            model_config = GEMORNA_3UTR_Config()
            # 3'UTR 词汇表
            # {'<sos>': 1, 'AGC': 2, 'AAG': 3, 'GGC': 4, 'AGA': 5, 'AUG': 6, 'AAA': 7, 'GCA': 8, 'CUG': 9, 'UGC': 10, 'GCU': 11, 'UCC': 12, 'CAU': 13, 'UAA': 14, 'UUC': 15, 'CUU': 16, 'CCC': 17, 'UGU': 18, 'GUU': 19, 'GGU': 20, 'GGG': 21, 'GGA': 22, 'GAC': 23, 'CAA': 24, 'CGC': 25, 'GAA': 26, 'CAC': 27, 'GUA': 28, 'CCU': 29, 'GCC': 30, 'UGG': 31, 'AUU': 32, 'UCG': 33, 'GAU': 34, 'ACA': 35, 'UAC': 36, 'CUC': 37, 'AAC': 38, 'UGA': 39, 'ACU': 40, 'CAG': 41, 'AGU': 42, 'AAU': 43, 'ACG': 44, 'AGG': 45, 'UUG': 46, 'CAN': 47, '<eos>': 48, 'GAG': 49, 'CGA': 50, 'UCU': 51, 'ACC': 52, 'AUC': 53, 'GUC': 54, 'GUG': 55, 'CCA': 56, 'UAU': 57, 'CGG': 58, 'UUU': 59, 'UUA': 60, 'AUA': 61, 'CUA': 62, 'GCG': 63, 'UCA': 64, 'CGU': 65, 'UNN': 66, 'UAG': 67, 'CCG': 68, 'UGN': 69, 'GUN': 70, 'CUN': 71, 'UUN': 72, 'ACN': 73, 'GCN': 74, 'ANN': 75, 'AUN': 76, 'AGN': 77, 'CCN': 78, 'GNN': 79, 'AAN': 80, 'UAN': 81, 'CNN': 82, 'UCN': 83, 'GAN': 84, 'UNU': 85, 'GGN': 86, 'CGN': 87, 'NNN': 88, 'NNA': 89, 'NGA': 90, 'NGG': 91, 'NCC': 92, 'GNG': 93, 'ANC': 94, 'ANU': 95, 'NAU': 96, 'NUU': 97, 'NNC': 98, 'NCA': 99, 'NAA': 100, 'NNU': 101, 'NAC': 102, 'NUA': 103, 'NCG': 104, 'NCU': 105, 'NUC': 106, 'NGC': 107, 'UNC': 108, 'NNG': 109, 'NUG': 110, 'ANG': 111, 'NAG': 112, 'NGU': 113, 'CNU': 114, 'NAN': 115, 'ANA': 116, 'CNC': 117, 'UNG': 118, 'UNA': 119, 'CNA': 120, 'GNU': 121, 'NUN': 122, 'GNC': 123, 'CNG': 124, 'GNA': 125, 'NGN': 126, 'NCN': 127, 'ARU': 128, 'GUW': 129, 'CUK': 130, 'GGR': 131, 'UMU': 132, 'YUU': 133, 'AWA': 134, 'GUY': 135, 'YCC': 136, 'CGR': 137, 'MCG': 138, 'RUC': 139, 'RGA': 140, 'CCK': 141, 'GCM': 142, 'YUC': 143, 'KGC': 144, 'CRG': 145, 'GCY': 146, 'AYA': 147, 'ARA': 148, 'UUR': 149, 'GRC': 150, 'CWC': 151, 'YGA': 152, 'RGG': 153, 'CKG': 154, 'GGS': 155, 'KGU': 156, 'AMA': 157, 'CYG': 158, 'CCY': 159, 'CCR': 160, 'GAR': 161, 'GSC': 162, 'GCW': 163, 'AKC': 164, 'AUR': 165, 'UAW': 166, 'UGY': 167, 'SUC': 168, 'KAA': 169, 'URA': 170, 'ARC': 171, 'GKA': 172, 'CRA': 173, 'GYU': 174, 'UYG': 175, 'UUY': 176, 'AYG': 177, 'AYU': 178, 'YAG': 179, 'URC': 180, 'CWG': 181, 'MUC': 182, 'GWC': 183, 'USU': 184, 'RUU': 185, 'CAY': 186, 'AKG': 187, 'RCU': 188, 'CRU': 189, 'UUW': 190, 'YAA': 191, 'RGU': 192, 'GAS': 193, 'URU': 194, 'CKC': 195, 'RUG': 196, 'YUA': 197, 'UGR': 198, 'AAW': 199, 'WCC': 200, 'AUY': 201, 'AWC': 202, 'UWC': 203, 'GCS': 204, 'RAC': 205, 'RAG': 206, 'CMC': 207, 'GYG': 208, 'RUA': 209, 'YCU': 210, 'AUW': 211, 'CWU': 212, 'YAC': 213, 'UCY': 214, 'GSG': 215, 'SAU': 216, 'WUU': 217, 'RCG': 218, 'RCC': 219, 'GAY': 220, 'CUY': 221, 'CAR': 222, 'RAU': 223, 'KAC': 224, 'MGA': 225, 'UYA': 226, 'YGC': 227, 'CCW': 228, 'ACY': 229, 'CMG': 230, 'KCU': 231, 'YUG': 232, 'CUS': 233, 'GUR': 234, 'KNN': 235, 'KGG': 236, 'CSG': 237, 'CYC': 238, 'YGU': 239, 'ACK': 240, 'AAR': 241, 'KCA': 242, 'ACS': 243, 'RAA': 244, 'MGU': 245, 'AGK': 246, 'MGC': 247, 'UWA': 248, 'SUU': 249, 'KCG': 250, 'GRA': 251, 'MGG': 252, 'CGK': 253, 'YCA': 254, 'AAM': 255, 'SCC': 256, 'CUR': 257, 'CYU': 258, 'WGC': 259, 'UYU': 260, 'UGS': 261, 'CSA': 262, 'GCR': 263, 'UGK': 264, 'GMC': 265, 'UWW': 266, 'YGG': 267, 'RCA': 268, 'MAU': 269, 'GRU': 270, 'CYA': 271, 'GGY': 272, 'UYC': 273, 'SUG': 274, 'SAG': 275, 'UAY': 276, 'ACR': 277, 'KCC': 278, 'UKC': 279, 'URG': 280, 'UCR': 281, 'GYA': 282, 'AAK': 283, 'GYR': 284, 'GSU': 285, 'ASA': 286, 'GYC': 287, 'WGG': 288, 'CGY': 289, 'UCW': 290, 'YAU': 291, 'WUG': 292, 'AWG': 293, 'UUM': 294, 'CSC': 295, 'SMA': 296, 'ARG': 297, 'GCK': 298, 'YCG': 299, 'AGM': 300, 'GRG': 301, 'ACM': 302, 'UKU': 303, 'KUG': 304, 'KUK': 305, 'RGC': 306, 'WAU': 307, 'GKG': 308, 'UMC': 309, 'GKU': 310, 'AKA': 311, 'AGY': 312, 'CMU': 313, 'AYC': 314, 'MUG': 315, 'AUM': 316, 'CSU': 317, 'AUK': 318, 'UAR': 319, 'UGM': 320, 'KUC': 321, 'WCA': 322, 'GUM': 323, 'AMC': 324, 'SAA': 325, 'AGR': 326, 'CRC': 327, 'GWU': 328, 'WAG': 329, 'CAW': 330, 'CUW': 331, 'AAY': 332, 'UWU': 333, 'MUA': 334, 'ASC': 335, 'ASG': 336, 'CAM': 337, 'KAG': 338, 'AAS': 339, 'HGG': 340, 'UKA': 341, 'UAS': 342, 'CCM': 343, 'MAA': 344, 'GWA': 345, 'WUC': 346, 'MWY': 347, 'URW': 348, 'KAU': 349, 'AKU': 350, 'AMG': 351, 'UWG': 352, 'GWG': 353, 'KUA': 354, 'CGW': 355, 'CKU': 356, 'UUK': 357, 'GKC': 358, 'AMU': 359, 'UCK': 360, 'USG': 361, 'GMG': 362, 'CKA': 363, 'MCU': 364, 'KKU': 365, 'AGS': 366, 'UMG': 367, 'CAK': 368, 'AGW': 369, 'UMA': 370, 'MRU': 371, 'KGA': 372, 'GMU': 373, 'WUA': 374, 'GGK': 375, 'UAK': 376, 'KUU': 377, 'GUS': 378, 'WAA': 379, 'UAM': 380, 'UCS': 381, 'WAC': 382, 'SGA': 383, 'WCU': 384, 'CWA': 385, 'SGU': 386, 'CAS': 387, 'MCC': 388, 'AUS': 389, 'SCU': 390, 'CUM': 391, 'GAK': 392, 'SAC': 393, 'MUU': 394, 'CCS': 395, 'SUA': 396, 'ASU': 397, 'GMA': 398, 'WGU': 399, 'GGM': 400, 'UKG': 401, 'AWU': 402, 'MAG': 403, 'UCM': 404, 'MCA': 405, 'UUS': 406, 'CMA': 407, 'GUK': 408, 'GAM': 409, 'GAW': 410, 'YKA': 411, 'ARK': 412, 'CGS': 413, 'UGW': 414, 'USA': 415, 'UKR': 416, 'VAG': 417, 'AAD': 418, 'GKY': 419, 'WGA': 420, 'SGC': 421, 'CGM': 422, 'GGW': 423, 'UYN': 424, 'WWA': 425, 'WCG': 426, 'SYA': 427, 'MAC': 428, 'SCA': 429, 'RUN': 430, 'SNN': 431}
            vocab = three_prime_utr_vocab  
        else:
            # 无效的模式参数
            print("Wrong mode!")
            sys.exit()

        # 构建 UTR 生成模型
        model = UTR(model_config)
        # 加载预训练模型权重（注意：检查点文件中模型权重存储在 'model' 键下）
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        model.to(device)  # 将模型移到指定设备
        model.eval()  # 设置为评估模式
        
        # 执行 UTR 序列生成：根据指定的长度类别生成 UTR 序列
        model.gen(mode, vocab, device, utr_len)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("GEMORNA 序列生成脚本 - 用于生成 CDS 和 UTR 序列")
    
    # 定义命令行参数
    parser.add_argument("--mode", type=str, help="生成模式：'cds'（编码序列）、'5utr'（5'非翻译区）或 '3utr'（3'非翻译区）")
    parser.add_argument("--ckpt_path", type=str, help="预训练模型检查点文件的路径")
    parser.add_argument("--protein_seq", type=str, help="输入的蛋白质序列（仅 CDS 模式需要）")
    parser.add_argument("--utr_length", type=str, help="UTR 序列的长度类别：short（短）、medium（中）或 long（长）")
    
    # 解析命令行参数并执行主函数
    args = parser.parse_args()
    main(args)


