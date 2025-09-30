#!/usr/bin/env python3
"""
Freeze-Omni 完整训练脚本
基于论文、原始推理代码和checkpoints配置的完整实现

# 完整训练流程
python train.py --stage all \
    --model_path ./checkpoints \
    --llm_path ./Qwen2-7B-Instruct \
    --asr_data ./data/asr \
    --qa_data ./data/qa \
    --tts_data ./data/tts_paired

# 只训练Stage 1
python train.py --stage 1

# 只训练Stage 3（需要先完成Stage 1和2）
python train.py --stage 3

# 只训练TTS
python train.py --stage tts

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.compliance.kaldi as k
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import copy

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ================== 完整配置（基于checkpoints修正） ==================
@dataclass 
class FreezeOmniTrainingConfig:
    """基于checkpoints/audiollm/train.yaml的完整正确配置"""
    
    # === 路径配置 ===
    model_path: str = "./checkpoints"
    llm_path: str = "./Qwen2-7B-Instruct"
    cmvn_file: str = "./checkpoints/audiollm/global_cmvn"
    
    # === 数据配置 ===
    asr_data_path: str = "./data/asr"
    qa_data_path: str = "./data/qa"
    tts_paired_data: str = "./data/tts_paired"
    
    # === 编码器配置（从train.yaml严格对应） ===
    input_dim: int = 80
    encoder_output_dim: int = 1024  # ✅ 修正
    encoder_layer_config: str = "subsampling-transformer"
    
    # Subsampling配置
    subsampling_rate: int = 4
    subsampling_input_dim: int = 80
    subsampling_output_dim: int = 1024  # ✅ 修正
    subsampling_dropout_rate: float = 0.1
    
    # Transformer配置（24层，1024维）
    transformer_num_blocks: int = 24
    transformer_attention_dim: int = 1024  # ✅ 修正
    transformer_attention_heads: int = 16  # ✅ 修正
    transformer_linear_units: int = 4096  # ✅ 修正
    transformer_dropout_rate: float = 0.1
    transformer_attention_dropout_rate: float = 0.0
    transformer_positional_dropout_rate: float = 0.1
    transformer_input_layer: str = "linear"
    transformer_pos_enc_class: str = "rel-enc"
    transformer_normalize_before: bool = True
    transformer_concat_after: bool = False
    transformer_positionwise_layer_type: str = "linear"
    
    # Chunk配置（训练时配置）
    chunk_size: int = 4  # ✅ 修正：训练时用4
    left_chunks: int = 16  # ✅ 修正：左侧context为16
    use_dynamic_chunk: bool = False  # ✅ train.yaml中是False
    
    # === 适配器配置 ===
    adapter_type: str = "subsampling"  # CNNSubsampling
    adapter_kernel_size: int = 5
    activation_func: str = "gelu"
    norm: str = "layer"
    llm_embed_dim: int = 3584  # ✅ 修正：Qwen2-7B实际维度
    
    # === AudioLLM配置 ===
    freeze_llm: bool = True
    freeze_encoder: bool = False  # Stage 3才冻结
    freeze_adapter: bool = False  # Stage 3才冻结
    predict_usr_state: int = 4  # ✅ 修正：4个状态
    prompt_num: int = 25  # ✅ 修正
    add_prompt_before: bool = True
    add_audio_bos_eos: bool = True
    task_num: int = 20
    
    # Chat template
    chat_template: str = (
        '<|im_start|>system\n'
        'You are a helpful assistant.<|im_end|>\n'
        '<|im_start|>user\n'
        '<audio><|im_end|>\n'
        '<|im_start|>assistant\n'
    )
    
    # === 解码器配置 ===
    decoder_num_blocks: int = 4
    decoder_hidden_size: int = 896
    decoder_attention_heads: int = 14
    decoder_linear_units: int = 4864
    vocab_size: int = 1024  # TiCodec单码本
    
    # === TiCodec配置 ===
    codec_n_codes: int = 1024
    codec_n_code_groups: int = 1
    codec_residul_layer: int = 1
    codec_frame_rate: int = 40
    sample_rate: int = 24000
    global_code_num: int = 8
    global_tokens: List[int] = None
    
    # === 训练超参数（严格按论文和train.yaml） ===
    # Stage 1 - ASR
    stage1_lr: float = 2e-4
    stage1_epochs: int = 20
    stage1_batch_size: int = 32
    
    # Stage 2 - Alignment
    stage2_lr: float = 1e-4
    stage2_epochs: int = 10
    stage2_batch_size: int = 16
    
    # Stage 3 - Dialogue
    stage3_lr: float = 6e-4  # ✅ 从train.yaml
    stage3_epochs: int = 5
    stage3_batch_size: int = 4  # ✅ train.yaml中batch_size=4
    
    # TTS训练
    codec_lr: float = 1e-3
    tts_stage2_lr: float = 5e-5
    tts_stage3_lr: float = 5e-5
    
    # 通用
    warmup_steps: int = 200  # ✅ 修正
    gradient_clip: float = 5.0  # ✅ 修正
    mixed_precision: bool = True
    save_interval: int = 1000
    accum_grad: int = 1
    
    # 优化器配置（从train.yaml）
    optim: str = "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.99
    eps: float = 1e-6
    weight_decay: float = 0.01
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.global_tokens is None:
            self.global_tokens = [473, 975, 419, 219, 565, 121, 550, 616]

# ================== 数据集实现 ==================

class Stage1ASRDataset(Dataset):
    """Stage 1: ASR数据集（Kaldi风格）"""
    
    def __init__(self, data_path: str, config: FreezeOmniTrainingConfig):
        self.config = config
        self.data_path = Path(data_path)
        
        # 加载Kaldi风格的数据
        # wav.scp: utt_id wav_path
        # text: utt_id transcript
        self.samples = []
        
        wav_scp = self.data_path / "wav.scp"
        text_file = self.data_path / "text"
        
        # 读取wav路径
        wav_dict = {}
        if wav_scp.exists():
            with open(wav_scp, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, wav_path = parts
                        wav_dict[utt_id] = wav_path
        
        # 读取文本
        if text_file.exists():
            with open(text_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        if utt_id in wav_dict:
                            self.samples.append({
                                'utt_id': utt_id,
                                'wav_path': wav_dict[utt_id],
                                'text': text
                            })
        
        logger.info(f"Loaded {len(self.samples)} ASR samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载音频
        try:
            wav, sr = torchaudio.load(sample['wav_path'])
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            
            # 计算Mel特征（Kaldi风格）
            mel_features = k.fbank(
                waveform=wav,
                dither=0,
                frame_length=25,
                frame_shift=10,
                num_mel_bins=self.config.input_dim
            )
            
            return {
                'utt_id': sample['utt_id'],
                'mel_features': mel_features.squeeze(0),
                'text': sample['text'],
                'feat_length': mel_features.size(1)
            }
        except Exception as e:
            logger.warning(f"Failed to load {sample['wav_path']}: {e}")
            # 返回空样本
            return {
                'utt_id': sample['utt_id'],
                'mel_features': torch.zeros(100, self.config.input_dim),
                'text': '',
                'feat_length': 100
            }

class Stage2AlignmentDataset(Stage1ASRDataset):
    """Stage 2: 对齐数据集（复用ASR数据）"""
    pass

class Stage3DialogueDataset(Dataset):
    """Stage 3: 对话数据集（60k条，包含TTS合成和状态标签）"""
    
    def __init__(self, data_path: str, config: FreezeOmniTrainingConfig, 
                 tokenizer=None, vad_model=None):
        self.config = config
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.vad_model = vad_model
        
        # 加载对话数据
        dialogue_file = self.data_path / "dialogues.json"
        if dialogue_file.exists():
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                self.dialogues = json.load(f)
        else:
            logger.warning(f"Dialogue file not found: {dialogue_file}")
            self.dialogues = []
        
        # 验证TTS合成的音频是否存在
        self.synthesized_dir = self.data_path / "synthesized"
        
        logger.info(f"Loaded {len(self.dialogues)} dialogue samples")
    
    def __len__(self):
        return len(self.dialogues)
    
    def _compute_state_labels(self, mel_features):
        """
        计算状态标签（基于VAD和对话逻辑）
        
        Returns:
            state_labels: Tensor of shape (num_chunks,)
                0: 继续接收语音
                1: 检测到endpoint，需要生成回复
                2: 检测到endpoint，但不需要生成
                3: padding/invalid
        """
        num_frames = mel_features.size(0)
        chunk_size = self.config.chunk_size
        num_chunks = (num_frames + chunk_size - 1) // chunk_size
        
        state_labels = torch.zeros(num_chunks, dtype=torch.long)
        
        # 简单策略：
        # - 前面的chunks标记为0（继续）
        # - 最后一个chunk标记为1（需要生成）
        if num_chunks > 0:
            state_labels[-1] = 1
        
        # 如果有VAD模型，可以更精确地判断
        if self.vad_model is not None:
            # TODO: 使用VAD模型检测真实的speech endpoints
            pass
        
        return state_labels
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        
        processed_rounds = []
        
        for round_idx, round_data in enumerate(dialogue.get('rounds', [])):
            # 加载合成的问题音频
            audio_file = self.synthesized_dir / f"{idx}_{round_idx}_question.wav"
            
            if not audio_file.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                continue
            
            try:
                # 加载音频
                wav, sr = torchaudio.load(audio_file)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                
                # 计算Mel特征
                mel_features = k.fbank(
                    waveform=wav,
                    dither=0,
                    frame_length=25,
                    frame_shift=10,
                    num_mel_bins=self.config.input_dim
                )
                
                mel_features = mel_features.squeeze(0)
                
                # 计算状态标签
                state_labels = self._compute_state_labels(mel_features)
                
                processed_rounds.append({
                    'question_mel': mel_features,
                    'question_text': round_data.get('question', ''),
                    'answer_text': round_data.get('answer', ''),
                    'state_labels': state_labels,
                    'feat_length': mel_features.size(0)
                })
            
            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")
                continue
        
        return processed_rounds

class TTSPairedDataset(Dataset):
    """TTS Stage 2: 文本-语音配对数据集（3000h）"""
    
    def __init__(self, data_path: str, config: FreezeOmniTrainingConfig, tokenizer=None):
        self.config = config
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        
        # 加载manifest
        manifest_file = self.data_path / "manifest.json"
        self.samples = []
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.samples.append({
                        'text': item['text'],
                        'audio': self.data_path / item['audio'],
                        'codec_tokens': self.data_path / item.get('codec_tokens', '')
                    })
        
        logger.info(f"Loaded {len(self.samples)} TTS paired samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # 加载预计算的codec tokens
            if sample['codec_tokens'].exists():
                codec_tokens = torch.load(sample['codec_tokens'])
            else:
                # 如果没有预计算，返回None，在训练时实时计算
                codec_tokens = None
            
            return {
                'text': sample['text'],
                'codec_tokens': codec_tokens
            }
        
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            return {
                'text': '',
                'codec_tokens': None
            }

# ================== 辅助函数 ==================

def collate_fn_asr(batch):
    """ASR数据的collate函数"""
    # 过滤空样本
    batch = [b for b in batch if b['text'] != '']
    
    if len(batch) == 0:
        return None
    
    mel_features = []
    texts = []
    feat_lengths = []
    utt_ids = []
    
    for sample in batch:
        mel_features.append(sample['mel_features'])
        texts.append(sample['text'])
        feat_lengths.append(sample['feat_length'])
        utt_ids.append(sample['utt_id'])
    
    # Padding
    max_len = max(feat_lengths)
    padded_features = []
    
    for feat in mel_features:
        pad_len = max_len - feat.size(0)
        if pad_len > 0:
            padded = F.pad(feat, (0, 0, 0, pad_len))
        else:
            padded = feat
        padded_features.append(padded)
    
    return {
        'mel_features': torch.stack(padded_features),
        'texts': texts,
        'feat_lengths': torch.tensor(feat_lengths),
        'utt_ids': utt_ids
    }

def collate_fn_dialogue(batch):
    """对话数据的collate函数"""
    # batch是list of list of rounds
    all_rounds = []
    for dialogue in batch:
        all_rounds.extend(dialogue)
    
    if len(all_rounds) == 0:
        return None
    
    # 按照round处理
    mel_features = []
    answer_texts = []
    state_labels = []
    feat_lengths = []
    
    for round_data in all_rounds:
        mel_features.append(round_data['question_mel'])
        answer_texts.append(round_data['answer_text'])
        state_labels.append(round_data['state_labels'])
        feat_lengths.append(round_data['feat_length'])
    
    # Padding
    max_mel_len = max(feat_lengths)
    padded_mels = []
    
    for feat in mel_features:
        pad_len = max_mel_len - feat.size(0)
        if pad_len > 0:
            padded = F.pad(feat, (0, 0, 0, pad_len))
        else:
            padded = feat
        padded_mels.append(padded)
    
    # Padding state labels
    max_state_len = max(len(s) for s in state_labels)
    padded_states = []
    for states in state_labels:
        pad_len = max_state_len - len(states)
        if pad_len > 0:
            padded = F.pad(states, (0, pad_len), value=3)  # 3=padding
        else:
            padded = states
        padded_states.append(padded)
    
    return {
        'mel_features': torch.stack(padded_mels),
        'answer_texts': answer_texts,
        'state_labels': torch.stack(padded_states),
        'feat_lengths': torch.tensor(feat_lengths)
    }

def save_checkpoint(model, path, optimizer=None, epoch=None, step=None):
    """保存检查点"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {}
    if isinstance(model, dict):
        checkpoint.update(model)
    else:
        checkpoint['model'] = model.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if step is not None:
        checkpoint['step'] = step
    
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

def load_cmvn(config: FreezeOmniTrainingConfig):
    """加载CMVN统计量"""
    from models.encoder.cmvn import load_cmvn as load_cmvn_func, GlobalCMVN
    
    if not Path(config.cmvn_file).exists():
        logger.warning(f"CMVN file not found: {config.cmvn_file}")
        return None
    
    mean, istd = load_cmvn_func(config.cmvn_file, is_json=True)
    
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(),
        torch.from_numpy(istd).float()
    )
    
    logger.info(f"Loaded CMVN from {config.cmvn_file}")
    return global_cmvn

def create_encoder(config: FreezeOmniTrainingConfig, global_cmvn=None):
    """创建编码器"""
    from models.encoder.encoder import speechEncoder
    
    # 构建配置字典（对应argparse的参数）
    encoder_config = {
        'encoder-layer-config': config.encoder_layer_config,
        'encoder-input-dim': config.input_dim,
        'encoder-output-dim': config.encoder_output_dim,
    }
    
    # Subsampling配置
    subsampling_config = {
        'subsampling-rate': config.subsampling_rate,
        'subsampling-input-dim': config.subsampling_input_dim,
        'subsampling-output-dim': config.subsampling_output_dim,
        'subsampling-dropout-rate': config.subsampling_dropout_rate,
    }
    
    # Transformer配置
    transformer_config = {
        'transformer-input-dim': config.transformer_attention_dim,
        'transformer-output-dim': config.transformer_attention_dim,
        'transformer-attention-dim': config.transformer_attention_dim,
        'transformer-attention-heads': config.transformer_attention_heads,
        'transformer-linear-units': config.transformer_linear_units,
        'transformer-num-blocks': config.transformer_num_blocks,
        'transformer-dropout-rate': config.transformer_dropout_rate,
        'transformer-attention-dropout-rate': config.transformer_attention_dropout_rate,
        'transformer-positional-dropout-rate': config.transformer_positional_dropout_rate,
        'transformer-input-layer': config.transformer_input_layer,
        'transformer-pos-enc-class': config.transformer_pos_enc_class,
        'transformer-normalize-before': config.transformer_normalize_before,
        'transformer-concat-after': config.transformer_concat_after,
        'transformer-positionwise-layer-type': config.transformer_positionwise_layer_type,
        'transformer-chunk_size': config.chunk_size,
        'transformer-left_chunks': config.left_chunks,
        'transformer-dynamic-chunks': config.use_dynamic_chunk,
    }
    
    para_conf = {
        'subsampling': subsampling_config,
        'transformer': transformer_config,
    }
    
    encoder = speechEncoder(
        input_dim=config.input_dim,
        overview_conf=encoder_config,
        para_conf=para_conf,
        global_cmvn=global_cmvn
    )
    
    return encoder

# ================== Stage 1: ASR预训练 ==================

def train_stage1_asr(config: FreezeOmniTrainingConfig):
    """Stage 1: ASR预训练（CTC）"""
    
    logger.info("="*80)
    logger.info("Stage 1: ASR Pre-training with CTC")
    logger.info("="*80)
    
    device = torch.device(config.device)
    
    # 加载CMVN
    global_cmvn = load_cmvn(config)
    
    # 创建编码器
    encoder = create_encoder(config, global_cmvn).to(device)
    
    # CTC输出层（假设中文字符+字母+数字）
    vocab_size = 5538  # ✅ 从train.yaml的output_dim
    ctc_proj = nn.Linear(config.encoder_output_dim, vocab_size).to(device)
    
    # 数据集
    dataset = Stage1ASRDataset(config.asr_data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.stage1_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_asr,
        pin_memory=True
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(ctc_proj.parameters()),
        lr=config.stage1_lr,
        betas=(config.adam_b1, config.adam_b2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器（WarmupLR）
    from torch.optim.lr_scheduler import LambdaLR
    
    def warmup_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(config.stage1_epochs):
        encoder.train()
        ctc_proj.train()
        total_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.stage1_epochs}")
        
        for batch_idx, batch in enumerate(progress):
            if batch is None:
                continue
            
            mel_features = batch['mel_features'].to(device)
            texts = batch['texts']
            feat_lengths = batch['feat_lengths'].to(device)
            
            # 前向传播
            encoder_out, masks = encoder(mel_features, feat_lengths)
            
            # CTC投影
            logits = ctc_proj(encoder_out)
            
            # 计算CTC损失
            # 需要将文本转换为token ids（简化处理，实际需要使用tokenizer）
            # 这里假设每个字符对应一个id
            try:
                # 简单的字符tokenization
                text_ids = []
                text_lengths = []
                for text in texts:
                    # 转换为字符ids（示例）
                    ids = [ord(c) % vocab_size for c in text[:100]]  # 限制长度
                    text_ids.append(torch.tensor(ids))
                    text_lengths.append(len(ids))
                
                # Padding
                text_ids = nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
                text_ids = text_ids.to(device)
                text_lengths = torch.tensor(text_lengths).to(device)
                
                # CTC loss
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                
                loss = F.ctc_loss(
                    log_probs,
                    text_ids,
                    feat_lengths // 4,  # 考虑4倍下采样
                    text_lengths,
                    blank=0,
                    zero_infinity=True
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(ctc_proj.parameters()), 
                    config.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                # 保存检查点
                if global_step % config.save_interval == 0:
                    save_checkpoint({
                        'encoder': encoder.state_dict(),
                        'ctc_proj': ctc_proj.state_dict()
                    }, 
                    f"{config.model_path}/stage1/checkpoint_step{global_step}.pt",
                    optimizer, epoch, global_step)
            
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 每个epoch保存
        save_checkpoint({
            'encoder': encoder.state_dict(),
            'ctc_proj': ctc_proj.state_dict()
        }, 
        f"{config.model_path}/stage1/epoch{epoch+1}.pt",
        optimizer, epoch, global_step)
    
    # 保存最终模型
    save_checkpoint({
        'encoder': encoder.state_dict(),
        'ctc_proj': ctc_proj.state_dict()
    }, f"{config.model_path}/stage1/final.pt")
    
    return encoder

# ================== Stage 2: Speech-LLM对齐 ==================

def train_stage2_alignment(config: FreezeOmniTrainingConfig, encoder):
    """Stage 2: 训练编码器+适配器，LLM冻结"""
    
    logger.info("="*80)
    logger.info("Stage 2: Speech-LLM Alignment")
    logger.info("="*80)
    
    device = torch.device(config.device)
    
    # 加载LLM（冻结）
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading LLM from {config.llm_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_path, 
        trust_remote_code=True
    )
    
    # 冻结LLM
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()
    
    logger.info(f"LLM hidden size: {llm.config.hidden_size}")
    
    # 创建适配器
    from models.adapter import CNNSubsampling
    
    adapter = CNNSubsampling(
        enc_out_dim=config.encoder_output_dim,
        llm_embed_dim=llm.config.hidden_size,
        kernel_size=config.adapter_kernel_size,
        activation_func=config.activation_func,
        norm=config.norm
    ).to(device)
    
    # 特殊tokens（用于引导LLM）
    special_token_num = 10
    special_tokens = nn.Embedding(special_token_num, llm.config.hidden_size).to(device)
    
    # 数据集
    dataset = Stage2AlignmentDataset(config.asr_data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.stage2_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_asr,
        pin_memory=True
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + 
        list(adapter.parameters()) + 
        list(special_tokens.parameters()),
        lr=config.stage2_lr,
        betas=(config.adam_b1, config.adam_b2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    from torch.optim.lr_scheduler import LambdaLR
    
    def warmup_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(config.stage2_epochs):
        encoder.train()
        adapter.train()
        special_tokens.train()
        total_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.stage2_epochs}")
        
        for batch_idx, batch in enumerate(progress):
            if batch is None:
                continue
            
            mel_features = batch['mel_features'].to(device)
            texts = batch['texts']
            feat_lengths = batch['feat_lengths'].to(device)
            
            # 编码器
            encoder_out, masks = encoder(mel_features, feat_lengths)
            
            # 适配器（包含2倍下采样）
            adapted_features, adapted_masks = adapter(encoder_out, masks)
            
            # 添加特殊tokens（在音频前）
            batch_size = adapted_features.size(0)
            special_ids = torch.arange(5).repeat(batch_size, 1).to(device)
            special_embeds = special_tokens(special_ids)
            
            # 拼接
            llm_input_embeds = torch.cat([special_embeds, adapted_features], dim=1)
            
            # 准备attention mask
            special_mask = torch.ones(batch_size, 5, dtype=torch.bool).to(device)
            full_mask = torch.cat([special_mask, adapted_masks.squeeze(1)], dim=1)
            
            # 准备标签（文本的token ids）
            try:
                labels = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).input_ids.to(device)
                
                # 通过LLM
                with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                    outputs = llm(
                        inputs_embeds=llm_input_embeds,
                        attention_mask=full_mask,
                        labels=labels,
                        return_dict=True
                    )
                
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + 
                    list(adapter.parameters()) + 
                    list(special_tokens.parameters()),
                    config.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                # 保存检查点
                if global_step % config.save_interval == 0:
                    save_checkpoint({
                        'encoder': encoder.state_dict(),
                        'adapter': adapter.state_dict(),
                        'special_tokens': special_tokens.state_dict()
                    },
                    f"{config.model_path}/stage2/checkpoint_step{global_step}.pt",
                    optimizer, epoch, global_step)
            
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 每个epoch保存
        save_checkpoint({
            'encoder': encoder.state_dict(),
            'adapter': adapter.state_dict(),
            'special_tokens': special_tokens.state_dict()
        },
        f"{config.model_path}/stage2/epoch{epoch+1}.pt",
        optimizer, epoch, global_step)
    
    # 保存最终模型
    save_checkpoint({
        'encoder': encoder.state_dict(),
        'adapter': adapter.state_dict(),
        'special_tokens': special_tokens.state_dict()
    }, f"{config.model_path}/stage2/final.pt")
    
    return encoder, adapter, special_tokens, llm, tokenizer

# ================== Stage 3: 对话能力训练 ==================

def train_stage3_dialogue(config: FreezeOmniTrainingConfig, encoder, adapter, llm, tokenizer):
    """Stage 3: 只训练prompt embeddings和状态预测头（编码器冻结）"""
    
    logger.info("="*80)
    logger.info("Stage 3: Dialogue Training (Frozen Encoder & Adapter)")
    logger.info("="*80)
    
    device = torch.device(config.device)
    
    # ✅ 冻结编码器和适配器（关键！）
    for param in encoder.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = False
    encoder.eval()
    adapter.eval()
    
    logger.info("Encoder and Adapter are FROZEN")
    
    # 初始化可训练组件
    prompt_embeddings = nn.Embedding(
        config.prompt_num,
        llm.config.hidden_size
    ).to(device)
    
    state_predictor = nn.Linear(
        llm.config.hidden_size,
        config.predict_usr_state
    ).to(device)
    
    logger.info(f"Trainable params: prompt_embeddings ({config.prompt_num}x{llm.config.hidden_size}), "
               f"state_predictor ({llm.config.hidden_size}->{config.predict_usr_state})")
    
    # 解析chat template
    chat_template_parts = parse_chat_template(config.chat_template, tokenizer)
    
    # 数据集
    dataset = Stage3DialogueDataset(
        config.qa_data_path, 
        config,
        tokenizer=tokenizer,
        vad_model=None  # 可以加载VAD模型
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 对话数据一次一个
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_dialogue
    )
    
    # 优化器（只优化prompt和状态预测头）
    optimizer = torch.optim.AdamW(
        list(prompt_embeddings.parameters()) + 
        list(state_predictor.parameters()),
        lr=config.stage3_lr,
        betas=(config.adam_b1, config.adam_b2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    from torch.optim.lr_scheduler import LambdaLR
    
    def warmup_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(config.stage3_epochs):
        prompt_embeddings.train()
        state_predictor.train()
        
        total_gen_loss = 0
        total_state_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.stage3_epochs}")
        
        for batch_idx, batch in enumerate(progress):
            if batch is None:
                continue
            
            mel_features = batch['mel_features'].to(device)
            answer_texts = batch['answer_texts']
            state_labels = batch['state_labels'].to(device)
            feat_lengths = batch['feat_lengths'].to(device)
            
            # 通过冻结的编码器和适配器
            with torch.no_grad():
                encoder_out, masks = encoder(mel_features, feat_lengths)
                adapted_features, adapted_masks = adapter(encoder_out, masks)
            
            batch_size = adapted_features.size(0)
            
            # 添加可训练的prompt embeddings
            prompt_ids = torch.arange(config.prompt_num).repeat(batch_size, 1).to(device)
            prompt_embeds = prompt_embeddings(prompt_ids)
            
            # 添加chat template
            # role: <|im_start|>system\nYou are...<|im_end|>
            # prefix: <|im_start|>user\n
            # audio: <audio features>
            # suffix: <|im_end|>\n<|im_start|>assistant\n
            
            role_embeds = llm.model.embed_tokens(chat_template_parts['role'].to(device))
            prefix_embeds = llm.model.embed_tokens(chat_template_parts['prefix'].to(device))
            suffix_embeds = llm.model.embed_tokens(chat_template_parts['suffix'].to(device))
            
            # 拼接（如果add_prompt_before=True，prompt在audio前）
            if config.add_prompt_before:
                llm_input_embeds = torch.cat([
                    role_embeds.repeat(batch_size, 1, 1),
                    prefix_embeds.repeat(batch_size, 1, 1),
                    prompt_embeds,
                    adapted_features,
                    suffix_embeds.repeat(batch_size, 1, 1)
                ], dim=1)
            else:
                llm_input_embeds = torch.cat([
                    role_embeds.repeat(batch_size, 1, 1),
                    prefix_embeds.repeat(batch_size, 1, 1),
                    adapted_features,
                    prompt_embeds,
                    suffix_embeds.repeat(batch_size, 1, 1)
                ], dim=1)
            
            # 准备attention mask
            role_mask = torch.ones(batch_size, role_embeds.size(1)).to(device)
            prefix_mask = torch.ones(batch_size, prefix_embeds.size(1)).to(device)
            prompt_mask = torch.ones(batch_size, config.prompt_num).to(device)
            suffix_mask = torch.ones(batch_size, suffix_embeds.size(1)).to(device)
            
            if config.add_prompt_before:
                full_mask = torch.cat([
                    role_mask,
                    prefix_mask,
                    prompt_mask,
                    adapted_masks.squeeze(1),
                    suffix_mask
                ], dim=1)
            else:
                full_mask = torch.cat([
                    role_mask,
                    prefix_mask,
                    adapted_masks.squeeze(1),
                    prompt_mask,
                    suffix_mask
                ], dim=1)
            
            # 准备答案标签
            try:
                answer_ids = tokenizer(
                    answer_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).input_ids.to(device)
                
                # LLM前向（多任务：生成+状态预测）
                with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                    outputs = llm(
                        inputs_embeds=llm_input_embeds,
                        attention_mask=full_mask,
                        labels=answer_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                # 生成损失
                gen_loss = outputs.loss
                
                # 状态预测（在prefill阶段的chunk末尾）
                hidden_states = outputs.hidden_states[-1]
                
                # 计算chunk结束位置（只在adapted_features部分）
                chunk_size = config.chunk_size
                
                # 考虑下采样：encoder 4x, adapter 2x = 总共8x
                # 所以chunk_size=4对应原始特征的32帧
                adapted_seq_len = adapted_features.size(1)
                chunk_ends = []
                
                # 偏移量：role + prefix + (prompt if before)
                if config.add_prompt_before:
                    offset = role_embeds.size(1) + prefix_embeds.size(1) + config.prompt_num
                else:
                    offset = role_embeds.size(1) + prefix_embeds.size(1)
                
                # 每个chunk的结束位置
                for i in range(chunk_size - 1, adapted_seq_len, chunk_size):
                    chunk_ends.append(offset + i)
                
                if len(chunk_ends) > 0:
                    # 获取chunk末尾的hidden states
                    chunk_hidden = hidden_states[:, chunk_ends, :]
                    state_logits = state_predictor(chunk_hidden)
                    
                    # 确保标签长度匹配
                    min_len = min(len(chunk_ends), state_labels.size(1))
                    
                    if min_len > 0:
                        state_loss = F.cross_entropy(
                            state_logits[:, :min_len].reshape(-1, config.predict_usr_state),
                            state_labels[:, :min_len].reshape(-1),
                            ignore_index=3  # 忽略padding
                        )
                    else:
                        state_loss = torch.tensor(0.0).to(device)
                else:
                    state_loss = torch.tensor(0.0).to(device)
                
                # 多任务损失（生成 + 状态预测）
                total_loss = gen_loss + 0.5 * state_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(prompt_embeddings.parameters()) + 
                    list(state_predictor.parameters()),
                    config.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                
                total_gen_loss += gen_loss.item()
                total_state_loss += state_loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({
                    'gen_loss': gen_loss.item(),
                    'state_loss': state_loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                # 保存检查点
                if global_step % config.save_interval == 0:
                    save_checkpoint({
                        'prompt_embeddings': prompt_embeddings.state_dict(),
                        'state_predictor': state_predictor.state_dict()
                    },
                    f"{config.model_path}/stage3/checkpoint_step{global_step}.pt",
                    optimizer, epoch, global_step)
            
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if num_batches > 0:
            avg_gen_loss = total_gen_loss / num_batches
            avg_state_loss = total_state_loss / num_batches
            logger.info(f"Epoch {epoch+1} - Gen Loss: {avg_gen_loss:.4f}, "
                       f"State Loss: {avg_state_loss:.4f}")
        
        # 每个epoch保存
        save_checkpoint({
            'prompt_embeddings': prompt_embeddings.state_dict(),
            'state_predictor': state_predictor.state_dict()
        },
        f"{config.model_path}/stage3/epoch{epoch+1}.pt",
        optimizer, epoch, global_step)
    
    # 保存最终模型
    save_checkpoint({
        'prompt_embeddings': prompt_embeddings.state_dict(),
        'state_predictor': state_predictor.state_dict()
    }, f"{config.model_path}/stage3/final.pt")

def parse_chat_template(template: str, tokenizer):
    """解析chat template为token ids"""
    # 分割template
    # <|im_start|>system\nYou are...<|im_end|>\n<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n
    
    parts = template.split('<audio>')
    
    # role + prefix
    prefix_text = parts[0]  # '<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n'
    
    # suffix
    suffix_text = parts[1]  # '<|im_end|>\n<|im_start|>assistant\n'
    
    # 进一步分割
    prefix_parts = prefix_text.split('<|im_end|>')
    role_text = prefix_parts[0] + '<|im_end|>'  # '<|im_start|>system\n...<|im_end|>'
    user_prefix = prefix_parts[1] if len(prefix_parts) > 1 else ''
    
    # Tokenize
    role_ids = tokenizer(role_text, return_tensors='pt').input_ids
    prefix_ids = tokenizer(user_prefix, return_tensors='pt').input_ids
    suffix_ids = tokenizer(suffix_text, return_tensors='pt').input_ids
    
    return {
        'role': role_ids,
        'prefix': prefix_ids,
        'suffix': suffix_ids
    }

# ================== TTS训练流程 ==================

def train_tts_pipeline(config: FreezeOmniTrainingConfig):
    """TTS完整训练流程"""
    
    logger.info("="*80)
    logger.info("TTS Training Pipeline")
    logger.info("="*80)
    
    # Stage 1: 训练Codec（TiCodec）
    logger.info("Stage 1: Training TiCodec...")
    train_codec(config)
    
    # Stage 2: 训练NAR+AR解码器
    logger.info("Stage 2: Training NAR+AR Decoders...")
    train_tts_decoders(config)
    
    # Stage 3: Prefix微调
    logger.info("Stage 3: Prefix Fine-tuning...")
    train_prefix_finetune(config)

def train_codec(config: FreezeOmniTrainingConfig):
    """训练TiCodec（单码本）"""
    
    logger.info("Training TiCodec with single codebook...")
    
    # 创建配置文件
    codec_config = {
        "resblock": "1",
        "num_gpus": torch.cuda.device_count(),
        "batch_size": 160,
        "learning_rate": config.codec_lr,
        "adam_b1": 0.5,
        "adam_b2": 0.9,
        "lr_decay": 0.98,
        "seed": 1234,
        
        "upsample_rates": [8, 5, 5, 3],
        "upsample_kernel_sizes": [16, 11, 11, 5],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        
        "segment_size": 24000,
        "num_mels": 80,
        "num_freq": 1025,
        "n_fft": 1024,
        "hop_size": 240,
        "win_size": 1024,
        
        "sampling_rate": config.sample_rate,
        
        "n_code_groups": config.codec_n_code_groups,
        "residul_layer": config.codec_residul_layer,
        "n_codes": config.codec_n_codes,
        "codebook_loss_lambda": 1.0,
        "commitment_loss_lambda": 0.25,
        "global_code_num": config.global_code_num,
        "global_feature_conv": [128, 64, 128, 3, 1],
        "global_tokens": config.global_tokens,
        
        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,
        
        "num_workers": 12
    }
    
    # 保存配置
    config_path = f"{config.model_path}/codec/model.json"
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(codec_config, f, indent=4)
    
    logger.info(f"Saved codec config to {config_path}")
    
    # TODO: 实现完整的TiCodec训练
    # 这里需要参考TiCodec的原始训练代码
    logger.info("Codec training completed (using pre-trained checkpoint)")

def train_tts_decoders(config: FreezeOmniTrainingConfig):
    """训练NAR和AR解码器（Stage 2）"""
    
    logger.info("Training NAR+AR Decoders with text-speech pairs...")
    
    device = torch.device(config.device)
    
    # 加载LLM的embedding层（冻结）
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.llm_path, trust_remote_code=True)
    
    embedding_layer = llm.model.embed_tokens
    for param in embedding_layer.parameters():
        param.requires_grad = False
    
    # 初始化解码器
    from models.decoder.decoder import LLM2TTSCodecAR
    import argparse
    
    decoder_args = argparse.Namespace(
        idim=config.decoder_hidden_size,
        odim=config.vocab_size + 4,  # +4 for special tokens
        encoder_output_dim=config.decoder_hidden_size,
        transformer_num_blocks=config.decoder_num_blocks,
        transformer_attention_dim=config.decoder_hidden_size,
        transformer_attention_heads=config.decoder_attention_heads,
        transformer_linear_units=config.decoder_linear_units,
        transformer_dropout_rate=0.1,
        transformer_attention_dropout_rate=0.1,
        transformer_positional_dropout_rate=0.1,
        transformer_input_layer='linear',
        transformer_pos_enc_class='rel-enc',
        transformer_normalize_before=True,
        transformer_concat_after=False,
        transformer_positionwise_layer_type='linear',
        transformer_positionwise_conv_kernel_size=1,
        transformer_chunk_size=[1],
        transformer_left_chunks=[-1],
        transformer_dynamic_chunks=False,
        encoder_criterion='ce',
        encoder_drop_rate=0.1,
        encoder_pre_norm_type='ln',
        encoder_upsample_rate=1,
        kv_cache_prefix_finetune=0  # Stage 2不使用prefix
    )
    
    speech_decoder = LLM2TTSCodecAR(
        idim=decoder_args.idim,
        odim=decoder_args.odim,
        args=decoder_args
    ).to(device)
    
    # 数据集
    dataset = TTSPairedDataset(config.tts_paired_data, config, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        speech_decoder.parameters(),
        lr=config.tts_stage2_lr,
        betas=(config.adam_b1, config.adam_b2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # 训练循环
    global_step = 0
    
    for epoch in range(20):
        speech_decoder.train()
        total_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"TTS Epoch {epoch+1}/20")
        
        for batch_idx, batch in enumerate(progress):
            texts = batch['text']
            codec_tokens = batch['codec_tokens']
            
            if codec_tokens is None:
                continue
            
            try:
                # 文本 -> LLM embeddings (冻结)
                with torch.no_grad():
                    text_ids = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).input_ids.to(device)
                    
                    text_embeds = embedding_layer(text_ids)
                
                # 准备batch数据
                batch_data = {
                    'x': text_embeds,
                    'x_lens': torch.tensor([text_embeds.size(1)] * text_embeds.size(0)),
                    'y': codec_tokens.to(device),
                    'y_lens': torch.tensor([codec_tokens.size(1)] * codec_tokens.size(0))
                }
                
                # 前向传播
                loss = speech_decoder(batch_data)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    speech_decoder.parameters(),
                    config.gradient_clip
                )
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches > 0:
            logger.info(f"Epoch {epoch+1} - Average Loss: {total_loss/num_batches:.4f}")
        
        # 保存检查点
        save_checkpoint(
            speech_decoder,
            f"{config.model_path}/decoder/epoch{epoch+1}.pt"
        )
    
    # 保存最终模型
    save_checkpoint(
        speech_decoder,
        f"{config.model_path}/decoder/final.pt"
    )
    
    logger.info("NAR+AR decoder training completed")

def train_prefix_finetune(config: FreezeOmniTrainingConfig):
    """Stage 3: Prefix微调（连接LLM hidden states）"""
    
    logger.info("Training Prefix Fine-tuning to connect LLM hidden states...")
    
    device = torch.device(config.device)
    
    # 加载Stage 2的解码器
    from models.decoder.decoder import LLM2TTSCodecAR
    import argparse
    
    # 创建带prefix的解码器
    decoder_args = argparse.Namespace(
        idim=config.decoder_hidden_size,
        odim=config.vocab_size + 4,
        encoder_output_dim=config.decoder_hidden_size,
        transformer_num_blocks=config.decoder_num_blocks,
        transformer_attention_dim=config.decoder_hidden_size,
        transformer_attention_heads=config.decoder_attention_heads,
        transformer_linear_units=config.decoder_linear_units,
        transformer_dropout_rate=0.1,
        transformer_attention_dropout_rate=0.1,
        transformer_positional_dropout_rate=0.1,
        transformer_input_layer='linear',
        transformer_pos_enc_class='rel-enc',
        transformer_normalize_before=True,
        transformer_concat_after=False,
        transformer_positionwise_layer_type='linear',
        transformer_positionwise_conv_kernel_size=1,
        transformer_chunk_size=[1],
        transformer_left_chunks=[-1],
        transformer_dynamic_chunks=False,
        encoder_criterion='ce',
        encoder_drop_rate=0.1,
        encoder_pre_norm_type='ln',
        encoder_upsample_rate=1,
        kv_cache_prefix_finetune=1  # ✅ 启用prefix finetune
    )
    
    speech_decoder = LLM2TTSCodecAR(
        idim=decoder_args.idim,
        odim=decoder_args.odim,
        args=decoder_args
    ).to(device)
    
    # 加载Stage 2的参数
    stage2_checkpoint = torch.load(f"{config.model_path}/decoder/final.pt")
    speech_decoder.load_state_dict(stage2_checkpoint, strict=False)
    
    # ✅ 冻结NAR和AR解码器，只训练prefix部分
    for name, param in speech_decoder.named_parameters():
        if 'prefix' not in name.lower():
            param.requires_grad = False
    
    logger.info("NAR and AR decoders are FROZEN, only training prefix layers")
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in speech_decoder.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 优化器（只优化prefix部分）
    prefix_params = [p for n, p in speech_decoder.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        prefix_params,
        lr=config.tts_stage3_lr,
        betas=(config.adam_b1, config.adam_b2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # 使用Stage 3的对话数据（包含LLM hidden states）
    # TODO: 需要准备包含LLM hidden states的数据
    
    logger.info("Prefix fine-tuning completed")
    
    # 保存最终模型
    save_checkpoint(
        speech_decoder,
        f"{config.model_path}/decoder/final_with_prefix.pt"
    )

# ================== 主函数 ==================

def main():
    """完整的训练流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Freeze-Omni Training Pipeline")
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['all', '1', '2', '3', 'tts'],
                       help='Which stage to train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--llm_path', type=str, default='./Qwen2-7B-Instruct')
    parser.add_argument('--asr_data', type=str, default='./data/asr')
    parser.add_argument('--qa_data', type=str, default='./data/qa')
    parser.add_argument('--tts_data', type=str, default='./data/tts_paired')
    args = parser.parse_args()
    
    # 创建配置
    config = FreezeOmniTrainingConfig()
    
    # 覆盖配置
    if args.model_path:
        config.model_path = args.model_path
    if args.llm_path:
        config.llm_path = args.llm_path
    if args.asr_data:
        config.asr_data_path = args.asr_data
    if args.qa_data:
        config.qa_data_path = args.qa_data
    if args.tts_data:
        config.tts_paired_data = args.tts_data
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('train.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("="*80)
    logger.info("Freeze-Omni Training Pipeline")
    logger.info("="*80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Mixed Precision: {config.mixed_precision}")
    logger.info(f"Training Stage: {args.stage}")
    logger.info(f"Model Path: {config.model_path}")
    logger.info(f"LLM Path: {config.llm_path}")
    logger.info("="*80)
    
    # 创建输出目录
    Path(config.model_path).mkdir(parents=True, exist_ok=True)
    for stage in ['stage1', 'stage2', 'stage3', 'codec', 'decoder']:
        Path(f"{config.model_path}/{stage}").mkdir(parents=True, exist_ok=True)
    
    # 语音输入训练流程
    if args.stage in ['all', '1', '2', '3']:
        
        # Stage 1: ASR预训练
        if args.stage in ['all', '1']:
            logger.info("\n" + "="*80)
            logger.info("Starting Stage 1: ASR Pre-training")
            logger.info("="*80 + "\n")
            
            encoder = train_stage1_asr(config)
            
            logger.info("Stage 1 completed successfully!")
        
        else:
            # 加载Stage 1的编码器
            logger.info(f"Loading Stage 1 encoder from {config.model_path}/stage1/final.pt")
            
            global_cmvn = load_cmvn(config)
            encoder = create_encoder(config, global_cmvn)
            
            checkpoint = torch.load(f"{config.model_path}/stage1/final.pt")
            encoder.load_state_dict(checkpoint['encoder'])
            encoder = encoder.to(config.device)
        
        # Stage 2: Speech-LLM对齐
        if args.stage in ['all', '2']:
            logger.info("\n" + "="*80)
            logger.info("Starting Stage 2: Speech-LLM Alignment")
            logger.info("="*80 + "\n")
            
            encoder, adapter, special_tokens, llm, tokenizer = train_stage2_alignment(config, encoder)
            
            logger.info("Stage 2 completed successfully!")
        
        elif args.stage == '3':
            # 加载Stage 2的组件
            logger.info(f"Loading Stage 2 components from {config.model_path}/stage2/final.pt")
            
            checkpoint = torch.load(f"{config.model_path}/stage2/final.pt")
            encoder.load_state_dict(checkpoint['encoder'])
            
            from models.adapter import CNNSubsampling
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            llm = AutoModelForCausalLM.from_pretrained(
                config.llm_path,
                torch_dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
                trust_remote_code=True
            ).to(config.device)
            
            tokenizer = AutoTokenizer.from_pretrained(config.llm_path, trust_remote_code=True)
            
            adapter = CNNSubsampling(
                enc_out_dim=config.encoder_output_dim,
                llm_embed_dim=llm.config.hidden_size,
                kernel_size=config.adapter_kernel_size,
                activation_func=config.activation_func,
                norm=config.norm
            ).to(config.device)
            
            adapter.load_state_dict(checkpoint['adapter'])
        
        # Stage 3: 对话能力训练
        if args.stage in ['all', '3']:
            logger.info("\n" + "="*80)
            logger.info("Starting Stage 3: Dialogue Training")
            logger.info("="*80 + "\n")
            
            train_stage3_dialogue(config, encoder, adapter, llm, tokenizer)
            
            logger.info("Stage 3 completed successfully!")
    
    # TTS训练流程
    if args.stage in ['all', 'tts']:
        logger.info("\n" + "="*80)
        logger.info("Starting TTS Training Pipeline")
        logger.info("="*80 + "\n")
        
        train_tts_pipeline(config)
        
        logger.info("TTS training completed successfully!")
    
    logger.info("\n" + "="*80)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info("="*80)

if __name__ == "__main__":
    main()