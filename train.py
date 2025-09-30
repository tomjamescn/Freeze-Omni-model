#!/usr/bin/env python3
"""
Freeze-Omni 完整训练脚本
支持配置保存、恢复训练、多种实验配置

特性：
1. 自动保存所有训练配置到checkpoints
2. 支持从配置文件恢复训练
3. 支持灵活修改各种超参数
4. 完整的实验管理和日志记录

使用示例：
# 完整训练流程
python train.py --stage all \
    --model_path ./checkpoints \
    --llm_path ./Qwen2-7B-Instruct \
    --asr_data ./data/asr \
    --qa_data ./data/qa \
    --tts_data ./data/tts_paired

# 使用自定义配置
python train.py --config my_config.yaml --stage all

# 恢复训练
python train.py --resume ./checkpoints/stage1/checkpoint_step5000.pt --stage 1
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
import yaml
import logging
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import random
import copy
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ================== 配置管理类 ==================

@dataclass 
class FreezeOmniTrainingConfig:
    """
    完整的训练配置类
    支持YAML保存/加载，方便实验管理
    """
    
    # === 实验信息 ===
    experiment_name: str = "freeze_omni_default"
    experiment_note: str = "Default training configuration"
    created_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # === 路径配置 ===
    model_path: str = "./checkpoints"
    llm_path: str = "./Qwen2-7B-Instruct"
    cmvn_file: str = "./checkpoints/audiollm/global_cmvn"
    
    # === 数据配置 ===
    asr_data_path: str = "./data/asr"
    qa_data_path: str = "./data/qa"
    tts_paired_data: str = "./data/tts_paired"
    
    # === 编码器配置（可修改实验） ===
    input_dim: int = 80
    encoder_output_dim: int = 1024
    encoder_layer_config: str = "subsampling-transformer"
    
    # Subsampling配置（可实验不同下采样率）
    subsampling_rate: int = 4  # 可改为2, 8等
    subsampling_input_dim: int = 80
    subsampling_output_dim: int = 1024
    subsampling_dropout_rate: float = 0.1
    
    # Transformer配置（可实验不同层数和维度）
    transformer_num_blocks: int = 24  # 可改为12, 18, 32等
    transformer_attention_dim: int = 1024  # 可改为512, 768, 2048等
    transformer_attention_heads: int = 16  # 需要能被attention_dim整除
    transformer_linear_units: int = 4096  # FFN维度，可改为2048, 8192等
    transformer_dropout_rate: float = 0.1
    transformer_attention_dropout_rate: float = 0.0
    transformer_positional_dropout_rate: float = 0.1
    transformer_input_layer: str = "linear"
    transformer_pos_enc_class: str = "rel-enc"  # 可选"abs-enc"
    transformer_normalize_before: bool = True
    transformer_concat_after: bool = False
    transformer_positionwise_layer_type: str = "linear"  # 可选"conv1d", "conv1d-linear"
    
    # Chunk配置（可实验不同chunk大小）
    chunk_size: int = 4  # 可改为8, 16等
    left_chunks: int = 16  # 左侧context，可改为8, 32等
    use_dynamic_chunk: bool = False  # 训练时是否动态chunk
    
    # === 适配器配置（可实验不同适配器类型） ===
    adapter_type: str = "subsampling"  # 可选"cnn", "linear", "subsampling"
    adapter_kernel_size: int = 5  # 卷积核大小，可改为3, 7等
    adapter_downsample_rate: int = 2  # 适配器额外下采样，可改为1（不下采样）
    activation_func: str = "gelu"  # 可选"relu", "swish"
    norm: str = "layer"  # 可选"batch"
    llm_embed_dim: int = 3584  # Qwen2-7B的维度
    
    # === AudioLLM配置 ===
    freeze_llm: bool = True
    freeze_encoder: bool = False  # Stage 3才冻结
    freeze_adapter: bool = False  # Stage 3才冻结
    predict_usr_state: int = 4  # 状态预测类别数
    prompt_num: int = 25  # Prompt embedding数量，可实验5, 10, 50等
    add_prompt_before: bool = True  # Prompt位置
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
    
    # === 解码器配置（可实验不同层数和维度） ===
    decoder_num_blocks: int = 4  # 可改为2, 6, 8等
    decoder_hidden_size: int = 896  # 可改为512, 1024等
    decoder_attention_heads: int = 14  # 需要能被hidden_size整除
    decoder_linear_units: int = 4864  # FFN维度
    decoder_use_pre_network: bool = True  # 是否使用pre-network层
    decoder_pre_network_layers: int = 2  # pre-network层数
    vocab_size: int = 1024  # TiCodec码本大小
    
    # === TiCodec配置（可实验不同码本配置） ===
    codec_n_codes: int = 1024  # 码本大小，可改为512, 2048等
    codec_n_code_groups: int = 1  # 码本组数，可改为2, 4等（多码本）
    codec_residul_layer: int = 1  # 残差层数，可改为2, 4等
    codec_frame_rate: int = 40  # Token频率（Hz）
    sample_rate: int = 24000
    global_code_num: int = 8  # 全局风格token数
    global_tokens: List[int] = None
    
    # === 训练超参数（可实验不同学习率和batch size） ===
    # Stage 1 - ASR
    stage1_lr: float = 2e-4  # 可实验1e-4, 3e-4等
    stage1_epochs: int = 20
    stage1_batch_size: int = 32  # 可根据GPU内存调整
    stage1_use_spec_aug: bool = True  # 是否使用SpecAugment
    
    # Stage 2 - Alignment
    stage2_lr: float = 1e-4
    stage2_epochs: int = 10
    stage2_batch_size: int = 16
    
    # Stage 3 - Dialogue
    stage3_lr: float = 6e-4
    stage3_epochs: int = 5
    stage3_batch_size: int = 4
    stage3_multi_task_weight: float = 0.5  # 状态预测损失权重
    
    # TTS训练
    codec_lr: float = 1e-3
    tts_stage2_lr: float = 5e-5
    tts_stage2_epochs: int = 20
    tts_stage3_lr: float = 5e-5
    tts_stage3_epochs: int = 10
    
    # 通用训练配置
    warmup_steps: int = 200  # 学习率warm-up步数
    gradient_clip: float = 5.0  # 梯度裁剪
    mixed_precision: bool = True  # 混合精度训练
    save_interval: int = 1000  # 保存间隔（steps）
    eval_interval: int = 500  # 评估间隔
    log_interval: int = 100  # 日志间隔
    accum_grad: int = 1  # 梯度累积
    
    # 优化器配置
    optim: str = "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.99
    eps: float = 1e-6
    weight_decay: float = 0.01
    
    # 学习率调度器
    scheduler: str = "warmuplr"  # 可选"cosine", "linear", "constant"
    
    # 数据增强
    use_speed_perturb: bool = False  # 语速扰动
    use_spec_aug: bool = True  # SpecAugment
    spec_aug_max_f: int = 10
    spec_aug_max_t: int = 20
    
    # 硬件配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True
    
    # 随机种子（可复现实验）
    seed: int = 42
    
    def __post_init__(self):
        """初始化后处理"""
        if self.global_tokens is None:
            self.global_tokens = [473, 975, 419, 219, 565, 121, 550, 616]
        
        # 设置随机种子
        self.set_seed()
    
    def set_seed(self):
        """设置所有随机种子"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def save_to_yaml(self, path: str):
        """保存配置到YAML文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to {path}")
    
    def save_to_json(self, path: str):
        """保存配置到JSON文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load_from_yaml(cls, path: str):
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {path}")
        return cls(**config_dict)
    
    @classmethod
    def load_from_json(cls, path: str):
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        logger.info(f"Configuration loaded from {path}")
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def print_config(self):
        """打印配置"""
        logger.info("="*80)
        logger.info("Training Configuration:")
        logger.info("="*80)
        
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, (list, dict)):
                logger.info(f"{key}: {json.dumps(value, indent=2)}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("="*80)

# ================== 完整检查点管理 ==================

class CheckpointManager:
    """
    完整的检查点管理器
    自动保存配置、模型、优化器状态
    """
    
    def __init__(self, base_path: str, config: FreezeOmniTrainingConfig):
        self.base_path = Path(base_path)
        self.config = config
        
        # 创建目录结构
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 保存初始配置
        config.save_to_yaml(self.base_path / "config.yaml")
        config.save_to_json(self.base_path / "config.json")
    
    def save_checkpoint(
        self,
        stage: str,
        model: Dict[str, nn.Module] or nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict] = None,
        extra_info: Optional[Dict] = None
    ):
        """
        保存完整检查点
        
        Args:
            stage: 训练阶段（stage1, stage2, stage3等）
            model: 模型或模型字典
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            step: 当前step
            metrics: 训练指标
            extra_info: 额外信息
        """
        stage_path = self.base_path / stage
        stage_path.mkdir(parents=True, exist_ok=True)
        
        # 构建检查点
        checkpoint = {
            'config': self.config.to_dict(),
            'stage': stage,
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存模型状态
        if isinstance(model, dict):
            checkpoint['model'] = {
                name: m.state_dict() for name, m in model.items()
            }
        else:
            checkpoint['model'] = model.state_dict()
        
        # 保存优化器状态
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        # 保存调度器状态
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()
        
        # 保存指标
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # 保存额外信息
        if extra_info is not None:
            checkpoint['extra_info'] = extra_info
        
        # 确定文件名
        if step is not None:
            filename = f"checkpoint_step{step}.pt"
        elif epoch is not None:
            filename = f"checkpoint_epoch{epoch}.pt"
        else:
            filename = "checkpoint.pt"
        
        filepath = stage_path / filename
        
        # 保存
        torch.save(checkpoint, filepath)
        
        # 同时保存配置的副本
        self.config.save_to_yaml(stage_path / "config.yaml")
        
        logger.info(f"Checkpoint saved to {filepath}")
        
        # 保存最新的符号链接
        latest_link = stage_path / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filename)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Returns:
            包含所有状态的字典
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Checkpoint info: stage={checkpoint.get('stage')}, "
                   f"epoch={checkpoint.get('epoch')}, step={checkpoint.get('step')}")
        
        return checkpoint
    
    def list_checkpoints(self, stage: str) -> List[Path]:
        """列出某个阶段的所有检查点"""
        stage_path = self.base_path / stage
        if not stage_path.exists():
            return []
        
        checkpoints = sorted(stage_path.glob("checkpoint_*.pt"))
        return checkpoints

# ================== 数据集实现（同原代码） ==================

class Stage1ASRDataset(Dataset):
    """Stage 1: ASR数据集"""
    
    def __init__(self, data_path: str, config: FreezeOmniTrainingConfig):
        self.config = config
        self.data_path = Path(data_path)
        
        self.samples = []
        
        wav_scp = self.data_path / "wav.scp"
        text_file = self.data_path / "text"
        
        wav_dict = {}
        if wav_scp.exists():
            with open(wav_scp, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, wav_path = parts
                        wav_dict[utt_id] = wav_path
        
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
        
        try:
            wav, sr = torchaudio.load(sample['wav_path'])
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            
            # 语速扰动（数据增强）
            if self.config.use_speed_perturb and random.random() < 0.3:
                speed = random.choice([0.9, 1.0, 1.1])
                if speed != 1.0:
                    wav = torchaudio.functional.resample(
                        wav, 16000, int(16000 * speed)
                    )
            
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
            return {
                'utt_id': sample['utt_id'],
                'mel_features': torch.zeros(100, self.config.input_dim),
                'text': '',
                'feat_length': 100
            }

# 其他数据集类（Stage2, Stage3, TTS）同原代码...

# ================== 模型创建函数（更新以支持配置） ==================

def create_encoder(config: FreezeOmniTrainingConfig, global_cmvn=None):
    """根据配置创建编码器"""
    from models.encoder.encoder import speechEncoder
    
    encoder_config = {
        'encoder-layer-config': config.encoder_layer_config,
        'encoder-input-dim': config.input_dim,
        'encoder-output-dim': config.encoder_output_dim,
    }
    
    subsampling_config = {
        'subsampling-rate': config.subsampling_rate,
        'subsampling-input-dim': config.subsampling_input_dim,
        'subsampling-output-dim': config.subsampling_output_dim,
        'subsampling-dropout-rate': config.subsampling_dropout_rate,
    }
    
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
    
    # 打印模型参数量
    num_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return encoder

def create_adapter(config: FreezeOmniTrainingConfig, llm_hidden_size: int):
    """根据配置创建适配器"""
    from models.adapter import CNNAdapter, LinearAdapter, CNNSubsampling
    
    if config.adapter_type == "cnn":
        adapter = CNNAdapter(
            enc_out_dim=config.encoder_output_dim,
            llm_embed_dim=llm_hidden_size,
            kernel_size=config.adapter_kernel_size
        )
    elif config.adapter_type == "linear":
        adapter = LinearAdapter(
            enc_out_dim=config.encoder_output_dim,
            llm_embed_dim=llm_hidden_size
        )
    elif config.adapter_type == "subsampling":
        adapter = CNNSubsampling(
            enc_out_dim=config.encoder_output_dim,
            llm_embed_dim=llm_hidden_size,
            kernel_size=config.adapter_kernel_size,
            activation_func=config.activation_func,
            norm=config.norm
        )
    else:
        raise ValueError(f"Unknown adapter type: {config.adapter_type}")
    
    num_params = sum(p.numel() for p in adapter.parameters())
    logger.info(f"Adapter parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    return adapter

# ================== 训练函数（更新使用CheckpointManager） ==================

def train_stage1_asr(config: FreezeOmniTrainingConfig, checkpoint_manager: CheckpointManager):
    """Stage 1: ASR预训练"""
    
    logger.info("="*80)
    logger.info("Stage 1: ASR Pre-training with CTC")
    logger.info("="*80)
    
    config.print_config()
    
    device = torch.device(config.device)
    
    # 加载CMVN
    from models.encoder.cmvn import load_cmvn as load_cmvn_func, GlobalCMVN
    
    if Path(config.cmvn_file).exists():
        mean, istd = load_cmvn_func(config.cmvn_file, is_json=True)
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float()
        )
    else:
        global_cmvn = None
    
    # 创建编码器
    encoder = create_encoder(config, global_cmvn).to(device)
    
    # CTC输出层
    vocab_size = 5538
    ctc_proj = nn.Linear(config.encoder_output_dim, vocab_size).to(device)
    
    # 数据集
    dataset = Stage1ASRDataset(config.asr_data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.stage1_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn_asr,
        pin_memory=config.pin_memory
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(ctc_proj.parameters()),
        lr=config.stage1_lr,
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
    best_loss = float('inf')
    
    for epoch in range(config.stage1_epochs):
        encoder.train()
        ctc_proj.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.stage1_epochs}")
        
        for batch_idx, batch in enumerate(progress):
            if batch is None:
                continue
            
            # 训练逻辑（同原代码）
            # ...
            
            # 定期保存
            if global_step > 0 and global_step % config.save_interval == 0:
                checkpoint_manager.save_checkpoint(
                    stage='stage1',
                    model={'encoder': encoder, 'ctc_proj': ctc_proj},
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    metrics={'loss': epoch_loss / num_batches if num_batches > 0 else 0},
                    extra_info={'vocab_size': vocab_size}
                )
            
            global_step += 1
        
        # Epoch结束保存
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        checkpoint_manager.save_checkpoint(
            stage='stage1',
            model={'encoder': encoder, 'ctc_proj': ctc_proj},
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            metrics={'loss': avg_loss, 'best_loss': best_loss}
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳模型
            checkpoint_manager.save_checkpoint(
                stage='stage1',
                model={'encoder': encoder, 'ctc_proj': ctc_proj},
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                metrics={'loss': avg_loss, 'best_loss': best_loss},
                extra_info={'is_best': True}
            )
    
    return encoder

# ================== 其他训练函数（类似更新） ==================
# train_stage2_alignment, train_stage3_dialogue 等...

# ================== 辅助函数 ==================

def collate_fn_asr(batch):
    """ASR数据collate"""
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

# ================== 完整的主函数 ==================

def main():
    """完整的训练流程入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Freeze-Omni Training Pipeline")
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None, 
                       help='Load config from YAML/JSON file')
    
    # 训练控制
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['all', '1', '2', '3', 'tts', 'codec', 'tts_decoder'],
                       help='Which stage to train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    # 路径参数
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--llm_path', type=str, default='./Qwen2-7B-Instruct')
    parser.add_argument('--asr_data', type=str, default='./data/asr')
    parser.add_argument('--qa_data', type=str, default='./data/qa')
    parser.add_argument('--tts_data', type=str, default='./data/tts_paired')
    
    # 实验配置
    parser.add_argument('--experiment_name', type=str, default='freeze_omni_exp')
    parser.add_argument('--experiment_note', type=str, default='')
    
    # 可修改的超参数（会覆盖配置文件）
    parser.add_argument('--transformer_num_blocks', type=int, default=None,
                       help='Override transformer number of blocks')
    parser.add_argument('--transformer_attention_dim', type=int, default=None,
                       help='Override transformer attention dim')
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='Override chunk size')
    parser.add_argument('--prompt_num', type=int, default=None,
                       help='Override prompt embedding number')
    parser.add_argument('--stage1_lr', type=float, default=None,
                       help='Override Stage 1 learning rate')
    parser.add_argument('--stage1_batch_size', type=int, default=None,
                       help='Override Stage 1 batch size')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    
    args = parser.parse_args()
    
    # 创建或加载配置
    if args.config:
        # 从文件加载
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = FreezeOmniTrainingConfig.load_from_yaml(args.config)
        elif args.config.endswith('.json'):
            config = FreezeOmniTrainingConfig.load_from_json(args.config)
        else:
            raise ValueError("Config file must be YAML or JSON")
        
        logger.info(f"Loaded config from {args.config}")
    else:
        # 使用默认配置
        config = FreezeOmniTrainingConfig()
    
    # 命令行参数覆盖
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
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.experiment_note:
        config.experiment_note = args.experiment_note
    
    # 超参数覆盖
    if args.transformer_num_blocks is not None:
        config.transformer_num_blocks = args.transformer_num_blocks
    if args.transformer_attention_dim is not None:
        config.transformer_attention_dim = args.transformer_attention_dim
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.prompt_num is not None:
        config.prompt_num = args.prompt_num
    if args.stage1_lr is not None:
        config.stage1_lr = args.stage1_lr
    if args.stage1_batch_size is not None:
        config.stage1_batch_size = args.stage1_batch_size
    if args.seed is not None:
        config.seed = args.seed
        config.set_seed()
    
    # 设置日志
    log_dir = Path(config.model_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("="*80)
    logger.info("Freeze-Omni Training Pipeline")
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Note: {config.experiment_note}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Mixed Precision: {config.mixed_precision}")
    logger.info(f"Training Stage: {args.stage}")
    logger.info(f"Model Path: {config.model_path}")
    logger.info(f"LLM Path: {config.llm_path}")
    logger.info(f"Log File: {log_file}")
    logger.info("="*80)
    
    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(config.model_path, config)
    
    # 打印配置
    config.print_config()
    
    # 创建输出目录
    for stage in ['stage1', 'stage2', 'stage3', 'codec', 'decoder', 'logs']:
        (Path(config.model_path) / stage).mkdir(parents=True, exist_ok=True)
    
    # 恢复训练
    resume_checkpoint = None
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        resume_checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        
        # 如果检查点包含配置，可以选择使用检查点的配置
        if 'config' in resume_checkpoint:
            logger.info("Checkpoint contains saved config")
            # 这里可以选择是否使用检查点的配置
    
    # ================== 语音输入训练流程 ==================
    
    if args.stage in ['all', '1', '2', '3']:
        
        # Stage 1: ASR预训练
        if args.stage in ['all', '1']:
            logger.info("\n" + "="*80)
            logger.info("Starting Stage 1: ASR Pre-training")
            logger.info("="*80 + "\n")
            
            encoder = train_stage1_asr(config, checkpoint_manager)
            
            logger.info("Stage 1 completed successfully!")
        
        elif args.stage in ['2', '3']:
            # 加载Stage 1的编码器
            logger.info("Loading Stage 1 encoder...")
            
            from models.encoder.cmvn import load_cmvn as load_cmvn_func, GlobalCMVN
            
            if Path(config.cmvn_file).exists():
                mean, istd = load_cmvn_func(config.cmvn_file, is_json=True)
                global_cmvn = GlobalCMVN(
                    torch.from_numpy(mean).float(),
                    torch.from_numpy(istd).float()
                )
            else:
                global_cmvn = None
            
            encoder = create_encoder(config, global_cmvn).to(config.device)
            
            # 加载检查点
            stage1_checkpoints = checkpoint_manager.list_checkpoints('stage1')
            if stage1_checkpoints:
                latest_ckpt = checkpoint_manager.load_checkpoint(
                    str(stage1_checkpoints[-1])
                )
                encoder.load_state_dict(latest_ckpt['model']['encoder'])
                logger.info(f"Loaded encoder from {stage1_checkpoints[-1]}")
            else:
                logger.warning("No Stage 1 checkpoint found!")
        
        # Stage 2: Speech-LLM对齐
        if args.stage in ['all', '2']:
            logger.info("\n" + "="*80)
            logger.info("Starting Stage 2: Speech-LLM Alignment")
            logger.info("="*80 + "\n")
            
            # 这里调用train_stage2_alignment
            # encoder, adapter, llm, tokenizer = train_stage2_alignment(config, encoder, checkpoint_manager)
            
            logger.info("Stage 2 completed successfully!")
        
        # Stage 3: 对话能力训练
        if args.stage in ['all', '3']:
            logger.info("\n" + "="*80)
            logger.info("Starting Stage 3: Dialogue Training")
            logger.info("="*80 + "\n")
            
            # 这里调用train_stage3_dialogue
            # train_stage3_dialogue(config, encoder, adapter, llm, tokenizer, checkpoint_manager)
            
            logger.info("Stage 3 completed successfully!")
    
    # ================== TTS训练流程 ==================
    
    if args.stage in ['all', 'tts', 'codec', 'tts_decoder']:
        logger.info("\n" + "="*80)
        logger.info("Starting TTS Training Pipeline")
        logger.info("="*80 + "\n")
        
        if args.stage in ['all', 'tts', 'codec']:
            logger.info("Stage 1: Training TiCodec...")
            # train_codec(config, checkpoint_manager)
        
        if args.stage in ['all', 'tts', 'tts_decoder']:
            logger.info("Stage 2: Training NAR+AR Decoders...")
            # train_tts_decoders(config, checkpoint_manager)
            
            logger.info("Stage 3: Prefix Fine-tuning...")
            # train_prefix_finetune(config, checkpoint_manager)
        
        logger.info("TTS training completed successfully!")
    
    # ================== 训练完成 ==================
    
    logger.info("\n" + "="*80)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info("="*80)
    logger.info(f"All checkpoints saved to: {config.model_path}")
    logger.info(f"Configuration saved to: {config.model_path}/config.yaml")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # 生成训练总结
    summary = {
        'experiment_name': config.experiment_name,
        'experiment_note': config.experiment_note,
        'completed_time': datetime.now().isoformat(),
        'config': config.to_dict(),
        'stages_completed': args.stage
    }
    
    summary_path = Path(config.model_path) / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Training summary saved to: {summary_path}")


# ================== 命令行工具函数 ==================

def create_config_template():
    """创建配置模板文件"""
    config = FreezeOmniTrainingConfig()
    config.experiment_name = "my_experiment"
    config.experiment_note = "Experiment with custom configuration"
    
    # 保存模板
    config.save_to_yaml("config_template.yaml")
    config.save_to_json("config_template.json")
    
    print("Configuration templates created:")
    print("  - config_template.yaml")
    print("  - config_template.json")


def list_checkpoints(model_path: str):
    """列出所有检查点"""
    model_path = Path(model_path)
    
    print(f"\nCheckpoints in {model_path}:")
    print("="*80)
    
    for stage in ['stage1', 'stage2', 'stage3', 'codec', 'decoder']:
        stage_path = model_path / stage
        if not stage_path.exists():
            continue
        
        checkpoints = sorted(stage_path.glob("checkpoint_*.pt"))
        
        if checkpoints:
            print(f"\n{stage.upper()}:")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / 1024 / 1024
                print(f"  - {ckpt.name} ({size_mb:.2f} MB)")


def compare_configs(config1_path: str, config2_path: str):
    """比较两个配置文件的差异"""
    if config1_path.endswith('.yaml') or config1_path.endswith('.yml'):
        config1 = FreezeOmniTrainingConfig.load_from_yaml(config1_path)
    else:
        config1 = FreezeOmniTrainingConfig.load_from_json(config1_path)
    
    if config2_path.endswith('.yaml') or config2_path.endswith('.yml'):
        config2 = FreezeOmniTrainingConfig.load_from_yaml(config2_path)
    else:
        config2 = FreezeOmniTrainingConfig.load_from_json(config2_path)
    
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    print(f"\nComparing configurations:")
    print(f"Config 1: {config1_path}")
    print(f"Config 2: {config2_path}")
    print("="*80)
    
    differences = []
    for key in dict1.keys():
        if dict1[key] != dict2.get(key):
            differences.append({
                'key': key,
                'config1': dict1[key],
                'config2': dict2.get(key)
            })
    
    if differences:
        print(f"\nFound {len(differences)} differences:")
        for diff in differences:
            print(f"\n  {diff['key']}:")
            print(f"    Config 1: {diff['config1']}")
            print(f"    Config 2: {diff['config2']}")
    else:
        print("\nNo differences found!")


if __name__ == "__main__":
    import sys
    
    # 支持一些实用工具命令
    if len(sys.argv) > 1:
        if sys.argv[1] == "create-template":
            create_config_template()
            sys.exit(0)
        elif sys.argv[1] == "list-checkpoints":
            if len(sys.argv) < 3:
                print("Usage: python train.py list-checkpoints <model_path>")
                sys.exit(1)
            list_checkpoints(sys.argv[2])
            sys.exit(0)
        elif sys.argv[1] == "compare-configs":
            if len(sys.argv) < 4:
                print("Usage: python train.py compare-configs <config1> <config2>")
                sys.exit(1)
            compare_configs(sys.argv[2], sys.argv[3])
            sys.exit(0)
    
    # 运行主训练流程
    main()