#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kolors 量化框架 - 第1步：模型加载 + 4-bit 量化（稳定版）
"""

import os
import sys
import torch
import gc
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models(quantize_mode="fp16", device="cuda"):
    """
    加载所有模型的框架函数

    Args:
        quantize_mode: "fp16" 或 "4bit"
        device: "cuda" 或 "cpu"

    Returns:
        dict: 包含所有加载的模型
    """

    ckpt_dir = f'{root_dir}/weights/Kolors'
    models = {}

    print("\n" + "=" * 70)
    print("📦 Kolors 模型加载框架")
    print("=" * 70)
    print(f"\n💡 配置:")
    print(f"   量化模式: {quantize_mode}")
    print(f"   设备: {device}")
    print(f"   模型路径: {ckpt_dir}\n")

    try:
        # ========== 加载文本编码器（支持 4-bit 量化）==========
        print("[1/5] 加载文本编码器...     ", end="", flush=True)
        sys.stdout.flush()

        if quantize_mode == "4bit":
            print("\n   → 使用 4-bit 量化")
            sys.stdout.flush()

            from transformers import BitsAndBytesConfig
            import warnings
            warnings.filterwarnings("ignore")

            # 4-bit 量化配置
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            print("   → 开始加载模型（可能需要一段时间）...    ", end="", flush=True)
            sys.stdout.flush()

            text_encoder = ChatGLMModel.from_pretrained(
                f'{ckpt_dir}/text_encoder',
                quantization_config=quantization_config,
                device_map="sequential",  # 改用 sequential 而不是 auto
                low_cpu_mem_usage=True,
            )

            print("✅")
            print("   ✅ 4-bit 量化成功\n")

        else:  # fp16
            print("\n   → 使用 FP16 精度")
            sys.stdout.flush()

            text_encoder = ChatGLMModel.from_pretrained(
                f'{ckpt_dir}/text_encoder',
                torch_dtype=torch.float16,
            ).half()

            if device == "cuda":
                text_encoder = text_encoder.to("cuda")

            print("   ✅ FP16 加载成功\n")

        text_encoder.eval()
        models['text_encoder'] = text_encoder
        print("✅\n")

        torch.cuda.empty_cache()
        gc.collect()

        # ========== 加载 Tokenizer ==========
        print("[2/5] 加载 Tokenizer...   ", end="", flush=True)
        sys.stdout.flush()

        tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
        models['tokenizer'] = tokenizer
        print("✅\n")

        torch.cuda.empty_cache()
        gc.collect()

        # ========== 加载 VAE ==========
        print("[3/5] 加载 VAE...   ", end="", flush=True)
        sys.stdout.flush()

        vae = AutoencoderKL.from_pretrained(
            f"{ckpt_dir}/vae",
            revision=None,
            torch_dtype=torch.float16,
        ).half()

        if device == "cuda":
            vae = vae.to("cuda")

        vae.eval()
        models['vae'] = vae
        print("✅\n")

        torch.cuda.empty_cache()
        gc.collect()

        # ========== 加载 UNet ==========
        print("[4/5] 加载 UNet...    ", end="", flush=True)
        sys.stdout.flush()

        unet = UNet2DConditionModel.from_pretrained(
            f"{ckpt_dir}/unet",
            revision=None,
            torch_dtype=torch.float16,
        ).half()

        if device == "cuda":
            unet = unet.to("cuda")

        unet.eval()
        models['unet'] = unet
        print("✅\n")

        torch.cuda.empty_cache()
        gc.collect()

        # ========== 加载 Scheduler ==========
        print("[5/5] 加载 Scheduler...   ", end="", flush=True)
        sys.stdout.flush()

        scheduler = EulerDiscreteScheduler.from_pretrained(
            f"{ckpt_dir}/scheduler"
        )

        models['scheduler'] = scheduler
        print("✅\n")

        # ========== 打印加载信息 ==========
        print("=" * 70)
        print("✅ 所有模型加载成功！")
        print("=" * 70)

        # 显存统计
        if device == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9

            print(f"\n📊 显存统计:")
            print(f"   总显存: {total_memory:. 2f} GB")
            print(f"   已用: {allocated_memory:.2f} GB")
            print(f"   峰值: {peak_memory:. 2f} GB")

        print(f"\n📦 加载的模型:")
        print(f"   ✅ text_encoder (文本编码器) - {quantize_mode}")
        print(f"   ✅ tokenizer (分词器)")
        print(f"   ✅ vae (变分自编码器) - FP16")
        print(f"   ✅ unet (扩散模型) - FP16")
        print(f"   ✅ scheduler (采样调度器)")
        print()

        return models

    except Exception as e:
        print(f"❌\n")
        print(f"❌ 加载失败: {type(e).__name__}")
        print(f"   {str(e)}\n")

        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_models(models):
    """
    测试模型是否正常加载

    Args:
        models: load_models() 返回的模型字典
    """

    print("=" * 70)
    print("🧪 测试模型")
    print("=" * 70 + "\n")

    try:
        # 测试 text_encoder
        print("[1/3] 测试 text_encoder...     ", end="", flush=True)
        sys.stdout.flush()

        tokenizer = models['tokenizer']
        text_encoder = models['text_encoder']

        test_text = "你好"
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            device = next(text_encoder.parameters()).device
            input_ids = inputs.input_ids.to(device)
            output = text_encoder(input_ids)

        print("✅")
        print(f"   输入: '{test_text}'")
        output_shape = output[0].shape if isinstance(output, tuple) else output.shape
        print(f"   输出形状: {output_shape}\n")

        # 测试 VAE
        print("[2/3] 测试 VAE...    ", end="", flush=True)
        sys.stdout.flush()

        vae = models['vae']

        # 创建假的图像张量
        fake_image = torch.randn(1, 4, 32, 32, dtype=torch.float16)
        if torch.cuda.is_available():
            fake_image = fake_image.to("cuda")

        with torch.no_grad():
            vae_output = vae.decode(fake_image).sample

        print("✅")
        print(f"   输入形状: {fake_image.shape}")
        print(f"   输出形状: {vae_output.shape}\n")

        # 测试 UNet
        print("[3/3] 测试 UNet...    ", end="", flush=True)
        sys.stdout.flush()

        unet = models['unet']

        # 创建假的张量
        latents = torch.randn(1, 4, 32, 32, dtype=torch.float16)
        timestep = torch.tensor([0], dtype=torch.long)
        encoder_hidden_states = torch.randn(1, 77, 4096, dtype=torch.float16)

        if torch.cuda.is_available():
            latents = latents.to("cuda")
            timestep = timestep.to("cuda")
            encoder_hidden_states = encoder_hidden_states.to("cuda")

        with torch.no_grad():
            unet_output = unet(latents, timestep, encoder_hidden_states).sample

        print("✅")
        print(f"   latents 形状: {latents.shape}")
        print(f"   输出形状: {unet_output.shape}\n")

        print("=" * 70)
        print("✅ 所有模型测试通过！")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"❌")
        print(f"\n❌ 测试失败: {type(e).__name__}")
        print(f"   {str(e)}\n")

        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description="Kolors 模型加载框架")
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default="fp16",
        choices=["fp16", "4bit"],
        help="量化模式: fp16 或 4bit"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="加载后是否进行模型测试"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备: cuda 或 cpu"
    )

    args = parser.parse_args()

    # 加载模型
    models = load_models(quantize_mode=args.quantize_mode, device=args.device)

    # 可选：测试模型
    if args.test:
        test_models(models)

    print("✅ 框架构建完成！模型已准备好进行下一步操作\n")

    return models


if __name__ == '__main__':
    main()