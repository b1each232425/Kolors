#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单量化测试 - 用 Qwen2-0.5B 模型
RTX 3050 可以轻松跑通
"""

import os
import sys
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_fp16():
    """加载 FP16 模型"""
    print("\n" + "=" * 70)
    print("📦 加载 FP16 模型")
    print("=" * 70 + "\n")

    print("[1/2] 加载 Tokenizer...     ", end="", flush=True)
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        trust_remote_code=True
    )
    print("✅\n")

    print("[2/2] 加载模型 (FP16)...    ", end="", flush=True)
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )

    model.eval()
    print("✅\n")

    # 显存统计
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated() / 1e9

    print("📊 显存统计:")
    print(f"   总显存: {total_memory:.2f} GB")
    print(f"   已用: {allocated_memory:.2f} GB")
    print(f"   使用率: {(allocated_memory / total_memory * 100):.1f}%\n")

    return model, tokenizer


def load_model_4bit():
    """加载 4-bit 量化模型"""
    print("\n" + "=" * 70)
    print("📦 加载 4-bit 量化模型")
    print("=" * 70 + "\n")

    print("[1/2] 加载 Tokenizer...     ", end="", flush=True)
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        trust_remote_code=True
    )
    print("✅\n")

    print("[2/2] 加载模型 (4-bit)...      ", end="", flush=True)
    sys.stdout.flush()

    # 4-bit 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    print("✅\n")

    # 显存统计
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated() / 1e9

    print("📊 显存统计:")
    print(f"   总显存: {total_memory:.2f} GB")
    print(f"   已用: {allocated_memory:.2f} GB")
    print(f"   使用率: {(allocated_memory / total_memory * 100):.1f}%\n")

    return model, tokenizer


def test_inference(model, tokenizer, text="你好，告诉我你是谁"):
    """测试推理"""
    print("=" * 70)
    print("🧪 测试推理")
    print("=" * 70 + "\n")

    print(f"📝 输入文本: {text}\n")
    print("⏳ 推理中...      ", end="", flush=True)
    sys.stdout.flush()

    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.95,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("✅\n")

        print(f"💬 输出文本:\n{response}\n")

        return True

    except Exception as e:
        print(f"❌")
        print(f"\n❌ 推理失败: {str(e)}\n")
        return False


def compare_models():
    """对比 FP16 和 4-bit"""
    print("\n\n" + "=" * 70)
    print("📊 FP16 vs 4-bit 对比")
    print("=" * 70 + "\n")

    # FP16
    print("1️⃣  加载 FP16 模型...")
    torch.cuda.empty_cache()
    gc.collect()

    model_fp16, tokenizer_fp16 = load_model_fp16()
    fp16_memory = torch.cuda.memory_allocated() / 1e9

    # 推理测试
    test_inference(model_fp16, tokenizer_fp16, "你好")

    # 清理
    del model_fp16
    torch.cuda.empty_cache()
    gc.collect()

    # 4-bit
    print("\n2️⃣  加载 4-bit 量化模型...")
    torch.cuda.empty_cache()
    gc.collect()

    model_4bit, tokenizer_4bit = load_model_4bit()
    bit4_memory = torch.cuda.memory_allocated() / 1e9

    # 推理测试
    test_inference(model_4bit, tokenizer_4bit, "你好")

    # 对比表格
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70 + "\n")

    print(f"{'模式':<15} {'显存占用':<15} {'节省比例':<15}")
    print("-" * 45)
    print(f"{'FP16':<15} {fp16_memory:>12.2f} GB {'-':>13}")
    print(f"{'4-bit':<15} {bit4_memory:>12.2f} GB {(1 - bit4_memory / fp16_memory) * 100:>12.1f}%")
    print()


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description="简单量化测试")
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["fp16", "4bit", "compare"],
        help="运行模式"
    )

    args = parser.parse_args()

    if args.mode == "fp16":
        model, tokenizer = load_model_fp16()
        test_inference(model, tokenizer)

    elif args.mode == "4bit":
        model, tokenizer = load_model_4bit()
        test_inference(model, tokenizer)

    elif args.mode == "compare":
        compare_models()

    print("\n✅ 测试完成！\n")


if __name__ == '__main__':
    main()