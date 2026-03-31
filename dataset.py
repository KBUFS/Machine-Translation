import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import MarianTokenizer
from torch.utils.data import DataLoader
import pickle
import time

def prepare_dataset(cache_dir="./processed_data", force_reprocess=False,batch_size=16):
    """
    准备机器翻译数据集
    Args:
        cache_dir: 处理后数据的保存目录
        force_reprocess: 是否强制重新处理
        batch_size: 批大小
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # 缓存文件路径
    cache_files = {
        "tokenizer": os.path.join(cache_dir, "tokenizer"),
        "train": os.path.join(cache_dir, "train"),
        "val": os.path.join(cache_dir, "val"),
        "test": os.path.join(cache_dir, "test"),
        "metadata": os.path.join(cache_dir, "metadata.pkl"),
    }
    
    # 检查缓存是否存在
    if not force_reprocess and all(os.path.exists(f) for f in cache_files.values()):
        print(f"从缓存加载预处理数据: {cache_dir}")
        start_time = time.time()
        
        # 加载元数据
        with open(cache_files["metadata"], "rb") as f:
            metadata = pickle.load(f)
        
        # 加载分词器
        tokenizer = MarianTokenizer.from_pretrained(cache_files["tokenizer"])
        
        # 加载数据集
        train_dataset = load_from_disk(cache_files["train"])
        val_dataset = load_from_disk(cache_files["val"])
        test_dataset = load_from_disk(cache_files["test"])
        
        load_time = time.time() - start_time
        print(f"加载完成，耗时: {load_time:.2f}秒")
        
    else:
        print("缓存不存在或强制重新处理，开始预处理...")
        start_time = time.time()
        
        # 1. 加载原始数据
        print("加载原始数据集...")
        dataset = load_dataset("opus100", "en-zh", cache_dir="./opus100_data")
        
        # 2. 加载分词器
        print("加载分词器...")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        
        def preprocess_batch(batch):
            """预处理批数据"""
            inputs = [ex["en"] for ex in batch["translation"]]
            targets = [ex["zh"] for ex in batch["translation"]]
            
            # 编码
            model_inputs = tokenizer(
                inputs,  # 源语言
                text_target=targets,  # 目标语言
                max_length=128,     # 最长128个token
                truncation=True,    # 超过128就截断
                padding="max_length",   # 不足128用padding补全
                return_tensors="pt"     # 返回PyTorch tensor
            )
            
            # 现在 model_inputs 包含:
            #   input_ids: 英文的token ID矩阵 [batch_size, 128]
            #   attention_mask: 英文的注意力掩码 [batch_size, 128]，标记哪些是真实token
            #   labels: 中文的token ID矩阵 [batch_size, 128]
            # 处理labels, padding token 需要替换为 -100（中文标签）
            labels = model_inputs["labels"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
            
            return model_inputs
        
        # 3. 处理数据
        print("预处理训练集...")
        train_dataset = dataset["train"].map(
            preprocess_batch,
            batched=True,
            batch_size=1000,
            remove_columns=["translation"]
        )
        
        print("预处理验证集...")
        val_dataset = dataset["validation"].map(
            preprocess_batch,
            batched=True,
            batch_size=1000,
            remove_columns=["translation"]
        )
        
        print("预处理测试集...")
        test_dataset = dataset["test"].map(
            preprocess_batch,
            batched=True,
            batch_size=1000,
            remove_columns=["translation"]
        )
        
        # 4. 保存处理后的数据
        print(f"保存预处理数据到: {cache_dir}")
        
        # 保存分词器
        tokenizer.save_pretrained(cache_files["tokenizer"])
        
        # 保存数据集
        train_dataset.save_to_disk(cache_files["train"])
        val_dataset.save_to_disk(cache_files["val"])
        test_dataset.save_to_disk(cache_files["test"])
        
        # 保存元数据
        metadata = {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "vocab_size": tokenizer.vocab_size,
            "max_length": 128,
            "processed_time": time.time()
        }
        with open(cache_files["metadata"], "wb") as f:
            pickle.dump(metadata, f)
        
        process_time = time.time() - start_time
        print(f"预处理完成，耗时: {process_time:.2f}秒")
    
    # 5. 设置torch格式
    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)
    
    # 6. 创建DataLoader
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 7. 打印信息
    print(f"\n数据集统计:")
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"序列长度: 128")
    
    return train_loader, val_loader, test_loader, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="机器翻译数据集预处理")
    parser.add_argument("--reprocess", action="store_true", help="强制重新处理数据")
    parser.add_argument("--cache_dir", default="./processed_data", help="缓存目录")
    
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader, tokenizer = prepare_dataset(
        cache_dir=args.cache_dir,
        force_reprocess=args.reprocess
    )
    
    # 测试
    batch = next(iter(train_loader))
    print(f"\n批次形状: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
    
    # 解码示例
    print("\n示例翻译:")
    for i in range(2):
        src_text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
        
        tgt_ids = batch["labels"][i].clone()
        tgt_ids[tgt_ids == -100] = tokenizer.pad_token_id
        tgt_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
        
        print(f"\n样本 {i+1}:")
        print(f"  英文: {src_text}")
        print(f"  中文: {tgt_text}")