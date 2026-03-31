# train_lora.py
"""
使用 LoRA 进行高效微调
不直接修改原始权重，而是在权重旁添加一个低秩适配器。
- 只训练 0.1-5% 的参数
- 训练速度提升 5-10 倍
- 内存占用大幅降低
- 效果接近全量微调
"""
import torch
import os
from peft import LoraConfig, get_peft_model, TaskType
from transformers import MarianMTModel, MarianTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import argparse

def setup_lora_model(model, lora_r=8, lora_alpha=32, lora_dropout=0.1):
    """
    为模型设置 LoRA 适配器
    
    Args:
        model: 原始模型
        lora_r: LoRA 的秩（rank），控制参数数量
        r=2: 极低参数 (0.1%)，训练最快，效果最差 r=4: 低参数 (0.2%)，训练快r=8: 推荐 (0.5%)，平衡效果和速度r=16: 高参数 (1%)，效果更好 r=32: 非常高参数 (2%)，接近全量微调
        lora_alpha: 缩放系数
        lora_dropout: dropout 率
    """
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # 序列到序列任务
        r=lora_r,                         # 低秩矩阵的秩
        lora_alpha=lora_alpha,            # 缩放系数
        target_modules=["q_proj", "v_proj"],  # 要应用LoRA的模块
        lora_dropout=lora_dropout,        # dropout
        bias="none",                      # 不训练偏置
    )
    
    # 应用 LoRA
    lora_model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    lora_model.print_trainable_parameters()
    
    return lora_model

def train_model_lora(model, train_loader, val_loader, tokenizer, num_epochs=3, save_path="./lora_checkpoints"):
    """使用 LoRA 训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)
    
    # 优化器 - 只训练可训练参数
    optimizer = AdamW(model.parameters(), lr=1e-3)  # LoRA 可以用更大的学习率
    
    # 学习率调度器
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # 记录损失
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # ===== 训练阶段 =====
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        # 使用tqdm进度条
        train_iterator = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", unit="batch")
        
        for batch_idx, batch in enumerate(train_iterator):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            # 更新进度条
            train_iterator.set_postfix({"loss": loss.item()})
            
            # 每50个batch打印一次（LoRA训练快）
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_time = time.time() - epoch_start_time
        
        print(f"平均训练损失: {avg_train_loss:.4f}")
        print(f"训练时间: {train_time:.2f}秒")
        
        # ===== 验证阶段 =====
        val_start_time = time.time()
        avg_val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(avg_val_loss)
        val_time = time.time() - val_start_time
        
        print(f"平均验证损失: {avg_val_loss:.4f}")
        print(f"验证时间: {val_time:.2f}秒")
        
        # ===== 保存检查点 =====
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            checkpoint_path = f"{save_path}/epoch_{epoch+1}"
            
            # 保存 LoRA 适配器
            model.save_pretrained(checkpoint_path)  # 只保存 LoRA 权重
            tokenizer.save_pretrained(checkpoint_path)
            print(f"LoRA 适配器保存到: {checkpoint_path}")
        
        # ===== 测试翻译效果 =====
        test_translation(model, tokenizer, device)
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses, num_epochs)
    
    return model, train_losses, val_losses

def evaluate_model(model, val_loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc="验证", unit="batch")
        for batch in val_iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            val_iterator.set_postfix({"val_loss": outputs.loss.item()})
    
    return total_loss / len(val_loader)

def test_translation(model, tokenizer, device, test_sentences=None):
    """测试翻译效果"""
    if test_sentences is None:
        test_sentences = [
            "Hello, how are you?",
            "I love machine learning.",
            "The weather is nice today.",
            "What is your name?",
            "This is a test sentence for translation.",
        ]
    
    print("\n" + "="*60)
    print("测试翻译效果:")
    print("="*60)
    
    model.eval()
    
    for i, sentence in enumerate(test_sentences, 1):
        # 编码输入
        inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成翻译
        with torch.no_grad():
            translated = model.generate(
                **inputs, 
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # 解码
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        print(f"\n{i}. 英文: {sentence}")
        print(f"   中文: {translation}")
    
    print("="*60)

def plot_loss_curves(train_losses, val_losses, num_epochs):
    """绘制损失曲线"""
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='s', linewidth=2)
    
    plt.title('LoRA Training - Loss Curves', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'lora_loss_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def merge_lora_with_base_model(lora_model_path, base_model_path, output_path):
    """
    将 LoRA 适配器合并到基础模型中
    """
    from peft import PeftModel
    
    # 加载基础模型
    base_model = MarianMTModel.from_pretrained(base_model_path)
    
    # 加载 LoRA 适配器
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # 合并权重
    merged_model = lora_model.merge_and_unload()
    
    # 保存合并后的模型
    merged_model.save_pretrained(output_path)
    print(f"合并模型保存到: {output_path}")
    
    return merged_model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 LoRA 进行高效微调")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lora_r", type=int, default=2, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--cache_dir", type=str, default="./processed_data", help="数据缓存目录")
    
    args = parser.parse_args()
    
    print("="*60)
    print("LoRA 高效微调 - 机器翻译")
    print("="*60)
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"LoRA 秩 (r): {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"学习率: {args.learning_rate}")
    
    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    from dataset import prepare_dataset
    
    # 修改 dataset.py 的 prepare_dataset 函数，添加 batch_size 参数
    # 或者在这里修改 DataLoader
    train_loader, val_loader, test_loader, tokenizer = prepare_dataset(
        cache_dir=args.cache_dir,
        force_reprocess=False
    )
    
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    
    # 2. 加载预训练模型
    print("\n[2/5] 加载预训练模型...")
    model_cache_dir = "./offline_model" 
    
    try:
        # 先尝试从本地加载
        base_model = MarianMTModel.from_pretrained(model_cache_dir, local_files_only=True)
        print(f"✓ 从本地加载模型成功: {model_cache_dir}")
    except:
        # 如果本地没有，再从网络下载
        print("本地模型不存在，从网络下载...")
        base_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-zh",
            cache_dir="./models",
            resume_download=True
        )
        print("✓ 从网络下载模型成功")
    
    # 3. 应用 LoRA
    print("\n[3/5] 应用 LoRA 适配器...")
    lora_model = setup_lora_model(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1
    )
    
    # 4. 测试初始翻译
    print("\n[4/5] 测试初始翻译效果（训练前）:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_translation(lora_model, tokenizer, device)
    
    # 5. 开始 LoRA 训练
    print("\n[5/5] 开始 LoRA 训练...")
    trained_model, train_losses, val_losses = train_model_lora(
        model=lora_model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
        save_path="./lora_checkpoints"
    )
    
    # 6. 最终测试
    print("\n最终测试结果:")
    test_translation(trained_model, tokenizer, device)
    
    # 7. 保存最终 LoRA 适配器
    print("\n保存最终 LoRA 适配器...")
    final_lora_path = "./final_lora_model"
    trained_model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)
    print(f"LoRA 适配器保存到: {final_lora_path}")
    
    # 8. 可选：合并 LoRA 到基础模型
    print("\n是否合并 LoRA 适配器到基础模型？")
    response = input("输入 'yes' 合并，或按回车跳过: ")
    if response.lower() == 'yes':
        merged_model = merge_lora_with_base_model(
            final_lora_path,
            model_cache_dir,
            "./merged_final_model"
        )
    
    return trained_model

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    model = main()