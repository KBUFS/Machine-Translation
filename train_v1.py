#采用预训练模型权重，在本地数据集中对全参数进行训练调整,在CPU上训练极慢
import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def train_model(model, train_loader, val_loader, tokenizer, num_epochs=3, save_path="./model_checkpoints"):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
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
    
    # 早停机制
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
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
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            # 更新进度条
            train_iterator.set_postfix({"loss": loss.item()})
            
            # 每100个batch打印一次
            if batch_idx % 100 == 0:
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
            
            # 保存模型
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"模型保存到: {checkpoint_path}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = f"{save_path}/best_model"
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                print(f"✨ 新的最佳模型! 验证损失: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"验证损失未改善，耐心计数: {patience_counter}/{patience}")
        
        # ===== 早停检查 =====
        if patience_counter >= patience:
            print(f"⏹️ 早停触发! 在 epoch {epoch+1} 停止")
            break
        
        # ===== 测试翻译效果 =====
        if epoch % 1 == 0:  # 每个epoch都测试
            test_translation(model, tokenizer, device)
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses, len(train_losses))
    
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
            "Deep learning is a subset of machine learning.",
            "The conference will be held next month.",
            "We need to improve the performance of the model."
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
                num_beams=5,  # 使用beam search提高质量
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
    
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='s', linewidth=2)
    
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加数据标签
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
        plt.text(i, train_loss, f'{train_loss:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, val_loss, f'{val_loss:.3f}', ha='center', va='top', fontsize=9)
    
    # 绘制损失下降百分比
    plt.subplot(2, 1, 2)
    if len(train_losses) > 1:
        train_improvement = [(train_losses[i-1] - train_losses[i])/train_losses[i-1]*100 
                           for i in range(1, len(train_losses))]
        val_improvement = [(val_losses[i-1] - val_losses[i])/val_losses[i-1]*100 
                          for i in range(1, len(val_losses))]
        
        epochs_imp = range(2, num_epochs + 1)
        plt.bar([e-0.2 for e in epochs_imp], train_improvement, width=0.4, label='Train Improvement', alpha=0.7)
        plt.bar([e+0.2 for e in epochs_imp], val_improvement, width=0.4, label='Val Improvement', alpha=0.7)
        
        plt.title('Loss Improvement (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Improvement %', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'loss_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("="*60)
    print("开始机器翻译模型训练")
    print("="*60)
    
    # 1. 加载数据（从你的dataset模块）
    print("\n1. 加载数据集...")
    from dataset import prepare_dataset
    train_loader, val_loader, test_loader, tokenizer = prepare_dataset()
    
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    print(f"测试批次数量: {len(test_loader)}")
    
    # 2. 加载预训练模型
    print("\n2. 加载预训练模型...")
    model_cache_dir = "./offline_model" 
    
    try:
        # 先尝试从本地加载
        model = MarianMTModel.from_pretrained(model_cache_dir, local_files_only=True)
        print(f"✓ 从本地加载模型成功: {model_cache_dir}")
    except:
        # 如果本地没有，再从网络下载
        print("本地模型不存在，从网络下载...")
        model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-zh",
            cache_dir="./models",  # 保存到本地目录
            resume_download=True   # 支持断点续传
        )
        print("✓ 从网络下载模型成功")
    
    # 3. 分析模型
    print("\n3. 模型信息:")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试初始翻译效果
    print("\n4. 测试初始翻译效果（训练前）:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_translation(model, tokenizer, device)
    
    # 4. 开始训练
    print("\n5. 开始训练...")
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        num_epochs=3,  # 先训练3个epoch
        save_path="./model_checkpoints"
    )
    
    # 5. 最终测试
    print("\n6. 最终测试结果:")
    test_translation(trained_model, tokenizer, device)
    
    # 6. 保存最终模型
    print("\n7. 保存最终模型...")
    final_model_path = "./final_model"
    trained_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"   最终模型保存到: {final_model_path}")
    
    # 7. 训练总结
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"总训练轮数: {len(train_losses)}")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")
    print(f"训练损失降低: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%")
    if len(val_losses) > 1:
        print(f"验证损失降低: {(val_losses[0] - val_losses[-1])/val_losses[0]*100:.1f}%")
    
    return trained_model

if __name__ == "__main__":
    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    model = main()