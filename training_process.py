# training_process.py
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    """简单模型示例"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)  # 3维输入 → 4维隐藏
        self.fc2 = nn.Linear(4, 2)  # 4维隐藏 → 2维输出
    
    def forward(self, x):
        """前向传播：定义如何计算输出"""
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def demonstrate_training_process():
    """演示完整训练流程"""
    print("="*60)
    print("神经网络训练完整流程演示")
    print("="*60)
    
    # 1. 准备
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器
    
    # 模拟数据
    inputs = torch.randn(4, 3)  # 4个样本，每个3维
    targets = torch.tensor([0, 1, 0, 1])  # 真实标签
    
    print(f"模型结构:")
    print(f"  fc1: {model.fc1}")
    print(f"  fc2: {model.fc2}")
    print(f"\n输入: {inputs.shape}")
    print(f"标签: {targets}")
    
    # 记录初始权重
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.clone()
    
    # 2. 训练循环（1次迭代）
    print(f"\n{'='*60}")
    print("开始一次训练迭代")
    print("="*60)
    
    # ----- 第1步：前向传播 -----
    print(f"\n[1] 前向传播 (Forward Pass)")
    print(f"  输入: {inputs.shape}")
    
    outputs = model(inputs)  # 调用 forward 方法
    print(f"  模型输出: {outputs.shape}")
    print(f"  预测值:\n{outputs.detach().numpy()}")
    
    # ----- 第2步：计算损失 -----
    print(f"\n[2] 计算损失 (Loss Calculation)")
    loss = criterion(outputs, targets)
    print(f"  损失值: {loss.item():.4f}")
    
    # ----- 第3步：反向传播 -----
    print(f"\n[3] 反向传播 (Backward Pass)")
    print(f"  清理旧梯度...")
    optimizer.zero_grad()  # 重要：清空前一次的梯度
    
    print(f"  计算梯度...")
    loss.backward()  # 反向传播，计算梯度
    
    # 查看梯度
    print(f"  梯度示例:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"    {name}: 梯度范数 = {grad_norm:.6f}")
    
    # ----- 第4步：优化器更新 -----
    print(f"\n[4] 优化器步骤 (Optimizer Step)")
    print(f"  使用梯度下降更新权重...")
    optimizer.step()  # 根据梯度更新权重
    
    # 比较权重变化
    print(f"\n权重变化:")
    for name, param in model.named_parameters():
        weight_change = torch.abs(param.data - initial_weights[name]).mean().item()
        print(f"  {name}: 平均变化 = {weight_change:.6f}")
    
    return model, loss

def visualize_gradient_flow():
    """可视化梯度流动"""
    print("\n" + "="*60)
    print("梯度流动可视化")
    print("="*60)
    
    # 创建更简单的计算图
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    w = torch.tensor([0.5, 0.3], requires_grad=True)
    b = torch.tensor([0.1], requires_grad=True)
    
    print(f"输入: x = {x}")
    print(f"权重: w = {w}")
    print(f"偏置: b = {b}")
    
    # 前向传播
    y = x * w  # 元素乘法
    z = y.sum() + b
    loss = z * 2  # 简单损失
    
    print(f"\n前向传播计算:")
    print(f"  y = x * w = {x} * {w} = {y}")
    print(f"  z = sum(y) + b = {y.sum()} + {b} = {z}")
    print(f"  loss = z * 2 = {z} * 2 = {loss}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n反向传播梯度:")
    print(f"  ∂loss/∂x = {x.grad}")  # 梯度 = 2 * w
    print(f"  ∂loss/∂w = {w.grad}")  # 梯度 = 2 * x
    print(f"  ∂loss/∂b = {b.grad}")  # 梯度 = 2
    
    # 验证
    print(f"\n手动验证梯度:")
    print(f"  ∂loss/∂x₁ = 2 * w₁ = 2 * 0.5 = 1.0 ✓")
    print(f"  ∂loss/∂w₁ = 2 * x₁ = 2 * 1.0 = 2.0 ✓")
    print(f"  ∂loss/∂b = 2 = 2.0 ✓")

if __name__ == "__main__":
    # 1. 演示完整训练流程
    model, loss = demonstrate_training_process()
    
    # 2. 可视化梯度
    visualize_gradient_flow()
    
    # 3. 总结
    print("\n" + "="*60)
    print("关键概念总结")
    print("="*60)
    
    summary = """
🔹 前向传播 (Forward Propagation):
    - 目的: 计算模型的预测输出
    - 触发: 调用 model(inputs) 或 model.forward(inputs)
    - 结果: 得到预测值，用于计算损失

🔹 损失计算 (Loss Calculation):
    - 目的: 量化模型预测的错误程度
    - 公式: loss = criterion(predictions, targets)
    - 结果: 单个标量值，表示"有多错"

🔹 反向传播 (Backward Propagation/Backpropagation):
    - 目的: 计算每个参数对损失的贡献（梯度）
    - 触发: loss.backward()
    - 结果: 每个参数的 .grad 属性被填充

🔹 优化器步骤 (Optimizer Step):
    - 目的: 根据梯度更新模型参数，也就是根据反向传播的梯度进行优化参数
    - 触发: optimizer.step()
    - 算法: 梯度下降、Adam 等
    
🔹 前馈网络 (Feed-Forward Network, FNN):
    - 定义: 一种网络结构，信号单向流动
    - 位置: Transformer 中的一个组件
    - 关系: 在前向传播中被调用
    """
    
    print(summary)