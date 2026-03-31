from transformers import MarianMTModel, MarianTokenizer
import os

# 创建目录
os.makedirs("./offline_model", exist_ok=True)

print("下载模型和分词器...")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

print("保存到本地...")
model.save_pretrained("./offline_model")
tokenizer.save_pretrained("./offline_model")

print("下载完成！文件列表:")
for file in os.listdir("./offline_model"):
    print(f"  {file}")