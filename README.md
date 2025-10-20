# Mercor AI Text Detection

一个用于检测AI生成文本的机器学习项目，结合了语义嵌入特征和传统统计特征来进行文本分类。

## 项目概述

该项目实现了一个AI文本检测系统，能够区分人类编写的文本和AI生成的文本。系统采用双重特征提取策略：
- **语义嵌入特征**：使用预训练的sentence-transformers模型提取文本的语义信息
- **统计特征**：提取文本的传统统计特征（长度、标点符号、词汇丰富度等）

## 项目结构

```
Mercor AI Text Detection/
├── mercor-ai-text-detection.py    # 主程序文件
├── requirements.txt               # 项目依赖
├── submission.csv                 # 预测结果输出
├── data/                         # 数据目录
└── README.md                     # 项目说明文档
```

## 核心组件

### 1. embedding_extractor 类
负责提取文本的语义嵌入特征：
- 使用 `all-MiniLM-L6-v2` 预训练模型
- 将文本转换为384维向量表示

### 2. statistical_feature_extractor 类
提取传统统计特征，包括：
- **基础特征**：文本长度、单词数量、平均词长
- **标点特征**：逗号、句号、问号、感叹号数量及比例
- **句子特征**：句子数量、平均句长
- **词汇丰富度**：独特词汇数量、类型-标记比率(TTR)
- **大写特征**：大写字母比例、全大写单词数量
- **数字特征**：数字字符数量及比例

### 3. feature_connector 类
整合两种特征提取方法：
- 协调嵌入特征和统计特征的提取
- 将特征向量拼接成完整的特征矩阵

### 4. ai_text_detector 类
AI文本检测分类器：
- 使用逻辑回归模型进行分类
- 支持训练和预测功能

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install pandas>=1.3.0
pip install numpy>=1.19.5
pip install sentence-transformers>=2.2.0
pip install scikit-learn>=1.0.0
pip install xgboost>=1.5.0
```

## 使用方法

### 基本使用

```python
# 导入主程序
from mercor-ai-text-detection import main

# 运行完整的训练和预测流程
submission = main()
```

### 分步使用

```python
import pandas as pd

# 1. 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 特征提取
connector = feature_connector()
X_train, X_test = connector.prepare_feature(train_df, test_df)

# 3. 训练模型
detector = ai_text_detector()
detector.train(X_train, train_df['is_cheating'])

# 4. 预测
predictions = detector.predict(X_test)

# 5. 保存结果
submission = pd.DataFrame({'id': test_df['id'], 'is_cheating': predictions})
submission.to_csv('submission.csv', index=False)
```

## 数据格式

### 输入数据
- `train.csv`: 训练数据，包含 `id`, `answer`, `is_cheating` 列
- `test.csv`: 测试数据，包含 `id`, `answer` 列

### 输出数据
- `submission.csv`: 预测结果，包含 `id`, `is_cheating` 列

## 特征工程详解

### 嵌入特征
- **模型**: all-MiniLM-L6-v2
- **维度**: 384维
- **特点**: 捕捉文本的语义信息和上下文关系

### 统计特征（15维）
1. `length`: 文本长度
2. `words`: 单词数量
3. `avg_word_length`: 平均词长
4. `commas`: 逗号数量
5. `periods`: 句号数量
6. `questions`: 问号数量
7. `exclamations`: 感叹号数量
8. `punctuation_ratio`: 标点符号比例
9. `sentences`: 句子数量
10. `avg_sentence_length`: 平均句长
11. `unique_words`: 独特词汇数量
12. `ttr`: 类型-标记比率
13. `upper_ratio`: 大写字母比例
14. `upper_words`: 全大写单词数量
15. `digits`: 数字字符数量
16. `digit_ratio`: 数字字符比例

## 模型架构

- **分类器**: 逻辑回归 (LogisticRegression)
- **最大迭代次数**: 1000
- **输入特征维度**: 400维 (384维嵌入 + 16维统计特征)
- **输出**: 二分类 (0: 人类文本, 1: AI文本)

## 性能特点

- **高效性**: 结合深度特征和传统特征，提高检测准确性
- **可解释性**: 统计特征提供可解释的文本特征
- **泛化性**: 预训练嵌入模型提供良好的语义泛化能力

## 适用场景

- 学术诚信检测
- 内容审核
- 文真伪鉴别
- 自动化内容评估

## 注意事项

1. 确保输入数据格式正确
2. 预训练模型文件路径可能需要根据环境调整
3. 建议在使用前进行小规模测试
4. 模型性能可能因数据分布差异而变化

## 扩展建议

- 尝试不同的预训练嵌入模型
- 添加更多统计特征
- 实验其他分类算法（如XGBoost、随机森林等）
- 实现交叉验证和超参数调优
- 添加特征重要性分析

## 许可证

本项目仅供学习和研究使用。