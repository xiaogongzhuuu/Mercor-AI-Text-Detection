import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class embedding_extractor():  #embedding提取
    def __init__(self, model_name="/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(model_name)

    def extract_embedding(self,df):
        texts = df["answer"].tolist()
        embeddings =self.model.encode(texts)
        return embeddings
        
class statistical_feature_extractor():#传统统计特征提取
    def extract_feature(self, df):
      features = pd.DataFrame()

      # === 基础特征 ===
      features['length'] = df['answer'].str.len()
      features['words'] = df['answer'].str.split().str.len()
      features['avg_word_length'] = features['length'] / (features['words'] + 1)

      # === 标点特征 ===
      features['commas'] = df['answer'].str.count(',')
      features['periods'] = df['answer'].str.count(r'\.')
      features['questions'] = df['answer'].str.count(r'\?')
      features['exclamations'] = df['answer'].str.count(r'!')
      features['punctuation_ratio'] = (features['commas'] + features['periods'] +
                                      features['questions'] + features['exclamations']) / (features['length'] + 1)

      # === 句子特征 ===
      features['sentences'] = df['answer'].str.count(r'[.!?]+')
      features['avg_sentence_length'] = features['words'] / (features['sentences'] + 1)

      # === 词汇丰富度 ===
      features['unique_words'] = df['answer'].apply(lambda x: len(set(str(x).lower().split())))
      features['ttr'] = features['unique_words'] / (features['words'] + 1)  # Type-Token Ratio

      # === 大写字母特征 ===
      features['upper_ratio'] = df['answer'].str.count(r'[A-Z]') / (features['length'] + 1)
      features['upper_words'] = df['answer'].str.split().apply(lambda x: sum(1 for w in x if w.isupper()))

      # === 数字特征 ===
      features['digits'] = df['answer'].str.count(r'\d')
      features['digit_ratio'] = features['digits'] / (features['length'] + 1)

      # 1. 连续性特征
      features['avg_word_freq'] = df['answer'].apply(self._word_frequency_variance)
        
        # 2. 结构规律性
      features['sentence_length_variance'] = df['answer'].apply(self._sentence_length_variance)
        
        # 3. 常见AI短语检测
      ai_phrases = ['as an ai', 'i am an', 'i cannot', 'i apologize', 'in conclusion', 
                      'furthermore', 'moreover', 'it is important to note']
      for phrase in ai_phrases:
          features[f'phrase_{phrase.replace(" ", "_")}'] = df['answer'].str.lower().str.contains(phrase).astype(int)
        
        # 4. 停用词比例
      features['stopword_ratio'] = df['answer'].apply(self._stopword_ratio)
        
        # 5. 平均词长方差
      features['word_length_variance'] = df['answer'].apply(self._word_length_variance)
      
      return features.values

    def _word_frequency_variance(self, text):
        """计算词频方差（AI文本通常词频更均匀）"""
        words = str(text).lower().split()
        if len(words) < 2:
            return 0
        from collections import Counter
        word_counts = Counter(words)
        frequencies = list(word_counts.values())
        return np.var(frequencies) if len(frequencies) > 1 else 0

    def _sentence_length_variance(self, text):
        """计算句子长度方差"""
        import re
        sentences = re.split(r'[.!?]+', str(text))
        lengths = [len(s.split()) for s in sentences if s.strip()]
        return np.var(lengths) if len(lengths) > 1 else 0

    def _stopword_ratio(self, text):
        """计算停用词比例"""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        words = str(text).lower().split()
        if not words:
            return 0
        stop_count = sum(1 for w in words if w in stopwords)
        return stop_count / len(words)

    def _word_length_variance(self, text):
        """计算词长方差"""
        words = str(text).split()
        if len(words) < 2:
            return 0
        lengths = [len(w) for w in words]
        return np.var(lengths)

        
        

class feature_connector():#embedding与特征工程整合
    def __init__(self):
        self.embedding_extractor = embedding_extractor()
        self.statistical_extractor = statistical_feature_extractor()

    def prepare_feature(self,train_df,test_df):
        train_emb = self.embedding_extractor.extract_embedding(train_df)
        test_emb = self.embedding_extractor.extract_embedding(test_df)
        print("=== 第1步：提取Embedding特征 ===")
        print(f"Embedding示例: {train_emb[0][:5]}...")


        train_stat = self.statistical_extractor.extract_feature(train_df)
        test_stat = self.statistical_extractor.extract_feature(test_df)
        print("=== 第2步：提取传统特征 ===")
        print(f"传统特征示例: {train_stat[0][:5]}")

        X_train=np.hstack([train_emb,train_stat])
        X_test = np.hstack([test_emb, test_stat])
        print("=== 第3步：特征拼接 ===")
        print(f"完整特征示例: {X_train[0][:5]}...")

        return X_train,X_test


class ai_text_detector():
    def __init__(self):
        self.model=LogisticRegression(C=0.5,max_iter=1000)

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    
    def predict(self,X_test):
        return self.model.predict(X_test)
        



def main():
    train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
    test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

    # 特征提取
    connector = feature_connector()
    X_train, X_test = connector.prepare_feature(train_df, test_df)

      # 训练预测
    detector = ai_text_detector()
    detector.train(X_train, train_df['is_cheating'])
    predictions = detector.predict(X_test)

      # 保存结果
    submission = pd.DataFrame({'id': test_df['id'], 'is_cheating': predictions})
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    return submission
    


if __name__ == "__main__":
    submission=main()