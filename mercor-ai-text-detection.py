import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

class embedding_extractor:  #embedding提取
    def __init__(self, model_name=""):
        from sentence_transformers import SentenceTranceformer
        self.model=SentenceTranceformer(model_name)

    def extract_embedding(self,df):
        texts = df["answer"].tolist()
        embeddings =self.model.encode(texts)
        return embeddings
        
class statistical_feature_extractor():#传统统计特征提取
    def __init__(self):
        pass

    def extract_feature(df):
        pass

class feature_connector():
    def __init__(self):
        pass

    def prepare_feature(self,train,test):
        pass

class ai_text_detector():
    def __init__(self):
        self.model=LogisticRegression()

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    
    def predict(self,X_test):
        return self.model.predict_proba(X_test)






def main():
    train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
    test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

    # 特征提取
      connector = feature_connector()
      X_train, X_test = connector.prepare(train, test)

      # 训练预测
      detector = ai_detector()
      detector.train(X_train, train['is_cheating'])
      predictions = detector.predict(X_test)

      # 保存结果
      submission = pd.DataFrame({'id': test['id'], 'is_cheating': predictions})
      submission.to_csv('submission.csv', index=False)
      return submission
    


if __name__ == "__main__":
    submission=main()
