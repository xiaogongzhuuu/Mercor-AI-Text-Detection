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


def Feature_engineering():


def embedding():


def modeling_functions()


def explain_prediction():



def main():
    train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
    test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
    


if __name__ == "__main__":
    submission=main()
