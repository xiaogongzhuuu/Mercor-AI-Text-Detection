"""
Mercor AI Text Detection - CHAMPION VERSION
===========================================
Target: 0.98+ CV AUC by focusing on what actually works
"""

import pandas as pd
import numpy as np
import re
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
import warnings
warnings.filterwarnings('ignore')

print("所有库导入完成!")

class ChampionFeatureExtractor:
    """冠军特征提取器 - 专注于真正有效的特征"""
    
    def __init__(self):
        # 经过验证的最有效AI指示器
        self.ai_connectors = [
            'in conclusion', 'in summary', 'furthermore', 'moreover', 
            'additionally', 'however', 'therefore', 'thus', 'consequently',
            'as a result', 'on the other hand', 'for instance', 'for example',
            'it is important to note', 'it is worth noting', 'that being said'
        ]
        
        self.formal_words = [
            'utilize', 'facilitate', 'implement', 'methodology', 'paradigm',
            'leverage', 'robust', 'optimal', 'enhance', 'demonstrate'
        ]
    
    def extract_champion_features(self, df):
        """提取冠军特征 - 基于0.978版本的成功经验"""
        features = pd.DataFrame()
        
        # === 基础指标 ===
        features['text_length'] = df['answer'].str.len()
        features['word_count'] = df['answer'].str.split().str.len()
        features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
        
        # === 句子分析 ===
        features['sentence_count'] = df['answer'].str.count(r'[.!?]+')
        features['avg_sentence_length'] = features['word_count'] / (features['sentence_count'] + 1)
        
        # === 标点模式 ===
        features['comma_count'] = df['answer'].str.count(',')
        features['period_count'] = df['answer'].str.count(r'\.')
        features['total_punctuation'] = features['comma_count'] + features['period_count']
        features['punctuation_ratio'] = features['total_punctuation'] / (features['text_length'] + 1)
        
        # === 词汇丰富度 ===
        features['unique_words'] = df['answer'].apply(lambda x: len(set(str(x).lower().split())))
        features['ttr'] = features['unique_words'] / (features['word_count'] + 1)
        
        # === AI特定模式 ===
        features['ai_connector_density'] = df['answer'].apply(
            lambda x: sum(1 for phrase in self.ai_connectors if phrase in str(x).lower()) / (len(str(x).split()) + 1)
        )
        
        features['formal_word_ratio'] = df['answer'].apply(
            lambda x: sum(1 for word in self.formal_words if word in str(x).lower()) / (len(str(x).split()) + 1)
        )
        
        # === 被动语态检测 ===
        passive_indicators = ['is made', 'was made', 'is given', 'was given', 'is shown', 'was shown']
        features['passive_voice_ratio'] = df['answer'].apply(
            lambda x: sum(1 for phrase in passive_indicators if phrase in str(x).lower()) / (len(str(x).split()) + 1)
        )
        
        # === 结构复杂度 ===
        features['subordinate_ratio'] = df['answer'].str.count(r'\b(that|which|who|when|where|while|although|because|if)\b') / (features['word_count'] + 1)
        
        # === 一致性指标 ===
        features['word_length_std'] = df['answer'].apply(
            lambda x: np.std([len(w) for w in str(x).split()]) if len(str(x).split()) > 1 else 0
        )
        
        # === 主题特征 ===
        topic_dummies = pd.get_dummies(df['topic'], prefix='topic')
        
        return pd.concat([features, topic_dummies], axis=1)

class ChampionAIDetector:
    """冠军AI检测器 - 专注于最有效的模型组合"""
    
    def __init__(self):
        self.feature_extractor = ChampionFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
    
    def prepare_champion_features(self, train_df, test_df):
        """准备冠军特征集"""
        print("提取冠军特征...")
        train_features = self.feature_extractor.extract_champion_features(train_df)
        test_features = self.feature_extractor.extract_champion_features(test_df)
        
        # 对齐列
        test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
        
        # TF-IDF特征（基于0.978版本的成功经验）
        print("创建TF-IDF特征...")
        tfidf_word = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            stop_words='english'
        )
        
        tfidf_char = TfidfVectorizer(
            max_features=1500,
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        )
        
        train_tfidf_word = tfidf_word.fit_transform(train_df['answer'])
        test_tfidf_word = tfidf_word.transform(test_df['answer'])
        
        train_tfidf_char = tfidf_char.fit_transform(train_df['answer'])
        test_tfidf_char = tfidf_char.transform(test_df['answer'])
        
        # 组合所有特征
        X_train = np.hstack([
            train_tfidf_word.toarray(),
            train_tfidf_char.toarray(),
            train_features.values
        ])
        
        X_test = np.hstack([
            test_tfidf_word.toarray(),
            test_tfidf_char.toarray(),
            test_features.values
        ])
        
        print(f"最终特征维度: {X_train.shape}")
        return X_train, X_test
    
    def train_champion_ensemble(self, train_df, n_folds=10):
        """训练冠军集成模型"""
        X_train, _ = self.prepare_champion_features(train_df, train_df)  # 只是为了获取特征维度
        y_train = train_df['is_cheating'].values
        
        print(f"训练集大小: {X_train.shape}")
        print(f"正样本比例: {y_train.mean():.6f}")
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []
        oof_predictions = np.zeros(len(train_df))
        
        # 存储每个fold的测试预测
        test_preds = {
            'lgb': np.zeros(len(train_df)),  # 临时存储，实际我们会重新计算
            'xgb': np.zeros(len(train_df)),
            'cat': np.zeros(len(train_df)),
            'lr': np.zeros(len(train_df))
        }
        
        actual_test_preds = {
            'lgb': None,
            'xgb': None, 
            'cat': None,
            'lr': None
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n训练 Fold {fold+1}/{n_folds}")
            
            # 为每个fold重新准备特征（避免数据泄露）
            X_train_fold, X_val_fold = self.prepare_champion_features(
                train_df.iloc[train_idx], train_df.iloc[val_idx]
            )
            
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            # === LightGBM (优化参数) ===
            lgb_model = lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.01,
                max_depth=7,
                num_leaves=63,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=-1
            )
            
            lgb_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
            )
            
            lgb_pred = lgb_model.predict_proba(X_val_fold)[:, 1]
            test_preds['lgb'][val_idx] = lgb_pred
            
            # === XGBoost ===
            xgb_model = xgb.XGBClassifier(
                n_estimators=2000,
                learning_rate=0.01,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42 + fold,
                eval_metric='auc',
                tree_method='hist',
                early_stopping_rounds=150,
                n_jobs=-1
            )
            
            xgb_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            xgb_pred = xgb_model.predict_proba(X_val_fold)[:, 1]
            test_preds['xgb'][val_idx] = xgb_pred
            
            # === CatBoost ===
            cat_model = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42 + fold,
                verbose=0,
                early_stopping_rounds=150
            )
            
            cat_model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                verbose=False
            )
            
            cat_pred = cat_model.predict_proba(X_val_fold)[:, 1]
            test_preds['cat'][val_idx] = cat_pred
            
            # === Logistic Regression ===
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            lr_model = LogisticRegression(
                C=0.3,
                max_iter=1000,
                random_state=42 + fold,
                n_jobs=-1,
                solver='saga'
            )
            
            lr_model.fit(X_train_scaled, y_train_fold)
            lr_pred = lr_model.predict_proba(X_val_scaled)[:, 1]
            test_preds['lr'][val_idx] = lr_pred
            
            # Fold集成
            ensemble_pred = (lgb_pred + xgb_pred + cat_pred + lr_pred) / 4
            fold_auc = roc_auc_score(y_val_fold, ensemble_pred)
            fold_scores.append(fold_auc)
            
            print(f"  Fold {fold+1} AUC: {fold_auc:.6f}")
        
        # 计算各模型OOF分数
        model_scores = {}
        for name, preds in test_preds.items():
            model_scores[name] = roc_auc_score(y_train, preds)
            print(f"  {name.upper()} OOF AUC: {model_scores[name]:.6f}")
        
        # 加权集成
        total_score = sum(model_scores.values())
        weights = {name: score/total_score for name, score in model_scores.items()}
        
        oof_ensemble = sum(weights[name] * test_preds[name] for name in test_preds.keys())
        oof_auc = roc_auc_score(y_train, oof_ensemble)
        
        print(f"\n冠军集成 OOF AUC: {oof_auc:.8f}")
        
        self.model_scores = model_scores
        self.weights = weights
        self.is_trained = True
        
        return oof_auc, oof_ensemble
    
    def predict_champion(self, train_df, test_df):
        """冠军预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练!")
        
        # 准备测试特征
        _, X_test = self.prepare_champion_features(train_df, test_df)
        
        # 为每个模型训练最终版本（在全量训练集上）
        print("\n训练最终模型进行预测...")
        
        X_train_full, _ = self.prepare_champion_features(train_df, train_df)
        y_train_full = train_df['is_cheating'].values
        
        # LightGBM最终模型
        lgb_final = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=7,
            num_leaves=63,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        lgb_final.fit(X_train_full, y_train_full)
        lgb_test_pred = lgb_final.predict_proba(X_test)[:, 1]
        
        # XGBoost最终模型
        xgb_final = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='auc',
            tree_method='hist',
            n_jobs=-1
        )
        xgb_final.fit(X_train_full, y_train_full)
        xgb_test_pred = xgb_final.predict_proba(X_test)[:, 1]
        
        # CatBoost最终模型
        cat_final = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0
        )
        cat_final.fit(X_train_full, y_train_full)
        cat_test_pred = cat_final.predict_proba(X_test)[:, 1]
        
        # Logistic Regression最终模型
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test)
        
        lr_final = LogisticRegression(
            C=0.3,
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            solver='saga'
        )
        lr_final.fit(X_train_scaled, y_train_full)
        lr_test_pred = lr_final.predict_proba(X_test_scaled)[:, 1]
        
        # 使用OOF计算的权重进行集成
        test_ensemble = (
            lgb_test_pred * self.weights['lgb'] +
            xgb_test_pred * self.weights['xgb'] + 
            cat_test_pred * self.weights['cat'] +
            lr_test_pred * self.weights['lr']
        )
        
        return test_ensemble, {
            'lgb': lgb_test_pred,
            'xgb': xgb_test_pred,
            'cat': cat_test_pred,
            'lr': lr_test_pred
        }

def champion_main():
    """冠军主函数"""
    print("=" * 60)
    print("冠军版 - Mercor AI文本检测")
    print("=" * 60)
    
    try:
        # 加载数据
        train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
        test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
        
        print(f"训练集大小: {train_df.shape}")
        print(f"测试集大小: {test_df.shape}")
        print(f"作弊比例: {train_df['is_cheating'].mean():.6f}")
        
        # 处理缺失值
        train_df['answer'] = train_df['answer'].fillna('')
        test_df['answer'] = test_df['answer'].fillna('')
        
        # 初始化冠军检测器
        detector = ChampionAIDetector()
        
        # 训练模型
        print("\n开始训练冠军模型...")
        oof_auc, oof_predictions = detector.train_champion_ensemble(train_df, n_folds=10)
        
        # 预测测试集
        print("\n预测测试集...")
        champion_proba, individual_preds = detector.predict_champion(train_df, test_df)
        
        # 应用校准
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(oof_predictions, train_df['is_cheating'].values)
        calibrated_proba = calibrator.transform(champion_proba)
        
        # 确保合理范围
        calibrated_proba = np.clip(calibrated_proba, 0.001, 0.999)
        
        # 创建提交文件
        submission_champion = pd.DataFrame({
            'id': test_df['id'],
            'is_cheating': calibrated_proba
        })
        
        submission_base = pd.DataFrame({
            'id': test_df['id'],
            'is_cheating': champion_proba
        })
        
        # 保存文件（确保高精度）
        submission_champion.to_csv('champion_calibrated.csv', index=False, float_format='%.10f')
        submission_base.to_csv('champion_base.csv', index=False, float_format='%.10f')
        
        print(f"\n提交文件已保存:")
        print(f"  champion_calibrated.csv - 冠军校准版本 (推荐)")
        print(f"  champion_base.csv - 冠军基础版本")
        
        # 详细分析
        print(f"\n冠军版本预测统计:")
        print(f"  最小值: {calibrated_proba.min():.6f}")
        print(f"  最大值: {calibrated_proba.max():.6f}")
        print(f"  平均值: {calibrated_proba.mean():.6f}")
        print(f"  中位数: {np.median(calibrated_proba):.6f}")
        
        print(f"\n前10个样本预测:")
        for i in range(min(10, len(test_df))):
            print(f"  样本{i}: {calibrated_proba[i]:.6f}")
        
        # 各模型贡献分析
        print(f"\n各模型权重:")
        for name, weight in detector.weights.items():
            print(f"  {name.upper()}: {weight:.4f} (OOF AUC: {detector.model_scores[name]:.6f})")
        
        return submission_champion, oof_auc
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return champion_fallback()

def champion_fallback():
    """冠军备用方案"""
    print("使用冠军备用方案...")
    
    train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
    test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
    
    # 使用冠军特征提取
    feature_extractor = ChampionFeatureExtractor()
    train_features = feature_extractor.extract_champion_features(train_df)
    test_features = feature_extractor.extract_champion_features(test_df)
    
    # 对齐列
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
    
    # 简单TF-IDF
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), stop_words='english')
    train_tfidf = tfidf.fit_transform(train_df['answer'])
    test_tfidf = tfidf.transform(test_df['answer'])
    
    # 组合特征
    X_train = np.hstack([train_tfidf.toarray(), train_features.values])
    X_test = np.hstack([test_tfidf.toarray(), test_features.values])
    y_train = train_df['is_cheating']
    
    # 训练LightGBM（最稳定的模型）
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        num_leaves=63,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'is_cheating': test_proba
    })
    
    submission.to_csv('champion_fallback.csv', index=False, float_format='%.10f')
    
    # 计算训练集AUC
    train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"训练集AUC: {train_auc:.6f}")
    print(f"前5个测试样本预测: {test_proba[:5]}")
    
    return submission, train_auc

# 运行冠军版本
if __name__ == "__main__":
    try:
        submission, oof_auc = champion_main()
        
        print("\n" + "=" * 60)
        print("冠军版训练完成!")
        print(f"OOF AUC: {oof_auc:.8f}")
        print("\n提交策略:")
        print("1. 主要提交: champion_calibrated.csv (冠军校准版本)")
        print("2. 备选: champion_base.csv (冠军基础版本)")
        print("3. 期望目标: 超越0.978")
        print("=" * 60)
        
    except Exception as e:
        print(f"主流程失败: {e}")
        print("使用冠军备用方案...")
        submission, auc = champion_fallback()
        print(f"备用方案完成，训练AUC: {auc:.6f}")