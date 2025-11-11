
"""
ML Model Training Pipeline
Trains multiple model types on synthetic HLD dataset with proper ML practices
"""


import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor  # type: ignore

logger = logging.getLogger(__name__)

class LargeScaleMLTrainer:
	def __init__(self):
		self.models = {
			'RandomForest': RandomForestRegressor(random_state=42),
			'GradientBoosting': GradientBoostingRegressor(random_state=42),
			'XGBoost': XGBRegressor(random_state=42)
		}
		self.results = {}
		self.df = None
		self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
		self.X_train_scaled, self.X_test_scaled = None, None
		self.scaler = StandardScaler()
		self.selected_features = None
		self.feature_importance_dict = {}
		self.correlation_matrix = None
		self.eda_stats = {}

	def load_dataset(self, filepath: str):
		"""Load dataset from CSV file"""
		logger.info("=" * 80)
		logger.info("LOADING DATASET")
		logger.info("=" * 80)
		logger.info(f"📂 Dataset path: {filepath}")
		
		if not os.path.exists(filepath):
			logger.error(f"❌ Dataset file not found: {filepath}")
			raise FileNotFoundError(f"Dataset not found: {filepath}")
		
		logger.info(f"📊 Reading CSV file...")
		self.df = pd.read_csv(filepath)
		logger.info(f"✅ Dataset loaded successfully")
		logger.info(f"   Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
		logger.info(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
		logger.info("")
		
		# Validate dataset structure
		if self.df.shape[1] < 2:
			logger.error(f"❌ Dataset must have at least 2 columns (features + target)")
			raise ValueError("Dataset must have features and target.")
		
		if 'quality_score' not in self.df.columns:
			logger.error(f"❌ Dataset missing 'quality_score' target column")
			raise ValueError("Dataset must contain 'quality_score' column")
		
		logger.info(f"✅ Dataset validation passed")
		logger.info(f"   Feature columns: {self.df.shape[1] - 1}")
		logger.info(f"   Target column: quality_score")
		logger.info("")
		
		# Perform EDA
		self.perform_eda()

	def perform_eda(self):
		"""Exploratory Data Analysis on the dataset"""
		logger.info("=" * 80)
		logger.info("PERFORMING EXPLORATORY DATA ANALYSIS (EDA)")
		logger.info("=" * 80)
		
		# Basic statistics
		logger.info("📊 Computing basic statistics...")
		self.eda_stats = {
			'shape': self.df.shape,
			'missing_values': self.df.isnull().sum().to_dict(),
			'duplicates': int(self.df.duplicated().sum()),
			'descriptive_stats': self.df.describe().to_dict()
		}
		
		missing_total = sum(self.eda_stats['missing_values'].values())
		logger.info(f"   Missing values: {missing_total} ({missing_total / (self.df.shape[0] * self.df.shape[1]) * 100:.2f}%)")
		logger.info(f"   Duplicate rows: {self.eda_stats['duplicates']}")
		
		# Target variable distribution
		logger.info("")
		logger.info("🎯 Analyzing target variable (quality_score)...")
		target = self.df['quality_score']
		self.eda_stats['target_stats'] = {
			'mean': float(target.mean()),
			'median': float(target.median()),
			'std': float(target.std()),
			'min': float(target.min()),
			'max': float(target.max()),
			'q25': float(target.quantile(0.25)),
			'q75': float(target.quantile(0.75))
		}
		
		logger.info(f"   Mean: {self.eda_stats['target_stats']['mean']:.2f}")
		logger.info(f"   Median: {self.eda_stats['target_stats']['median']:.2f}")
		logger.info(f"   Std Dev: {self.eda_stats['target_stats']['std']:.2f}")
		logger.info(f"   Range: [{self.eda_stats['target_stats']['min']:.2f}, {self.eda_stats['target_stats']['max']:.2f}]")
		logger.info(f"   IQR: [{self.eda_stats['target_stats']['q25']:.2f}, {self.eda_stats['target_stats']['q75']:.2f}]")
		
		# Feature correlation with target
		logger.info("")
		logger.info("🔗 Computing feature correlations with target...")
		features = [col for col in self.df.columns if col != 'quality_score']
		correlations = {}
		for feature in features:
			corr = self.df[feature].corr(self.df['quality_score'])
			correlations[feature] = float(corr)
		
		self.eda_stats['feature_correlations'] = dict(sorted(
			correlations.items(), 
			key=lambda x: abs(x[1]), 
			reverse=True
		))
		
		# Log top 10 correlations
		logger.info("   Top 10 features by correlation with target:")
		top_10_corr = list(self.eda_stats['feature_correlations'].items())[:10]
		for feat_name, corr_value in top_10_corr:
			logger.info(f"     - {feat_name}: {corr_value:.4f}")
		
		# Correlation matrix (top features)
		logger.info("")
		logger.info("📈 Building correlation matrix for top 15 features...")
		# Get top 15 features by absolute correlation
		sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
		top_features = [feat_name for feat_name, _ in sorted_features[:15]]
		
		self.correlation_matrix = self.df[top_features + ['quality_score']].corr()
		logger.info(f"   Correlation matrix shape: {self.correlation_matrix.shape}")
		
		# Outlier detection (IQR method)
		logger.info("")
		logger.info("🔍 Detecting outliers (IQR method) in top 10 features...")
		outliers = {}
		for feature in features[:10]:  # Check top 10 features
			Q1 = self.df[feature].quantile(0.25)
			Q3 = self.df[feature].quantile(0.75)
			IQR = Q3 - Q1
			outlier_count = ((self.df[feature] < (Q1 - 1.5 * IQR)) | 
			                 (self.df[feature] > (Q3 + 1.5 * IQR))).sum()
			outliers[feature] = int(outlier_count)
			if outlier_count > 0:
				logger.info(f"   - {feature}: {outlier_count} outliers ({outlier_count / len(self.df) * 100:.2f}%)")
		
		self.eda_stats['outliers'] = outliers
		
		logger.info("")
		logger.info("✅ EDA COMPLETE")
		logger.info("=" * 80)
		logger.info("")

	def prepare_data(self, use_feature_selection=True, use_scaling=True):
		"""Prepare data for training with feature selection and scaling"""
		logger.info("=" * 80)
		logger.info("PREPARING DATA FOR TRAINING")
		logger.info("=" * 80)
		
		logger.info("🔀 Splitting data into train/test sets (80/20 split)...")
		features = [col for col in self.df.columns if col != 'quality_score']
		X = self.df[features]
		y = self.df['quality_score']
		
		logger.info(f"   Total samples: {len(X)}")
		logger.info(f"   Total features: {len(features)}")
		
		# Split data
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			X, y, test_size=0.2, random_state=42
		)
		logger.info(f"   ✅ Train set: {self.X_train.shape[0]} samples")
		logger.info(f"   ✅ Test set: {self.X_test.shape[0]} samples")
		logger.info("")
		
		# Feature Selection
		if use_feature_selection:
			logger.info("🎯 Performing feature selection (SelectKBest)...")
			k_features = min(25, len(features))
			logger.info(f"   Selecting top {k_features} features out of {len(features)}")
			
			selector = SelectKBest(score_func=f_regression, k=k_features)
			
			logger.info(f"   Fitting selector on training data...")
			X_train_selected = selector.fit_transform(self.X_train, self.y_train)
			selected_mask = selector.get_support()
			selected_columns = self.X_train.columns[selected_mask]
			
			self.X_train = pd.DataFrame(
				X_train_selected,
				columns=selected_columns,
				index=self.X_train.index
			)
			self.X_test = pd.DataFrame(
				selector.transform(self.X_test),
				columns=selected_columns,
				index=self.X_test.index
			)
			self.selected_features = list(self.X_train.columns)
			
			logger.info(f"   ✅ Selected {len(self.selected_features)} features")
			logger.info(f"   Top 10 selected features: {self.selected_features[:10]}")
			
			# Log feature scores
			feature_scores = dict(zip(features, selector.scores_))
			sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
			logger.info(f"   Top 5 features by F-score:")
			for feat_name, score in sorted_scores[:5]:
				logger.info(f"     - {feat_name}: {score:.2f}")
			logger.info("")
		else:
			self.selected_features = features
			logger.info("ℹ️  Feature selection disabled - using all features")
			logger.info("")
		
		# Feature Scaling
		if use_scaling:
			logger.info("📏 Scaling features (StandardScaler)...")
			logger.info(f"   Fitting scaler on training data...")
			
			X_train_scaled = self.scaler.fit_transform(self.X_train)
			X_test_scaled = self.scaler.transform(self.X_test)
			
			self.X_train_scaled = pd.DataFrame(
				X_train_scaled,
				columns=self.X_train.columns,
				index=self.X_train.index
			)
			self.X_test_scaled = pd.DataFrame(
				X_test_scaled,
				columns=self.X_test.columns,
				index=self.X_test.index
			)
			
			logger.info(f"   ✅ Features scaled")
			logger.info(f"   Scaler mean: {self.scaler.mean_[:3]}... (showing first 3)")
			logger.info(f"   Scaler std: {self.scaler.scale_[:3]}... (showing first 3)")
			logger.info("")
		else:
			self.X_train_scaled = self.X_train.copy()
			self.X_test_scaled = self.X_test.copy()
			logger.info("ℹ️  Feature scaling disabled")
			logger.info("")
		
		logger.info("✅ DATA PREPARATION COMPLETE")
		logger.info(f"   Final training shape: {self.X_train_scaled.shape}")
		logger.info(f"   Final test shape: {self.X_test_scaled.shape}")
		logger.info("=" * 80)
		logger.info("")

	def train_models(self, use_cross_validation=True):
		"""Train all models with optional cross-validation"""
		logger.info("=" * 80)
		logger.info("STARTING MODEL TRAINING")
		logger.info("=" * 80)
		logger.info(f"Number of models to train: {len(self.models)}")
		logger.info(f"Training data shape: {self.X_train_scaled.shape}")
		logger.info(f"Cross-validation: {'Enabled (5-fold)' if use_cross_validation else 'Disabled'}")
		logger.info("")
		
		# Convert models dict to list to avoid modification during iteration
		model_items = list(self.models.items())
		
		for idx, (name, model) in enumerate(model_items, 1):
			logger.info(f"[{idx}/{len(model_items)}] Training {name} model...")
			logger.info(f"  Model type: {type(model).__name__}")
			logger.info(f"  Training samples: {self.X_train_scaled.shape[0]}")
			logger.info(f"  Features: {self.X_train_scaled.shape[1]}")
			
			try:
				# Train on scaled data
				logger.info(f"  Fitting {name} model on training data...")
				model.fit(self.X_train_scaled, self.y_train)
				self.models[name] = model
				logger.info(f"  ✅ {name} model fitted successfully")
				
				# Cross-validation score
				if use_cross_validation:
					logger.info(f"  Running 5-fold cross-validation for {name}...")
					cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
					cv_mean = cv_scores.mean()
					cv_std = cv_scores.std()
					logger.info(f"  📊 {name} CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
					logger.info(f"     Individual fold scores: {[f'{s:.4f}' for s in cv_scores]}")
				
				# Feature importance
				if hasattr(model, 'feature_importances_'):
					logger.info(f"  Extracting feature importances for {name}...")
					importances = dict(zip(self.X_train_scaled.columns, model.feature_importances_))
					sorted_importances = dict(sorted(
						importances.items(), 
						key=lambda x: x[1], 
						reverse=True
					))
					self.feature_importance_dict[name] = dict(list(sorted_importances.items())[:15])  # Top 15
					
					# Log top 5 features
					top_5 = list(sorted_importances.items())[:5]
					logger.info(f"  🎯 Top 5 important features for {name}:")
					for feat_name, importance in top_5:
						logger.info(f"     - {feat_name}: {importance:.4f}")
				else:
					logger.info(f"  ℹ️  {name} does not provide feature importances")
				
				logger.info(f"  ✅ {name} model training complete")
				logger.info("")
				
			except Exception as e:
				logger.error(f"  ❌ Error training {name} model: {str(e)}")
				logger.exception(e)
				raise
		
		logger.info("=" * 80)
		logger.info("✅ ALL MODELS TRAINED SUCCESSFULLY")
		logger.info("=" * 80)
		logger.info("")

	def tune_hyperparameters(self, model_name='RandomForest', param_grid=None):
		"""Hyperparameter tuning using GridSearchCV"""
		logger.info(f"Tuning hyperparameters for {model_name}...")
		
		if param_grid is None:
			# Default parameter grids
			param_grids = {
				'RandomForest': {
					'n_estimators': [100, 200],
					'max_depth': [10, 20, None],
					'min_samples_split': [2, 5]
				},
				'GradientBoosting': {
					'n_estimators': [100, 200],
					'learning_rate': [0.01, 0.1],
					'max_depth': [3, 5]
				},
				'XGBoost': {
					'n_estimators': [100, 200],
					'learning_rate': [0.01, 0.1],
					'max_depth': [3, 5]
				}
			}
			param_grid = param_grids.get(model_name, {})
		
		if not param_grid:
			logger.warning(f"No parameter grid provided for {model_name}")
			return
		
		model = self.models[model_name]
		grid_search = GridSearchCV(
			model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
		)
		grid_search.fit(self.X_train_scaled, self.y_train)
		
		logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
		logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
		
		# Update model with best parameters
		self.models[model_name] = grid_search.best_estimator_
		
		return grid_search.best_params_, grid_search.best_score_

	def evaluate_models(self):
		"""Evaluate all trained models on test set"""
		logger.info("=" * 80)
		logger.info("EVALUATING MODELS")
		logger.info("=" * 80)
		logger.info(f"Test set size: {self.X_test_scaled.shape[0]} samples")
		logger.info("")
		
		for idx, (name, model) in enumerate(self.models.items(), 1):
			logger.info(f"[{idx}/{len(self.models)}] Evaluating {name}...")
			
			try:
				# Make predictions
				y_pred = model.predict(self.X_test_scaled)
				
				# Calculate metrics
				r2 = r2_score(self.y_test, y_pred)
				mse = mean_squared_error(self.y_test, y_pred)
				rmse = float(np.sqrt(mse))
				mae = mean_absolute_error(self.y_test, y_pred)
				mape = np.mean(np.abs((self.y_test - y_pred) / np.maximum(np.abs(self.y_test), 1e-8))) * 100
				
				# Additional metrics
				residuals = self.y_test - y_pred
				residual_std = float(np.std(residuals))
				
				self.results[name] = {
					"R2": r2, 
					"RMSE": rmse, 
					"MAE": mae, 
					"MAPE": mape,
					"Residual_STD": residual_std
				}
				
				logger.info(f"   📊 Metrics for {name}:")
				logger.info(f"      R² Score: {r2:.4f}")
				logger.info(f"      RMSE: {rmse:.2f}")
				logger.info(f"      MAE: {mae:.2f}")
				logger.info(f"      MAPE: {mape:.2f}%")
				logger.info(f"      Residual Std: {residual_std:.2f}")
				logger.info("")
				
			except Exception as e:
				logger.error(f"   ❌ Error evaluating {name}: {str(e)}")
				logger.exception(e)
				raise
		
		# Find best model
		best_model = max(self.results.items(), key=lambda x: x[1]['R2'])
		logger.info("🏆 BEST MODEL PERFORMANCE")
		logger.info(f"   Model: {best_model[0]}")
		logger.info(f"   R² Score: {best_model[1]['R2']:.4f}")
		logger.info(f"   RMSE: {best_model[1]['RMSE']:.2f}")
		logger.info("")
		
		logger.info("✅ MODEL EVALUATION COMPLETE")
		logger.info("=" * 80)
		logger.info("")

	def save_models(self, out_dir: str):
		"""Save trained models, scaler, and metadata to disk"""
		logger.info("=" * 80)
		logger.info("SAVING MODELS AND ARTIFACTS")
		logger.info("=" * 80)
		logger.info(f"📁 Output directory: {out_dir}")
		
		os.makedirs(out_dir, exist_ok=True)
		logger.info(f"   ✅ Directory created/verified")
		logger.info("")
		
		# Save models
		logger.info("💾 Saving trained models...")
		for name, model in self.models.items():
			model_path = os.path.join(out_dir, f"{name}_model.pkl")
			with open(model_path, 'wb') as f:
				pickle.dump(model, f)
			file_size = os.path.getsize(model_path) / 1024
			logger.info(f"   ✅ {name}_model.pkl ({file_size:.2f} KB)")
		logger.info("")
		
		# Save scaler
		logger.info("📏 Saving feature scaler...")
		scaler_path = os.path.join(out_dir, "scaler.pkl")
		with open(scaler_path, 'wb') as f:
			pickle.dump(self.scaler, f)
		file_size = os.path.getsize(scaler_path) / 1024
		logger.info(f"   ✅ scaler.pkl ({file_size:.2f} KB)")
		logger.info("")
		
		# Save metadata
		logger.info("📋 Saving training metadata...")
		metadata = {
			'selected_features': self.selected_features,
			'feature_importance': self.feature_importance_dict,
			'eda_stats': self.eda_stats,
			'model_results': self.results
		}
		metadata_path = os.path.join(out_dir, "training_metadata.pkl")
		with open(metadata_path, 'wb') as f:
			pickle.dump(metadata, f)
		file_size = os.path.getsize(metadata_path) / 1024
		logger.info(f"   ✅ training_metadata.pkl ({file_size:.2f} KB)")
		logger.info("")
		
		logger.info("✅ ALL ARTIFACTS SAVED SUCCESSFULLY")
		logger.info(f"   Total files: {len(self.models) + 2} (.pkl files)")
		logger.info(f"   Location: {out_dir}")
		logger.info("=" * 80)
		logger.info("")

	def get_feature_importance(self, model_name: str):
		model = self.models.get(model_name)
		if hasattr(model, 'feature_importances_'):
			return model.feature_importances_
		return None

	def cross_validation_score(self, model_name: str, cv=5):
		from sklearn.model_selection import cross_val_score
		model = self.models.get(model_name)
		scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv)
		return float(np.mean(scores))
	
	def get_eda_stats(self):
		"""Return EDA statistics for visualization"""
		return self.eda_stats
	
	def get_correlation_matrix(self):
		"""Return correlation matrix for heatmap"""
		return self.correlation_matrix
	
	def get_feature_importance_summary(self):
		"""Return feature importance across all models"""
		return self.feature_importance_dict
	
	def get_model_comparison(self):
		"""Return comparative metrics for all models"""
		if not self.results:
			return None
		
		comparison = pd.DataFrame(self.results).T
		comparison = comparison.round(4)
		comparison = comparison.sort_values('R2', ascending=False)
		return comparison

