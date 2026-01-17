"""
Advanced Sklearn: Hyperparameter Tuning
========================================
Master GridSearchCV, RandomizedSearchCV, and advanced tuning strategies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, train_test_split,
    cross_val_score, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from scipy.stats import randint, uniform
import time

# ============================================================================
# PART 1: GridSearchCV Basics
# ============================================================================

def grid_search_basics():
    """Basic GridSearchCV example"""
    print("=" * 60)
    print("GRID SEARCH BASICS")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define model
    rf = RandomForestClassifier(random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print(f"Total combinations: {3 * 4 * 3 * 3} = 108")
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit
    print("\nFitting GridSearchCV...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\nTime taken: {elapsed_time:.2f} seconds")
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Test set score: {grid_search.score(X_test, y_test):.3f}")
    
    # Examine results
    results_df = pd.DataFrame(grid_search.cv_results_)
    print("\nTop 5 parameter combinations:")
    print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
          .sort_values('rank_test_score')
          .head())
    
    return grid_search


# ============================================================================
# PART 2: RandomizedSearchCV
# ============================================================================

def randomized_search_example():
    """RandomizedSearchCV for efficient hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("RANDOMIZED SEARCH")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define model
    rf = RandomForestClassifier(random_state=42)
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': randint(50, 500),  # sample integers from 50 to 499
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0.1, 0.9)  # sample floats from 0.1 to 1.0
    }
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=50,  # try 50 random combinations
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit
    print("\nFitting RandomizedSearchCV...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\nTime taken: {elapsed_time:.2f} seconds")
    print(f"Sampled {random_search.n_iter} combinations")
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.3f}")
    print(f"Test set score: {random_search.score(X_test, y_test):.3f}")
    
    return random_search


# ============================================================================
# PART 3: Grid Search vs Randomized Search Comparison
# ============================================================================

def compare_search_methods():
    """Compare GridSearchCV vs RandomizedSearchCV"""
    print("\n" + "=" * 60)
    print("GRID SEARCH VS RANDOMIZED SEARCH")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with smaller grid
    grid_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Random search with broader distributions
    random_params = {
        'n_estimators': randint(50, 300),
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': randint(2, 20)
    }
    
    # Grid Search
    grid = GridSearchCV(rf, grid_params, cv=3, n_jobs=-1)
    start = time.time()
    grid.fit(X_train, y_train)
    grid_time = time.time() - start
    
    # Random Search (same number of iterations as grid)
    n_grid_combinations = len(grid_params['n_estimators']) * \
                          len(grid_params['max_depth']) * \
                          len(grid_params['min_samples_split'])
    
    random = RandomizedSearchCV(
        rf, random_params, n_iter=n_grid_combinations,
        cv=3, n_jobs=-1, random_state=42
    )
    start = time.time()
    random.fit(X_train, y_train)
    random_time = time.time() - start
    
    print("\nResults comparison:")
    print(f"Grid Search:")
    print(f"  Time: {grid_time:.2f}s")
    print(f"  Best CV score: {grid.best_score_:.3f}")
    print(f"  Test score: {grid.score(X_test, y_test):.3f}")
    print(f"  Combinations tried: {n_grid_combinations}")
    
    print(f"\nRandomized Search:")
    print(f"  Time: {random_time:.2f}s")
    print(f"  Best CV score: {random.best_score_:.3f}")
    print(f"  Test score: {random.score(X_test, y_test):.3f}")
    print(f"  Combinations tried: {random.n_iter}")
    
    return grid, random


# ============================================================================
# PART 4: Hyperparameter Tuning with Pipelines
# ============================================================================

def pipeline_hyperparameter_tuning():
    """Tune hyperparameters across entire pipeline"""
    print("\n" + "=" * 60)
    print("PIPELINE HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid
    # Note: use 'step_name__parameter_name' syntax
    param_grid = {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5]
    }
    
    # Grid search on pipeline
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    print(f"Test score: {grid_search.score(X_test, y_test):.3f}")
    
    # Access best estimator
    best_pipeline = grid_search.best_estimator_
    print(f"\nBest scaler settings:")
    print(f"  with_mean: {best_pipeline.named_steps['scaler'].with_mean}")
    print(f"  with_std: {best_pipeline.named_steps['scaler'].with_std}")
    
    return grid_search


# ============================================================================
# PART 5: Multiple Scoring Metrics
# ============================================================================

def multiple_scoring_metrics():
    """Use multiple metrics for model evaluation"""
    print("\n" + "=" * 60)
    print("MULTIPLE SCORING METRICS")
    print("=" * 60)
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        weights=[0.9, 0.1],  # imbalanced
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Define model
    rf = RandomForestClassifier(random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'class_weight': [None, 'balanced']
    }
    
    # Define multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Grid search with multiple metrics
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring=scoring,
        refit='f1',  # refit using F1 score
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Results for different metrics
    results = pd.DataFrame(grid_search.cv_results_)
    
    print("\nBest parameters (optimized for F1):")
    print(grid_search.best_params_)
    
    print("\nScores for best model:")
    for metric in scoring.keys():
        score_key = f'mean_test_{metric}'
        best_idx = grid_search.best_index_
        print(f"  {metric}: {results.loc[best_idx, score_key]:.3f}")
    
    print("\nComparison of top models by different metrics:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_idx = results[f'mean_test_{metric}'].idxmax()
        print(f"\nBest by {metric}:")
        print(f"  Params: {results.loc[best_idx, 'params']}")
        print(f"  {metric}: {results.loc[best_idx, f'mean_test_{metric}']:.3f}")
    
    return grid_search


# ============================================================================
# PART 6: Custom Scoring Function
# ============================================================================

def custom_scoring_function():
    """Create and use custom scoring functions"""
    print("\n" + "=" * 60)
    print("CUSTOM SCORING FUNCTION")
    print("=" * 60)
    
    from sklearn.metrics import make_scorer, confusion_matrix
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000, n_features=20,
        weights=[0.7, 0.3], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define custom scorer: weighted cost function
    # False positives cost 1, false negatives cost 5
    def custom_cost(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * 1 + fn * 5
        return -cost  # negative because we want to minimize
    
    custom_scorer = make_scorer(custom_cost, greater_is_better=True)
    
    # Define model and parameters
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'class_weight': [None, 'balanced', {0: 1, 1: 5}]
    }
    
    # Grid search with custom scorer
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring=custom_scorer,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best custom score: {grid_search.best_score_:.3f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\nTest set confusion matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    print(f"  Total cost: {fp * 1 + fn * 5}")
    
    return grid_search


# ============================================================================
# PART 7: Nested Cross-Validation
# ============================================================================

def nested_cross_validation():
    """Proper evaluation using nested cross-validation"""
    print("\n" + "=" * 60)
    print("NESTED CROSS-VALIDATION")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=15,
        random_state=42
    )
    
    # Define model and parameters
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    # Inner CV for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Outer CV for model evaluation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create grid search object
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Nested cross-validation
    nested_scores = cross_val_score(
        grid_search,
        X, y,
        cv=outer_cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("Nested CV scores:", nested_scores)
    print(f"Mean: {nested_scores.mean():.3f}")
    print(f"Std: {nested_scores.std():.3f}")
    print(f"95% CI: [{nested_scores.mean() - 1.96*nested_scores.std():.3f}, "
          f"{nested_scores.mean() + 1.96*nested_scores.std():.3f}]")
    
    # Compare with non-nested CV (biased estimate)
    print("\nFor comparison, non-nested CV:")
    grid_search.fit(X, y)
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    print("(This is an optimistically biased estimate!)")
    
    return nested_scores


# ============================================================================
# PART 8: Advanced: Model Selection Across Different Algorithms
# ============================================================================

def model_selection_comparison():
    """Compare different algorithms with hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("MODEL SELECTION ACROSS ALGORITHMS")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models and their parameter spaces
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.5],
                'max_depth': [3, 5]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
    }
    
    # Compare models
    results = {}
    
    for name, config in models.items():
        print(f"\nTuning {name}...")
        
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        start = time.time()
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start
        
        results[name] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_score': grid_search.score(X_test, y_test),
            'time': elapsed
        }
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('test_score', ascending=False)
    print(results_df)
    
    print(f"\nBest model: {results_df.index[0]}")
    print(f"Best test score: {results_df.iloc[0]['test_score']:.3f}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED SKLEARN: HYPERPARAMETER TUNING")
    print("="*60 + "\n")
    
    # Run all examples
    grid_search_basics()
    randomized_search_example()
    compare_search_methods()
    pipeline_hyperparameter_tuning()
    multiple_scoring_metrics()
    custom_scoring_function()
    nested_cross_validation()
    model_selection_comparison()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED!")
    print("="*60)
    
    print("\nKEY TAKEAWAYS:")
    print("1. GridSearchCV: exhaustive but slow for large parameter spaces")
    print("2. RandomizedSearchCV: faster, good for continuous parameters")
    print("3. Use pipelines to tune preprocessing and model together")
    print("4. Multiple metrics: define scoring dict, specify refit metric")
    print("5. Custom scorers: use make_scorer for domain-specific metrics")
    print("6. Nested CV: unbiased performance estimate")
    print("7. Compare algorithms: test multiple models systematically")
