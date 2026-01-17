"""
Advanced Sklearn: Pipelines and ColumnTransformer
==================================================
Master creating reproducible, production-ready ML workflows
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================================
# PART 1: Basic Pipeline
# ============================================================================

def basic_pipeline_example():
    """Simple pipeline without ColumnTransformer"""
    print("=" * 60)
    print("BASIC PIPELINE EXAMPLE")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    
    # Fit and score
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    print(f"Pipeline accuracy: {score:.3f}")
    
    # Access individual steps
    print(f"\nScaler mean: {pipeline.named_steps['scaler'].mean_[:3]}")
    print(f"Classifier coef shape: {pipeline.named_steps['classifier'].coef_.shape}")
    
    return pipeline


# ============================================================================
# PART 2: ColumnTransformer - Handle Different Column Types
# ============================================================================

def column_transformer_example():
    """Handle numeric and categorical columns differently"""
    print("\n" + "=" * 60)
    print("COLUMN TRANSFORMER EXAMPLE")
    print("=" * 60)
    
    # Create mixed-type dataset
    data = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 35],
        'salary': [50000, 60000, 55000, np.nan, 70000],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'education': ['BS', 'MS', 'PhD', 'BS', 'MS'],
        'purchased': [0, 1, 1, 0, 1]
    })
    
    X = data.drop('purchased', axis=1)
    y = data['purchased']
    
    print("Original data:")
    print(X.head())
    
    # Define column types
    numeric_features = ['age', 'salary']
    categorical_features = ['city', 'education']
    
    # Create transformers for each type
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # drop any columns not specified
    )
    
    # Create full pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # Fit the model
    model.fit(X, y)
    
    # Transform data to see what it looks like
    X_transformed = preprocessor.fit_transform(X)
    print(f"\nTransformed shape: {X_transformed.shape}")
    print(f"Original shape: {X.shape}")
    print("\nFirst row transformed:")
    print(X_transformed[0])
    
    # Get feature names after transformation
    feature_names = (numeric_features + 
                    preprocessor.named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names_out(categorical_features).tolist())
    print(f"\nFeature names after transformation: {feature_names}")
    
    return model


# ============================================================================
# PART 3: Advanced ColumnTransformer Patterns
# ============================================================================

def advanced_column_transformer():
    """Advanced patterns with ColumnTransformer"""
    print("\n" + "=" * 60)
    print("ADVANCED COLUMN TRANSFORMER PATTERNS")
    print("=" * 60)
    
    # Create complex dataset
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, 200),
        'income': np.random.randint(30000, 150000, 200),
        'credit_score': np.random.randint(300, 850, 200),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston'], 200),
        'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 200),
        'owns_home': np.random.choice([0, 1], 200),
        'target': np.random.choice([0, 1], 200)
    })
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Pattern 1: Different scalers for different numeric columns
    preprocessor1 = ColumnTransformer([
        ('standard', StandardScaler(), ['age', 'credit_score']),
        ('minmax', MinMaxScaler(), ['income']),
        ('cat', OneHotEncoder(sparse_output=False), ['city', 'education']),
        ('binary', 'passthrough', ['owns_home'])  # leave as-is
    ])
    
    # Pattern 2: Using make_column_transformer (cleaner syntax)
    preprocessor2 = make_column_transformer(
        (StandardScaler(), ['age', 'credit_score']),
        (MinMaxScaler(), ['income']),
        (OneHotEncoder(sparse_output=False), ['city', 'education']),
        remainder='passthrough'  # keep owns_home
    )
    
    # Pattern 3: Using column selectors
    from sklearn.compose import make_column_selector
    
    preprocessor3 = ColumnTransformer([
        ('num', StandardScaler(), 
         make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(sparse_output=False), 
         make_column_selector(dtype_include=object))
    ])
    
    print("Testing different preprocessor patterns...")
    for i, preprocessor in enumerate([preprocessor1, preprocessor2, preprocessor3], 1):
        X_transformed = preprocessor.fit_transform(X)
        print(f"Preprocessor {i} output shape: {X_transformed.shape}")
    
    return preprocessor1


# ============================================================================
# PART 4: Nested Pipelines
# ============================================================================

def nested_pipelines_example():
    """Create complex nested pipeline structures"""
    print("\n" + "=" * 60)
    print("NESTED PIPELINES")
    print("=" * 60)
    
    # Create dataset
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, 300),
        'income': np.random.randint(30000, 150000, 300),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 300),
        'job_type': np.random.choice(['Tech', 'Finance', 'Healthcare'], 300),
        'target': np.random.choice([0, 1], 300)
    })
    
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create nested preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into preprocessor
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, ['age', 'income']),
        ('categorical', categorical_pipeline, ['city', 'job_type'])
    ])
    
    # Full model pipeline with multiple models
    model_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train and evaluate
    model_pipeline.fit(X_train, y_train)
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    # Cross-validation with the full pipeline
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return model_pipeline


# ============================================================================
# PART 5: Pipeline with Feature Selection
# ============================================================================

def pipeline_with_feature_selection():
    """Add feature selection to pipeline"""
    print("\n" + "=" * 60)
    print("PIPELINE WITH FEATURE SELECTION")
    print("=" * 60)
    
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    
    # Create dataset with many features
    np.random.seed(42)
    X = np.random.randn(200, 20)
    # Make first 5 features actually predictive
    y = (X[:, 0] + X[:, 1] - X[:, 2] + X[:, 3] - X[:, 4] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline with feature selection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=10)),
        ('classifier', LogisticRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    
    print(f"Training accuracy: {pipeline.score(X_train, y_train):.3f}")
    print(f"Testing accuracy: {pipeline.score(X_test, y_test):.3f}")
    
    # See which features were selected
    selected_features = pipeline.named_steps['feature_selection'].get_support()
    print(f"\nSelected features: {np.where(selected_features)[0]}")
    print(f"Feature scores: {pipeline.named_steps['feature_selection'].scores_}")
    
    # Alternative: RFE (Recursive Feature Elimination)
    pipeline_rfe = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', RFE(LogisticRegression(), n_features_to_select=10)),
        ('classifier', LogisticRegression())
    ])
    
    pipeline_rfe.fit(X_train, y_train)
    print(f"\nRFE Testing accuracy: {pipeline_rfe.score(X_test, y_test):.3f}")
    
    return pipeline


# ============================================================================
# PART 6: Accessing Pipeline Components
# ============================================================================

def accessing_pipeline_components():
    """Learn to access and inspect pipeline components"""
    print("\n" + "=" * 60)
    print("ACCESSING PIPELINE COMPONENTS")
    print("=" * 60)
    
    # Create a complex pipeline
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['feature1', 'feature2']),
        ('cat', OneHotEncoder(sparse_output=False), ['category'])
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    # Access methods
    print("1. Access by name:")
    print(f"   Classifier: {pipeline.named_steps['classifier']}")
    
    print("\n2. Access by index:")
    print(f"   Preprocessor: {pipeline[0]}")
    
    print("\n3. Access nested components:")
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
    print(f"   Scaler mean: {scaler.mean_}")
    
    print("\n4. Get all step names:")
    print(f"   Steps: {[name for name, _ in pipeline.steps]}")
    
    print("\n5. Access classifier features:")
    print(f"   Feature importances shape: {pipeline.named_steps['classifier'].feature_importances_.shape}")
    
    print("\n6. Transform only (no prediction):")
    X_transformed = pipeline[:-1].transform(X)
    print(f"   Transformed shape: {X_transformed.shape}")
    
    return pipeline


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED SKLEARN: PIPELINES AND COLUMNTRANSFORMER")
    print("="*60 + "\n")
    
    # Run all examples
    basic_pipeline_example()
    column_transformer_example()
    advanced_column_transformer()
    nested_pipelines_example()
    pipeline_with_feature_selection()
    accessing_pipeline_components()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED!")
    print("="*60)
    
    print("\nKEY TAKEAWAYS:")
    print("1. Pipelines prevent data leakage by ensuring transform fit on train only")
    print("2. ColumnTransformer handles different column types elegantly")
    print("3. Nested pipelines allow complex preprocessing workflows")
    print("4. Pipelines make models reproducible and production-ready")
    print("5. Access pipeline components with named_steps or indexing")
