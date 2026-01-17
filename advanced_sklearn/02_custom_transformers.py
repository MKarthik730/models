"""
Advanced Sklearn: Custom Transformers
======================================
Learn to build your own sklearn-compatible transformers for domain-specific preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# ============================================================================
# PART 1: Basic Custom Transformer
# ============================================================================

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips outliers to specified quantiles
    
    This transformer learns the quantile bounds from training data
    and applies them to both training and test data
    """
    
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
    
    def fit(self, X, y=None):
        """Learn the quantile bounds from training data"""
        self.lower_bound_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bound_ = np.quantile(X, self.upper_quantile, axis=0)
        return self
    
    def transform(self, X):
        """Clip values to learned bounds"""
        X_clipped = np.clip(X, self.lower_bound_, self.upper_bound_)
        return X_clipped
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names (for sklearn >= 1.0)"""
        return input_features


def test_outlier_clipper():
    """Test the OutlierClipper transformer"""
    print("=" * 60)
    print("OUTLIER CLIPPER TRANSFORMER")
    print("=" * 60)
    
    # Create data with outliers
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[0, 0] = 10  # outlier
    X[1, 1] = -10  # outlier
    
    print("Original data range:")
    print(f"  Min: {X.min(axis=0)}")
    print(f"  Max: {X.max(axis=0)}")
    
    # Fit and transform
    clipper = OutlierClipper(lower_quantile=0.05, upper_quantile=0.95)
    X_clipped = clipper.fit_transform(X)
    
    print("\nClipped data range:")
    print(f"  Min: {X_clipped.min(axis=0)}")
    print(f"  Max: {X_clipped.max(axis=0)}")
    
    print("\nLearned bounds:")
    print(f"  Lower: {clipper.lower_bound_}")
    print(f"  Upper: {clipper.upper_bound_}")
    
    return clipper


# ============================================================================
# PART 2: Feature Engineering Transformer
# ============================================================================

class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Create polynomial and interaction features
    Simpler version of sklearn's PolynomialFeatures for educational purposes
    """
    
    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only
    
    def fit(self, X, y=None):
        """Store number of input features"""
        self.n_input_features_ = X.shape[1]
        return self
    
    def transform(self, X):
        """Create polynomial features"""
        n_samples, n_features = X.shape
        
        if self.degree == 1:
            return X
        
        # Start with original features
        features = [X]
        
        # Add squared terms (if not interaction_only)
        if not self.interaction_only:
            features.append(X ** 2)
        
        # Add interaction terms
        if n_features > 1:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        return np.hstack(features)
    
    def get_feature_names_out(self, input_features=None):
        """Generate feature names"""
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]
        
        feature_names = list(input_features)
        
        if not self.interaction_only:
            feature_names.extend([f"{name}^2" for name in input_features])
        
        n_features = len(input_features)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature_names.append(f"{input_features[i]}*{input_features[j]}")
        
        return np.array(feature_names)


def test_polynomial_features():
    """Test the PolynomialFeatures transformer"""
    print("\n" + "=" * 60)
    print("POLYNOMIAL FEATURES TRANSFORMER")
    print("=" * 60)
    
    # Create simple data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    print("Original data:")
    print(X)
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    print("\nPolynomial features:")
    print(X_poly)
    print(f"\nFeature names: {poly.get_feature_names_out(['a', 'b'])}")
    
    return poly


# ============================================================================
# PART 3: Log Transform Transformer
# ============================================================================

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log transformation to skewed features
    Handles zeros and negative values
    """
    
    def __init__(self, offset=1.0):
        """
        Parameters:
        -----------
        offset : float, default=1.0
            Value to add before taking log (to handle zeros)
        """
        self.offset = offset
    
    def fit(self, X, y=None):
        """Check for negative values"""
        if np.any(X < 0):
            raise ValueError("LogTransformer cannot handle negative values")
        return self
    
    def transform(self, X):
        """Apply log(X + offset)"""
        return np.log(X + self.offset)
    
    def inverse_transform(self, X):
        """Reverse the log transformation"""
        return np.exp(X) - self.offset


def test_log_transformer():
    """Test the LogTransformer"""
    print("\n" + "=" * 60)
    print("LOG TRANSFORMER")
    print("=" * 60)
    
    # Create right-skewed data
    np.random.seed(42)
    X = np.random.exponential(scale=2.0, size=(100, 1))
    
    print(f"Original data - Mean: {X.mean():.2f}, Std: {X.std():.2f}")
    print(f"Original data - Skewness: {pd.Series(X.flatten()).skew():.2f}")
    
    log_transformer = LogTransformer(offset=1.0)
    X_log = log_transformer.fit_transform(X)
    
    print(f"\nLog-transformed - Mean: {X_log.mean():.2f}, Std: {X_log.std():.2f}")
    print(f"Log-transformed - Skewness: {pd.Series(X_log.flatten()).skew():.2f}")
    
    # Test inverse transform
    X_inverse = log_transformer.inverse_transform(X_log)
    print(f"\nInverse transform error: {np.abs(X - X_inverse).max():.10f}")
    
    return log_transformer


# ============================================================================
# PART 4: DataFrame Selector (for Pandas DataFrames)
# ============================================================================

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from a pandas DataFrame
    Useful for creating pipelines that work with DataFrames
    """
    
    def __init__(self, columns):
        """
        Parameters:
        -----------
        columns : list of str
            Column names to select
        """
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Select and return specified columns"""
        return X[self.columns].values


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Combine multiple DataFrame transformers
    Alternative to FeatureUnion for DataFrames
    """
    
    def __init__(self, transformer_list):
        """
        Parameters:
        -----------
        transformer_list : list of (name, transformer) tuples
        """
        self.transformer_list = transformer_list
    
    def fit(self, X, y=None):
        for name, transformer in self.transformer_list:
            transformer.fit(X, y)
        return self
    
    def transform(self, X):
        """Apply all transformers and concatenate results"""
        features = []
        for name, transformer in self.transformer_list:
            features.append(transformer.transform(X))
        return np.hstack(features)


def test_dataframe_transformers():
    """Test DataFrame-specific transformers"""
    print("\n" + "=" * 60)
    print("DATAFRAME TRANSFORMERS")
    print("=" * 60)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'score': [85, 90, 78, 88, 92]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Select numeric columns
    numeric_selector = DataFrameSelector(['age', 'income', 'score'])
    X_numeric = numeric_selector.fit_transform(df)
    
    print("\nNumeric columns selected:")
    print(X_numeric)
    
    return numeric_selector


# ============================================================================
# PART 5: Binning Transformer
# ============================================================================

class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    Bin continuous features into discrete bins
    """
    
    def __init__(self, n_bins=5, strategy='quantile'):
        """
        Parameters:
        -----------
        n_bins : int, default=5
            Number of bins to create
        strategy : str, default='quantile'
            Strategy to define bin edges: 'uniform' or 'quantile'
        """
        self.n_bins = n_bins
        self.strategy = strategy
    
    def fit(self, X, y=None):
        """Learn bin edges from training data"""
        if self.strategy == 'uniform':
            self.bins_ = np.linspace(X.min(), X.max(), self.n_bins + 1)
        elif self.strategy == 'quantile':
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self.bins_ = np.percentile(X, quantiles)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self
    
    def transform(self, X):
        """Assign each value to a bin"""
        return np.digitize(X, self.bins_[1:-1])


def test_binning_transformer():
    """Test the BinningTransformer"""
    print("\n" + "=" * 60)
    print("BINNING TRANSFORMER")
    print("=" * 60)
    
    # Create continuous data
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10 + 50
    
    print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Uniform binning
    uniform_binner = BinningTransformer(n_bins=5, strategy='uniform')
    X_uniform = uniform_binner.fit_transform(X)
    print(f"\nUniform bins: {uniform_binner.bins_}")
    print(f"Bin counts: {np.bincount(X_uniform.flatten())}")
    
    # Quantile binning
    quantile_binner = BinningTransformer(n_bins=5, strategy='quantile')
    X_quantile = quantile_binner.fit_transform(X)
    print(f"\nQuantile bins: {quantile_binner.bins_}")
    print(f"Bin counts: {np.bincount(X_quantile.flatten())}")
    
    return uniform_binner


# ============================================================================
# PART 6: Conditional Transformer
# ============================================================================

class ConditionalTransformer(BaseEstimator, TransformerMixin):
    """
    Apply different transformations based on a condition
    Example: Scale only if variance is above threshold
    """
    
    def __init__(self, transformer, condition_func):
        """
        Parameters:
        -----------
        transformer : sklearn transformer
            Transformer to apply when condition is met
        condition_func : callable
            Function that takes X and returns boolean array
        """
        self.transformer = transformer
        self.condition_func = condition_func
    
    def fit(self, X, y=None):
        """Fit transformer only on columns meeting condition"""
        self.condition_mask_ = self.condition_func(X)
        if np.any(self.condition_mask_):
            self.transformer.fit(X[:, self.condition_mask_], y)
        return self
    
    def transform(self, X):
        """Apply transformer only to columns meeting condition"""
        X_transformed = X.copy()
        if np.any(self.condition_mask_):
            X_transformed[:, self.condition_mask_] = self.transformer.transform(
                X[:, self.condition_mask_]
            )
        return X_transformed


def test_conditional_transformer():
    """Test ConditionalTransformer"""
    print("\n" + "=" * 60)
    print("CONDITIONAL TRANSFORMER")
    print("=" * 60)
    
    from sklearn.preprocessing import StandardScaler
    
    # Create data with different variances
    np.random.seed(42)
    X = np.column_stack([
        np.random.randn(100) * 10,  # high variance
        np.random.randn(100) * 0.1,  # low variance
        np.random.randn(100) * 5     # medium variance
    ])
    
    print("Original standard deviations:")
    print(X.std(axis=0))
    
    # Only scale columns with std > 1
    condition = lambda X: X.std(axis=0) > 1
    conditional_scaler = ConditionalTransformer(StandardScaler(), condition)
    
    X_transformed = conditional_scaler.fit_transform(X)
    
    print("\nTransformed standard deviations:")
    print(X_transformed.std(axis=0))
    print(f"\nColumns scaled: {conditional_scaler.condition_mask_}")
    
    return conditional_scaler


# ============================================================================
# PART 7: Complete Pipeline with Custom Transformers
# ============================================================================

def complete_pipeline_example():
    """Build a complete pipeline using custom transformers"""
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE WITH CUSTOM TRANSFORMERS")
    print("=" * 60)
    
    # Create realistic dataset
    np.random.seed(42)
    n_samples = 500
    
    # Features with different characteristics
    age = np.random.randint(18, 80, n_samples)
    income = np.random.exponential(50000, n_samples) + 20000  # skewed
    credit_score = np.random.normal(700, 100, n_samples)
    credit_score[10:20] = 900  # some outliers
    
    # Create binary target
    y = ((age > 35) & (income > 60000) & (credit_score > 650)).astype(int)
    
    # Combine into dataset
    X = np.column_stack([age, income, credit_score])
    feature_names = ['age', 'income', 'credit_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build pipeline with custom transformers
    pipeline = Pipeline([
        ('clip_outliers', OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
        ('log_income', FunctionTransformer(
            lambda X: np.column_stack([X[:, 0], np.log(X[:, 1]), X[:, 2]])
        )),
        ('polynomial', PolynomialFeatures(degree=2, interaction_only=True)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"\nCV scores: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return pipeline


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED SKLEARN: CUSTOM TRANSFORMERS")
    print("="*60 + "\n")
    
    # Run all examples
    test_outlier_clipper()
    test_polynomial_features()
    test_log_transformer()
    test_dataframe_transformers()
    test_binning_transformer()
    test_conditional_transformer()
    complete_pipeline_example()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED!")
    print("="*60)
    
    print("\nKEY TAKEAWAYS:")
    print("1. Custom transformers must inherit from BaseEstimator and TransformerMixin")
    print("2. Implement fit() to learn from training data, transform() to apply")
    print("3. Always return self from fit() for method chaining")
    print("4. Custom transformers integrate seamlessly with pipelines")
    print("5. Use get_feature_names_out() for sklearn >= 1.0 compatibility")
    print("6. Custom transformers enable domain-specific preprocessing")
