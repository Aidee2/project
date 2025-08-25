
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Starting Project 2: End-to-End Machine Learning Project")
print("="*60)

# Step 1: Create Dataset
print("\n1. Creating Dataset...")
np.random.seed(42)
n_samples = 1000

# Create feature data
data = {
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft_living': np.random.normal(2000, 800, n_samples),
    'sqft_lot': np.random.normal(8000, 3000, n_samples),
    'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
    'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(3, 14, n_samples)
}

# Calculate realistic price with noise
price = (
    data['bedrooms'] * 15000 +
    data['bathrooms'] * 10000 +
    data['sqft_living'] * 100 +
    data['sqft_lot'] * 5 +
    data['floors'] * 20000 +
    data['waterfront'] * 100000 +
    data['condition'] * 10000 +
    data['grade'] * 15000 +
    np.random.normal(0, 50000, n_samples)
)

data['price'] = np.maximum(price, 50000)  # Minimum price floor
df = pd.DataFrame(data)

print("Dataset created successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Step 2: Exploratory Data Analysis
print("\n" + "="*60)
print("2. Exploratory Data Analysis")
print("="*60)

print("\nDataset Description:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.dtypes)

# Correlation analysis
print("\nCreating correlation heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Price distribution and relationships
print("Creating data visualization plots...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.scatter(df['sqft_living'], df['price'], alpha=0.6, color='blue')
plt.title('Price vs Living Area')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Price ($)')

plt.subplot(1, 3, 3)
plt.scatter(df['grade'], df['price'], alpha=0.6, color='green')
plt.title('Price vs Grade')
plt.xlabel('Grade')
plt.ylabel('Price ($)')

plt.tight_layout()
plt.show()

# Step 3: Data Preprocessing
print("\n" + "="*60)
print("3. Data Preprocessing")
print("="*60)

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

print(f"Features: {list(X.columns)}")
print(f"Target: price")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Target variable range: ${y.min():,.2f} - ${y.max():,.2f}")

# Step 4: Model Training and Evaluation
print("\n" + "="*60)
print("4. Model Training and Evaluation")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Linear Regression':
        # Use scaled data for Linear Regression
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Use original data for Random Forest
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Predictions': y_pred,
        'Model': model
    }
    
    print(f"Results for {name}:")
    print(f"  Mean Squared Error: ${mse:,.2f}")
    print(f"  Root Mean Squared Error: ${rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Accuracy: {r2*100:.2f}%")

# Feature importance for Random Forest
print("\n" + "="*60)
print("5. Feature Importance Analysis")
print("="*60)

rf_model = results['Random Forest']['Model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance (Random Forest):")
print(feature_importance)

# Step 5: Results Visualization
print("\n" + "="*60)
print("6. Results Visualization")
print("="*60)

plt.figure(figsize=(18, 6))

# Predictions vs Actual plots
for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test, result['Predictions'], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(f'{name}\nR² = {result["R2"]:.4f}')
    
    # Add perfect prediction line
    plt.grid(True, alpha=0.3)

# Feature importance plot
plt.subplot(1, 3, 3)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model comparison
print("\nModel Comparison:")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[model]['RMSE'] for model in results.keys()],
    'R² Score': [results[model]['R2'] for model in results.keys()]
})
print(comparison_df)

# Step 6: Save the Best Model
print("\n" + "="*60)
print("7. Model Saving and Deployment")
print("="*60)

# Select best model based on R² score
best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
best_model = results[best_model_name]['Model']

print(f"Best performing model: {best_model_name}")
print(f"Best R² Score: {results[best_model_name]['R2']:.4f}")

# Save the best model and scaler
joblib.dump(best_model, 'house_price_model.joblib')
joblib.dump(scaler, 'price_scaler.joblib')

print("Models saved successfully!")
print("- house_price_model.joblib")
print("- price_scaler.joblib")

# Step 7: Sample Predictions
print("\n" + "="*60)
print("8. Sample Predictions")
print("="*60)

# Create sample houses for prediction
sample_houses = pd.DataFrame({
    'bedrooms': [3, 4, 2, 5],
    'bathrooms': [2, 3, 1, 3],
    'sqft_living': [2000, 2500, 1200, 3000],
    'sqft_lot': [8000, 10000, 5000, 12000],
    'floors': [2, 2.5, 1, 2],
    'waterfront': [0, 1, 0, 0],
    'condition': [4, 5, 3, 4],
    'grade': [7, 9, 5, 8]
})

print("Sample house predictions:")
print("-" * 80)
print(f"{'House':<8}{'Bedrooms':<10}{'Bathrooms':<11}{'Sqft':<8}{'Predicted Price':<15}")
print("-" * 80)

for i, (idx, house) in enumerate(sample_houses.iterrows()):
    if best_model_name == 'Linear Regression':
        house_scaled = scaler.transform([house.values])
        predicted_price = best_model.predict(house_scaled)[0]
    else:
        predicted_price = best_model.predict([house.values])[0]
    
    print(f"House {i+1:<3}{house['bedrooms']:<10}{house['bathrooms']:<11}{house['sqft_living']:<8}${predicted_price:,.2f}")

# Step 8: Model Performance Summary
print("\n" + "="*60)
print("9. Final Performance Summary")
print("="*60)

print("Project 2: End-to-End Machine Learning - COMPLETED")
print(f"✅ Dataset created: {n_samples} samples with {len(X.columns)} features")
print(f"✅ Best model: {best_model_name}")
print(f"✅ Model accuracy: {results[best_model_name]['R2']*100:.2f}%")
print(f"✅ Average prediction error: ${results[best_model_name]['RMSE']:,.2f}")
print("✅ Models saved for future use")

print("\nFiles created:")
print("- house_price_model.joblib (trained model)")
print("- price_scaler.joblib (feature scaler)")

print("\nKey Skills Demonstrated:")
print("- Data generation and preprocessing")
print("- Exploratory data analysis")
print("- Feature engineering and scaling")
print("- Model training and evaluation")
print("- Model comparison and selection")
print("- Data visualization")
print("- Model persistence and deployment")

print("\n" + "="*60)
print("PROJECT 2 COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
