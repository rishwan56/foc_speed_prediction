import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    # 1. Load the dataset
    dataset_path = 'BLDC_KNN_Dataset.xlsx'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_excel(dataset_path)
    
    # Define features and target
    target_col = 'MotorSpeed_RPM'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Hyperparameter Tuning using GridSearchCV
    print("Tuning K-Nearest Neighbors Regressor hyperparameters...")
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn = grid_search.best_estimator_
    print(f"\nBest Parameters found:")
    print(grid_search.best_params_)
    
    # 5. Evaluate the model
    # Predict on train and test
    y_train_pred = best_knn.predict(X_train_scaled)
    y_test_pred = best_knn.predict(X_test_scaled)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)
    
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE METRICS")
    print("="*40)
    print(f"Training Set:")
    print(f"  R2 Score:                  {r2_train:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_train:.2f} RPM")
    print(f"  Root Mean Sq. Error (RMSE):{rmse_train:.2f} RPM")
    print(f"Test Set:")
    print(f"  R2 Score:                  {r2_test:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_test:.2f} RPM")
    print(f"  Root Mean Sq. Error (RMSE):{rmse_test:.2f} RPM")
    print("="*40)
    
    # 6. Save the trained model and scaler
    model_filename = 'knn_model.joblib'
    scaler_filename = 'scaler.joblib'
    
    joblib.dump(best_knn, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Saved model to {model_filename}")
    print(f"Saved scaler to {scaler_filename}")
    
    # 7. Visualization
    print("Generating performance plot...")
    plt.figure(figsize=(8, 6))
    
    # Plot perfect predictions reference line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Predictions (1:1)')
    
    # Plot predictions
    plt.scatter(y_test, y_test_pred, color='dodgerblue', alpha=0.7, edgecolors='k', label='Predicted vs Actual')
    
    plt.xlabel('Actual Motor Speed (RPM)')
    plt.ylabel('Predicted Motor Speed (RPM)')
    plt.title('KNN Fan Speed Prediction Model Evaluation (Test Set)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add metrics text box
    textstr = '\n'.join((
        f'Test R² = {r2_test:.4f}',
        f'Test MAE = {mae_test:.1f} RPM',
        f'Test RMSE = {rmse_test:.1f} RPM'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plot_filename = 'knn_performance.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved performance plot to {plot_filename}")

if __name__ == '__main__':
    main()
