# ============================================================
# PROJECT 2: Data Classification Using AI
# DecodeLabs Industrial Training - Batch 2026
# Algorithm: K-Nearest Neighbors on Iris Dataset
# ============================================================


#STEP 1: IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             f1_score)


#STEP 2: LOAD & EXPLORE THE DATASET
print("=" * 50)
print("       LOADING THE IRIS DATASET")
print("=" * 50)

iris = load_iris()

# Convert to a readable table (DataFrame)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

print("\n--- First 5 rows of data ---")
print(df.head())

print("\n--- Dataset Shape (rows, columns) ---")
print(df.shape)

print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Flower Count per Species ---")
print(df['species_name'].value_counts())


#STEP 3: SEPARATE FEATURES AND LABELS
print("\n" + "=" * 50)
print("  SEPARATING INPUTS (X) AND ANSWERS (y)")
print("=" * 50)

X = iris.data    #INPUT
y = iris.target  #ANSWER

print(f"\nFeatures shape : {X.shape}")  
print(f"Labels shape   : {y.shape}")    
print(f"Class names    : {iris.target_names}")


#STEP 4: FEATURE SCALING
print("\n" + "=" * 50)
print("       FEATURE SCALING")
print("=" * 50)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Before Scaling (first row) ---")
print(X[0])

print("\n--- After Scaling (first row) ---")
print(X_scaled[0])


#STEP 5: TRAIN-TEST SPLIT
print("\n" + "=" * 50)
print("       TRAIN-TEST SPLIT (80/20)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,    
    random_state=42,  
    shuffle=True      
)

print(f"\nTraining samples : {len(X_train)}") 
print(f"Testing samples  : {len(X_test)}")    


#STEP 6: TRAIN THE KNN MODEL
print("\n" + "=" * 50)
print("       TRAINING THE KNN MODEL")
print("=" * 50)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n--- Predictions vs Actual ---")
print(f"Predicted : {predictions}")
print(f"Actual    : {y_test}")


#STEP 7: EVALUATE THE MODEL
print("\n" + "=" * 50)
print("       MODEL EVALUATION")
print("=" * 50)

print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))

f1 = f1_score(y_test, predictions, average='weighted')
print(f"✅ Weighted F1 Score : {f1:.2f}")


#STEP 8: CONFUSION MATRIX
print("\n" + "=" * 50)
print("       CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title('Confusion Matrix — KNN Iris Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as confusion_matrix.png ✅")
plt.show()


#FINDING THE BEST K
print("\n" + "=" * 50)
print("       BONUS: FINDING THE BEST K VALUE")
print("=" * 50)

error_rates = []
k_range = range(1, 21, 2)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(
        1 - f1_score(preds, y_test, average='weighted')
    )

best_k = k_range[error_rates.index(min(error_rates))]
print(f"\n✅ Best K value is : {best_k}")

plt.figure(figsize=(8, 4))
plt.plot(k_range, error_rates, marker='o',
         color='navy', linestyle='--')
plt.axvline(x=best_k, color='red',
            linestyle=':', label=f'Best K = {best_k}')
plt.title('Elbow Method — Finding Optimal K')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.legend()
plt.tight_layout()
plt.savefig('elbow_method.png')
print("Elbow method chart saved as elbow_method.png ✅")
plt.show()


#FINAL SUMMARY
print("\n" + "=" * 50)
print("            PROJECT 2 COMPLETE ✅")
print("=" * 50)
print(f"  Algorithm   : K-Nearest Neighbors")
print(f"  Dataset     : Iris (150 samples, 3 classes)")
print(f"  Best K      : {best_k}")
print(f"  F1 Score    : {f1:.2f}")
print("  Output files: confusion_matrix.png")
print("                elbow_method.png")
print("=" * 50)