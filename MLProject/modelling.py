import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # Mengatur lokasi penyimpanan database MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 1. Mengaktifkan MLflow Autologging
    mlflow.autolog()

    # [PERBAIKAN] Baris di bawah dinonaktifkan agar tidak bentrok dengan GitHub Actions
    # mlflow.set_experiment("Credit_Risk_Modelling")

    # 2. Memuat Data Bersih hasil Preprocessing
    print("Memuat data bersih...")
    # Pastikan file credit_risk_preprocessing.csv ada di folder yang sama
    df = pd.read_csv('credit_risk_preprocessing.csv')

    # Memisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Membagi data menjadi 80% Train dan 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Menjalankan MLflow Run
    with mlflow.start_run() as run:
        print("Mulai melatih model Random Forest...")
        
        # Inisialisasi model Scikit-Learn
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Proses Pelatihan
        model.fit(X_train, y_train)
        
        # Evaluasi Model
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        print(f"\n[SUKSES] Model berhasil dilatih!")
        print(f"Akurasi Model: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Menampilkan Run ID untuk referensi
        print(f"\nSimpan Run ID ini jika diperlukan: {run.info.run_id}")

if __name__ == "__main__":
    train_model()