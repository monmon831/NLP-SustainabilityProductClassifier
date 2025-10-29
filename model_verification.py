import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def verify_saved_model():
    """
    Fungsi untuk memverifikasi model yang tersimpan dalam file pickle
    """
    
    print("="*80)
    print("SUSTAINABILITY MODEL VERIFICATION")
    print("="*80)
    
    try:
        # 1. Load model yang tersimpan
        print("\n1. LOADING SAVED MODEL...")
        model = joblib.load('sustainability_model_model.pkl')
        
        # Cek tipe model
        model_type = type(model).__name__
        model_module = type(model).__module__
        
        print(f"Model Type: {model_type}")
        print(f"Model Module: {model_module}")
        print(f"Full Model Class: {type(model)}")
        
        # 2. Cek parameter model (jika ada)
        print(f"\n2. MODEL PARAMETERS:")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print("  Model tidak memiliki method get_params()")
            
        # 3. Cek apakah model sudah di-fit
        print(f"\n3. MODEL STATUS:")
        fitted_attributes = []
        common_fitted_attrs = ['coef_', 'intercept_', 'classes_', 'feature_names_in_', 'n_features_in_']
        
        for attr in common_fitted_attrs:
            if hasattr(model, attr):
                fitted_attributes.append(attr)
                attr_value = getattr(model, attr)
                if isinstance(attr_value, np.ndarray):
                    print(f"  {attr}: shape {attr_value.shape}")
                else:
                    print(f"  {attr}: {attr_value}")
        
        if fitted_attributes:
            print("  ✓ Model sudah di-fit (trained)")
        else:
            print("  ✗ Model belum di-fit atau atribut tidak ditemukan")
            
        # 4. Load vectorizer dan preprocessing info
        print(f"\n4. LOADING SUPPORTING FILES...")
        try:
            vectorizer = joblib.load('sustainability_model_vectorizer.pkl')
            print(f"  Vectorizer Type: {type(vectorizer).__name__}")
            if hasattr(vectorizer, 'vocabulary_'):
                print(f"  Vocabulary Size: {len(vectorizer.vocabulary_)}")
                
            preprocessing_info = joblib.load('sustainability_model_preprocessing.pkl')
            print(f"  Preprocessing Info Type: {type(preprocessing_info)}")
            if isinstance(preprocessing_info, dict):
                print(f"  Preprocessing Keys: {list(preprocessing_info.keys())}")
                
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            
        # 5. Test prediction jika memungkinkan
        print(f"\n5. TESTING MODEL PREDICTION...")
        try:
            # Sample texts untuk testing
            test_reviews = [
                "Barang bagus, tahan lama dan berkualitas tinggi. Recommended!",
                "Kemasan terlalu banyak plastik, boros banget packaging nya",
                "Produk cepat rusak, tidak awet, mengecewakan",
                "Pengiriman cepat, terima kasih"
            ]
            
            # Preprocess dan predict (perlu disesuaikan dengan pipeline Anda)
            # Ini adalah contoh umum, mungkin perlu disesuaikan
            if 'vectorizer' in locals():
                X_test = vectorizer.transform(test_reviews)
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                print("  Test Predictions:")
                for i, (review, pred) in enumerate(zip(test_reviews, predictions)):
                    prob_text = ""
                    if probabilities is not None:
                        max_prob = np.max(probabilities[i])
                        prob_text = f" (confidence: {max_prob:.4f})"
                    print(f"    Review {i+1}: {pred}{prob_text}")
                    print(f"    Text: {review[:50]}...")
            else:
                print("  Cannot test prediction - vectorizer not loaded")
                
        except Exception as e:
            print(f"  Error in prediction test: {e}")
            
        return model, model_type
        
    except FileNotFoundError:
        print("Error: File 'sustainability_model_model.pkl' tidak ditemukan!")
        print("Pastikan file ada di direktori yang sama dengan script ini.")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def compare_with_documentation():
    """
    Membandingkan hasil verifikasi dengan dokumentasi
    """
    print("\n" + "="*80)
    print("COMPARISON WITH DOCUMENTATION")
    print("="*80)
    
    documented_results = {
        "Naive Bayes": {"accuracy": 0.8561, "f1_macro": 0.8610},
        "Logistic Regression": {"accuracy": 0.9480, "f1_macro": 0.9499},
        "Linear SVC": {"accuracy": 0.9685, "f1_macro": 0.9698}
    }
    
    web_interface_claims = {
        "accuracy": 0.9733,  # 97.33%
        "f1_macro": 0.9746   # 97.46%
    }
    
    print("\nDOCUMENTED MODEL PERFORMANCES:")
    for model_name, metrics in documented_results.items():
        print(f"  {model_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"    F1-Score: {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    
    print(f"\nWEB INTERFACE CLAIMS:")
    print(f"  Accuracy: {web_interface_claims['accuracy']:.4f} ({web_interface_claims['accuracy']*100:.2f}%)")
    print(f"  F1-Score: {web_interface_claims['f1_macro']:.4f} ({web_interface_claims['f1_macro']*100:.2f}%)")
    
    print(f"\nANALYSIS:")
    print("  - Web interface menunjukkan performa Linear SVC")
    print("  - Namun nilai di web (97.33%, 97.46%) lebih tinggi dari dokumentasi (96.85%, 96.98%)")
    print("  - Kemungkinan ada hyperparameter tuning tambahan atau evaluasi pada test set berbeda")

def detailed_model_inspection():
    """
    Inspeksi detail untuk berbagai jenis model
    """
    print("\n" + "="*80)
    print("DETAILED MODEL INSPECTION")
    print("="*80)
    
    try:
        model = joblib.load('sustainability_model_model.pkl')
        model_type = type(model).__name__
        
        print(f"\nINSPECTING {model_type}...")
        
        # Untuk Naive Bayes
        if 'Naive' in model_type or 'NB' in model_type:
            print("  NAIVE BAYES SPECIFIC ATTRIBUTES:")
            if hasattr(model, 'alpha'):
                print(f"    Alpha (smoothing): {model.alpha}")
            if hasattr(model, 'class_log_prior_'):
                print(f"    Classes: {model.classes_}")
                print(f"    Class log priors shape: {model.class_log_prior_.shape}")
            if hasattr(model, 'feature_log_prob_'):
                print(f"    Feature log probabilities shape: {model.feature_log_prob_.shape}")
                
        # Untuk Logistic Regression
        elif 'Logistic' in model_type:
            print("  LOGISTIC REGRESSION SPECIFIC ATTRIBUTES:")
            if hasattr(model, 'C'):
                print(f"    C (regularization): {model.C}")
            if hasattr(model, 'coef_'):
                print(f"    Coefficients shape: {model.coef_.shape}")
            if hasattr(model, 'intercept_'):
                print(f"    Intercept: {model.intercept_}")
                
        # Untuk SVC
        elif 'SVC' in model_type or 'SVM' in model_type:
            print("  SVC SPECIFIC ATTRIBUTES:")
            if hasattr(model, 'C'):
                print(f"    C (regularization): {model.C}")
            if hasattr(model, 'kernel'):
                print(f"    Kernel: {model.kernel}")
            if hasattr(model, 'support_vectors_'):
                print(f"    Number of support vectors: {len(model.support_vectors_) if model.support_vectors_ is not None else 'None'}")
            if hasattr(model, 'coef_'):
                print(f"    Coefficients shape: {model.coef_.shape}")
                
        # Atribut umum
        print(f"\n  COMMON ATTRIBUTES:")
        if hasattr(model, 'classes_'):
            print(f"    Classes: {model.classes_}")
        if hasattr(model, 'n_features_in_'):
            print(f"    Number of features: {model.n_features_in_}")
            
    except Exception as e:
        print(f"Error in detailed inspection: {e}")

if __name__ == "__main__":
    # Jalankan verifikasi
    model, model_type = verify_saved_model()
    
    if model is not None:
        print(f"\n🎯 CONCLUSION: Model yang tersimpan adalah {model_type}")
        
        # Inspeksi detail
        detailed_model_inspection()
        
        # Bandingkan dengan dokumentasi
        compare_with_documentation()
        
        print(f"\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        print("1. Update web interface untuk menampilkan model yang benar")
        print("2. Pastikan konsistensi antara dokumentasi, model tersimpan, dan UI")
        print("3. Jika ingin menggunakan Linear SVC, train ulang dan simpan model tersebut")
        print("4. Jika ingin menggunakan Naive Bayes, pastikan UI menampilkan metrik yang benar")
    else:
        print("❌ Tidak dapat memverifikasi model - file tidak ditemukan atau error")