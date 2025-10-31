from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import numpy as np
from datetime import datetime
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class SustainabilityClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.preprocessing_info = None
        self.label_mapping = {
            'kemasan_boros': 'Kemasan Boros',
            'produk_tidak_tahan_lama': 'Produk Tidak Tahan Lama', 
            'produk_awet_berkualitas': 'Produk Awet & Berkualitas',
            'netral': 'Netral'
        }
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessing components"""
        try:
            # Load model
            model_path = 'sustainability_model_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load vectorizer
            vectorizer_path = 'sustainability_model_vectorizer.pkl'
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Vectorizer loaded successfully")
            else:
                logger.error(f"Vectorizer file not found: {vectorizer_path}")
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
            
            # Load preprocessing info
            preprocessing_path = 'sustainability_model_preprocessing.pkl'
            if os.path.exists(preprocessing_path):
                with open(preprocessing_path, 'rb') as f:
                    self.preprocessing_info = pickle.load(f)
                logger.info("Preprocessing info loaded successfully")
            else:
                logger.warning(f"Preprocessing file not found: {preprocessing_path}")
                self.preprocessing_info = {}
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise e
    
    def preprocess_text(self, text):
        """
        Preprocess text based on the pipeline used in training
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers (keep letters that might be attached)
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation except apostrophes
        text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        
        return ' '.join(words)
    
    def predict(self, text):
        """
        Predict sustainability label for given text
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                return {
                    'prediction': 'netral',
                    'prediction_label': 'Netral',
                    'confidence': 0.5,
                    'all_probabilities': {},
                    'processed_text': processed_text
                }
            
            # Vectorize text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Get prediction
            prediction = self.model.predict(text_vector)[0]
            
            # Get probabilities
            probabilities = self.model.predict_proba(text_vector)[0]
            
            # Get class labels
            classes = self.model.classes_
            
            # Create probability dictionary
            prob_dict = {}
            for i, class_name in enumerate(classes):
                prob_dict[class_name] = float(probabilities[i])
            
            # Get confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Get prediction label
            prediction_label = self.label_mapping.get(prediction, prediction)
            
            return {
                'prediction': prediction,
                'prediction_label': prediction_label,
                'confidence': confidence,
                'all_probabilities': prob_dict,
                'processed_text': processed_text
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise e

# Initialize classifier
classifier = SustainabilityClassifier()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint to check if API is running"""
    return jsonify({
        'message': 'Sustainability Classification API is running!',
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': classifier.model is not None,
        'vectorizer_loaded': classifier.vectorizer is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_status': 'loaded' if classifier.model is not None else 'not_loaded',
        'vectorizer_status': 'loaded' if classifier.vectorizer is not None else 'not_loaded'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        # Extract review text
        review_text = data.get('review', '')
        
        if not review_text or not review_text.strip():
            return jsonify({
                'error': 'Review text is required and cannot be empty',
                'status': 'error'
            }), 400
        
        # Validate review length
        if len(review_text) > 5000:
            return jsonify({
                'error': 'Review text is too long (maximum 5000 characters)',
                'status': 'error'
            }), 400
        
        # Make prediction
        result = classifier.predict(review_text)
        
        # Add metadata
        result.update({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'original_text': review_text,
            'text_length': len(review_text)
        })
        
        logger.info(f"Prediction successful: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple reviews"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        reviews = data.get('reviews', [])
        
        if not isinstance(reviews, list):
            return jsonify({
                'error': 'Reviews should be a list',
                'status': 'error'
            }), 400
        
        if len(reviews) == 0:
            return jsonify({
                'error': 'At least one review is required',
                'status': 'error'
            }), 400
        
        if len(reviews) > 100:
            return jsonify({
                'error': 'Maximum 100 reviews allowed per batch',
                'status': 'error'
            }), 400
        
        # Process all reviews
        results = []
        for i, review in enumerate(reviews):
            if not isinstance(review, str):
                results.append({
                    'index': i,
                    'error': 'Review must be a string',
                    'status': 'error'
                })
                continue
            
            if len(review.strip()) == 0:
                results.append({
                    'index': i,
                    'error': 'Review cannot be empty',
                    'status': 'error'
                })
                continue
            
            try:
                result = classifier.predict(review)
                result.update({
                    'index': i,
                    'original_text': review,
                    'status': 'success'
                })
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        model_info = {
            'model_type': type(classifier.model).__name__ if classifier.model else None,
            'model_classes': list(classifier.model.classes_) if classifier.model else None,
            'vectorizer_type': type(classifier.vectorizer).__name__ if classifier.vectorizer else None,
            'vocabulary_size': len(classifier.vectorizer.vocabulary_) if classifier.vectorizer else None,
            'preprocessing_info': classifier.preprocessing_info,
            'label_mapping': classifier.label_mapping,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': f'Error getting model info: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("="*60)
    print("🌱 SUSTAINABILITY CLASSIFICATION API")
    print("="*60)
    print("📊 SDG 12: Responsible Consumption and Production")
    print("🚀 Server starting...")
    print("📍 Available endpoints:")
    print("   GET  /           - Home page")
    print("   GET  /health     - Health check")
    print("   POST /predict    - Single prediction")
    print("   POST /predict/batch - Batch prediction")
    print("   GET  /model/info - Model information")
    print("="*60)
    
    try:
        # Test model loading
        test_result = classifier.predict("Test review untuk cek model")
        print("✅ Model loaded and working correctly")
        print(f"   Test prediction: {test_result['prediction']}")
        print("="*60)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("   Please ensure model files are in the same directory:")
        print("   - sustainability_model_model.pkl")
        print("   - sustainability_model_vectorizer.pkl")
        print("   - sustainability_model_preprocessing.pkl")
        print("="*60)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Port 5000
        debug=True       # Enable debug mode
    )
