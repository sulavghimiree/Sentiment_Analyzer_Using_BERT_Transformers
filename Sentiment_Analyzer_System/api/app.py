from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
"""Enable CORS for all routes to allow requests from file:// and other ports during local dev."""
CORS(app, resources={r"/*": {"origins": "*"}})


tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global tokenizer, model
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'bert-sentiment')
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        
        logger.info("Model and tokenizer loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_sentiment(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        predicted_class_id = predictions.argmax().item()
        confidence_scores = predictions[0].tolist()
        
        id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        predicted_label = id2label[predicted_class_id]
        
        results = {
            "text": text,
            "prediction": predicted_label,
            "confidence": confidence_scores[predicted_class_id],
            "all_scores": {
                "Negative": confidence_scores[0],
                "Neutral": confidence_scores[1],
                "Positive": confidence_scores[2]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise e

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "description": "BERT-based sentiment analysis for text classification",
        "endpoints": {
            "POST /predict": "Analyze sentiment of a single text",
            "POST /predict_batch": "Analyze sentiment of multiple texts",
            "GET /health": "Check API health status"
        },
        "model_info": {
            "classes": ["Negative", "Neutral", "Positive"],
            "model_type": "BERT for Sequence Classification"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    global model, tokenizer
    
    model_loaded = model is not None and tokenizer is not None
    
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or tokenizer is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        result = predict_sentiment(text.strip())
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if model is None or tokenizer is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                "error": "'texts' must be a list of strings"
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                "error": "Texts list cannot be empty"
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                "error": "Maximum batch size is 100 texts"
            }), 400
        
        results = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append({
                    "index": i,
                    "error": "Empty text"
                })
                continue
            
            try:
                result = predict_sentiment(text.strip())
                result["index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "index": i,
                    "error": f"Prediction failed: {str(e)}"
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "total_processed": len(texts)
        })
        
    except Exception as e:
        logger.error(f"Error in predict_batch endpoint: {str(e)}")
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Please check the API documentation at the root endpoint '/'"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "Please check the allowed methods for this endpoint"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs for more details"
    }), 500

if __name__ == '__main__':
    logger.info("Starting Sentiment Analysis API...")
    
    if load_model():
        logger.info("Model loaded successfully. Starting Flask server...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False
        )
    else:
        logger.error("Failed to load model. Exiting...")
        exit(1)