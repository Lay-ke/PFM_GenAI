# inference.py
import os
import json
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# --------------------------
# Model Class Definition (matching train.py exactly)
# --------------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Take the last time step
        return self.fc(out)

# --------------------------
# Model loading
# --------------------------
def model_fn(model_dir):
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # Load model checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device)
        
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract model parameters from checkpoint
        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        num_layers = checkpoint["num_layers"]
        num_classes = checkpoint["num_classes"]
        dropout = checkpoint.get("dropout", 0.3)  # Default to 0.3 if not saved
        
        logger.info(f"Model params - input_size: {input_size}, hidden_size: {hidden_size}, "
                   f"num_layers: {num_layers}, num_classes: {num_classes}, dropout: {dropout}")
        
        # Rebuild model with saved parameters
        model = RNNClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        
        # Load scaler and metadata
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        
        with open(os.path.join(model_dir, "features.txt")) as f:
            features = [line.strip() for line in f]
        
        with open(os.path.join(model_dir, "classes.txt")) as f:
            classes = [line.strip() for line in f]
        
        logger.info(f"Loaded features: {features}")
        logger.info(f"Loaded classes: {classes}")
        
        return {
            "model": model, 
            "scaler": scaler, 
            "features": features, 
            "classes": classes,
            "device": device
        }
        
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise e

# --------------------------
# Input parsing
# --------------------------
def input_fn(request_body, request_content_type):
    try:
        logger.info(f"Content type: {request_content_type}")
        logger.info(f"Request body type: {type(request_body)}")
        
        if request_content_type == "application/json":
            if isinstance(request_body, str):
                data = json.loads(request_body)
            else:
                data = json.loads(request_body.decode('utf-8'))
            df = pd.DataFrame(data["instances"])
            logger.info(f"Input DataFrame shape: {df.shape}")
            logger.info(f"Input DataFrame columns: {list(df.columns)}")
            return df
            
        elif request_content_type == "application/x-npy":
            # Handle numpy array input (SageMaker default)
            import io
            if isinstance(request_body, str):
                request_body = request_body.encode('utf-8')
            
            # Try to load as numpy array
            try:
                data = np.load(io.BytesIO(request_body), allow_pickle=True)
                logger.info(f"Loaded numpy data shape: {data.shape}")
                logger.info(f"Loaded numpy data: {data}")
                
                # If it's a structured array or object array containing JSON
                if data.dtype == 'O':
                    # Handle object array containing JSON data
                    json_data = data.item()
                    if isinstance(json_data, dict):
                        df = pd.DataFrame(json_data["instances"])
                    else:
                        # Try parsing as JSON string
                        json_data = json.loads(str(json_data))
                        df = pd.DataFrame(json_data["instances"])
                else:
                    # Handle regular numpy array - assume it's feature data
                    # This case would need feature names to be provided separately
                    raise ValueError("Raw numpy arrays not supported without feature names")
                    
                logger.info(f"Input DataFrame shape: {df.shape}")
                logger.info(f"Input DataFrame columns: {list(df.columns)}")
                return df
                
            except Exception as npy_error:
                logger.error(f"Error loading numpy data: {npy_error}")
                # Fallback: try treating as JSON string
                try:
                    if isinstance(request_body, bytes):
                        request_body = request_body.decode('utf-8')
                    data = json.loads(request_body)
                    df = pd.DataFrame(data["instances"])
                    return df
                except:
                    raise ValueError(f"Could not parse data as numpy or JSON: {npy_error}")
                    
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
            
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise e

# --------------------------
# Prediction
# --------------------------
def predict_fn(input_data, model_artifacts):
    try:
        model = model_artifacts["model"]
        scaler = model_artifacts["scaler"]
        features = model_artifacts["features"]
        classes = model_artifacts["classes"]
        device = model_artifacts["device"]
        
        logger.info(f"Input data columns: {list(input_data.columns)}")
        logger.info(f"Required features: {features}")
        
        # Check if all required features are present
        missing_features = [f for f in features if f not in input_data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Keep only required features in correct order
        X = input_data[features].values
        logger.info(f"Feature matrix shape before scaling: {X.shape}")
        
        # Scale the features
        X_scaled = scaler.transform(X)
        logger.info(f"Feature matrix shape after scaling: {X_scaled.shape}")
        
        # For single predictions, we need to create a sequence
        # Since the training used sequences, we'll replicate the single sample
        # to create a sequence of length 1 (minimum for RNN)
        seq_len = 1  # We'll use sequence length of 1 for inference
        
        # Convert to tensor: (batch, seq_len, features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
        logger.info(f"Input tensor shape: {X_tensor.shape}")
        
        # Predict
        with torch.no_grad():
            outputs = model(X_tensor)
            logger.info(f"Model outputs shape: {outputs.shape}")
            logger.info(f"Model outputs: {outputs}")
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = probabilities.cpu().numpy()
            
            logger.info(f"Predictions: {preds}")
            logger.info(f"Probabilities: {probs}")
        
        # Convert predictions to class names
        result = []
        for i, pred in enumerate(preds):
            result.append({
                "predicted_class": classes[pred],
                "confidence": float(probs[i][pred])
            })
        
        logger.info(f"Final result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise e

# --------------------------
# Output formatting
# --------------------------
def output_fn(prediction, response_content_type):
    try:
        logger.info(f"Formatting output: {prediction}")
        
        if response_content_type == "application/json":
            return json.dumps({"predictions": prediction})
        else:
            # Default to JSON
            return json.dumps({"predictions": prediction})
            
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise e