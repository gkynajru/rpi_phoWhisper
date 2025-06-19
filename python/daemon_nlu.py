import os
import time
import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

# Model paths - adjust these to match your actual model structure
MODEL_PATH = "models/phobert"
INTENT_MODEL_PATH = os.path.join(MODEL_PATH, "intent_classifier_final")
NER_MODEL_PATH = os.path.join(MODEL_PATH, "ner_model_final")

# Data directories (following the same pattern as daemon_stt.py)
TXT_DIR = "data/transcriptions"
OUT_DIR = "data/nlu_results"

class VNSLUPipeline:
    """Vietnamese SLU Pipeline for the rpi_phoWhisper project"""
    
    def __init__(self):
        print(" Loading Vietnamese NLU models...")
        
        # Load tokenizer (PhoBERT base)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # Load trained models
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
        
        # Load label encoders
        self._load_label_mappings()
        
        # Set models to evaluation mode
        self.intent_model.eval()
        self.ner_model.eval()
        
        print(" NLU pipeline ready!")
    
    def _load_label_mappings(self):
        """Load intent encoder and NER label mappings"""
        try:
            # Load intent encoder
            intent_encoder_path = os.path.join(MODEL_PATH, "intent_encoder.pkl")
            with open(intent_encoder_path, 'rb') as f:
                self.intent_encoder = pickle.load(f)
            
            # Load NER label mappings
            label_mappings_path = os.path.join(NER_MODEL_PATH, "label_mappings.json")
            with open(label_mappings_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            
            print(" Label mappings loaded")
            
        except Exception as e:
            print(f" Error loading label mappings: {e}")
            # Fallback - create basic mappings if files don't exist
            print("  Using fallback label mappings")
            self.intent_encoder = None
            self.id2label = {0: 'O'}  # Basic 'Outside' label
    
    def predict_intent(self, text):
        """Predict intent for the input text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
        # Convert ID to intent label
        if self.intent_encoder:
            try:
                predicted_intent = self.intent_encoder.inverse_transform([predicted_id])[0]
            except:
                predicted_intent = f"intent_{predicted_id}"
        else:
            predicted_intent = f"intent_{predicted_id}"
        
        return predicted_intent, confidence
    
    def predict_entities(self, text):
        """Predict entities/slots for the input text"""
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Prepare inputs
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Extract entities using BIO tagging
        entities = self._extract_entities_from_predictions(tokens, predictions)
        
        return entities
    
    def _extract_entities_from_predictions(self, tokens, predictions):
        """Extract entities from BIO predictions"""
        entities = []
        current_entity = None
        current_tokens = []
        
        # Handle single prediction case
        if isinstance(predictions, int):
            predictions = [predictions]
        
        # Skip [CLS] token prediction if present
        if len(predictions) > len(tokens):
            predictions = predictions[1:len(tokens)+1]
        
        # Ensure we don't have more tokens than predictions
        min_len = min(len(tokens), len(predictions))
        
        for i in range(min_len):
            token = tokens[i]
            pred_id = predictions[i]
            
            # Get label
            if pred_id in self.id2label:
                label = self.id2label[pred_id]
            else:
                label = 'O'  # Outside
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_tokens).strip()
                    entities.append(f"{current_entity}:{entity_text}")
                
                # Start new entity
                current_entity = label[2:]  # Remove 'B-' prefix
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
                
            else:
                # End current entity
                if current_entity and current_tokens:
                    entity_text = self.tokenizer.convert_tokens_to_string(current_tokens).strip()
                    entities.append(f"{current_entity}:{entity_text}")
                current_entity = None
                current_tokens = []
        
        # Don't forget the last entity
        if current_entity and current_tokens:
            entity_text = self.tokenizer.convert_tokens_to_string(current_tokens).strip()
            entities.append(f"{current_entity}:{entity_text}")
        
        return entities
    
    def process(self, text):
        """Main processing function - like the original predict() but more comprehensive"""
        if not text or not text.strip():
            return {"error": "Empty input"}
        
        try:
            # Predict intent
            intent, confidence = self.predict_intent(text)
            
            # Predict entities
            entities = self.predict_entities(text)
            
            # Format result
            result = {
                "text": text,
                "intent": intent,
                "confidence": f"{confidence:.3f}",
                "entities": entities,
                "entity_count": len(entities)
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

# Initialize the NLU pipeline (like loading models in the original)
nlu_pipeline = VNSLUPipeline()

def predict(text):
    """Main prediction function (keeping the same interface as original)"""
    result = nlu_pipeline.process(text)
    
    # Convert to JSON string (like original output format)
    return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f" Monitoring {TXT_DIR} for transcription files...")
    
    # Main daemon loop (following the exact same pattern as daemon_stt.py)
    while True:
        try:
            # Check if transcriptions directory exists
            if not os.path.exists(TXT_DIR):
                time.sleep(1)
                continue
            
            # Process each .txt file (same pattern as daemon_stt.py)
            for f in os.listdir(TXT_DIR):
                if not f.endswith(".txt"): 
                    continue
                
                path = os.path.join(TXT_DIR, f)
                
                # Read transcription (with encoding handling)
                try:
                    with open(path, 'r', encoding='utf-8') as r:
                        txt = r.read().strip()
                except UnicodeDecodeError:
                    try:
                        with open(path, 'r', encoding='utf-16') as r:
                            txt = r.read().strip()
                    except:
                        print(f" Cannot read {f}, skipping...")
                        os.remove(path)
                        continue
                
                # Skip empty files
                if not txt:
                    os.remove(path)
                    continue
                
                # Process with NLU pipeline
                result = predict(txt)
                
                # Save result (same pattern as daemon_stt.py)
                out = os.path.join(OUT_DIR, f.replace(".txt", "_nlu.txt"))
                with open(out, "w", encoding='utf-8') as w:
                    w.write(result)
                
                # Remove processed transcription (same as daemon_stt.py)
                os.remove(path)
                
                # Log success (same format as daemon_stt.py)
                print(f"  NLU processed → {out}")
                print(f"    Input: {txt}")
                print(f"    Output: {result}")
                
        except KeyboardInterrupt:
            print("\n NLU daemon stopped")
            break
        except Exception as e:
            print(f" Error in main loop: {e}")
        
        time.sleep(1)
import os 
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/phobert"
TXT_DIR = "data/transcriptions"
OUT_DIR = "data/nlu_results"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text):
    inp = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = model(**inp).logits
    return logits.argmax(dim=-1).item()

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    while True:
        for f in os.listdir(TXT_DIR):
            if not f.endswith(".txt"): continue
            path = os.path.join(TXT_DIR, f)
            with open(path) as r: txt = r.read()
            cls = predict(txt)
            out = os.path.join(OUT_DIR, f.replace(".txt", "_nlu.txt"))
            with open(out, "w") as w: w.write(str(cls))
            os.remove(path)
            print(f"✔️  NLU → {out}")
        time.sleep(1)
