
import os
import time
import json
import pickle
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

# Model paths - adjust these based on your actual model directory structure
MODEL_BASE_PATH = "models/phobert"
INTENT_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "intent_classifier_final")
NER_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "ner_model_final")
INTENT_ENCODER_PATH = os.path.join(MODEL_BASE_PATH, "intent_encoder.pkl")
LABEL_MAPPINGS_PATH = os.path.join(NER_MODEL_PATH, "label_mappings.json")

# Data directories
TXT_DIR = "data/transcriptions"
OUT_DIR = "data/nlu_results"

class VNSLUProcessor:
    """Vietnamese Spoken Language Understanding processor using PhoBERT"""
    
    def __init__(self):
        """Initialize the NLU processor with trained models"""
        print(" Loading Vietnamese NLU models...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # Load models
        try:
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
            print(" Models loaded successfully")
        except Exception as e:
            print(f" Error loading models: {e}")
            raise
        
        # Load encoders and label mappings
        try:
            with open(INTENT_ENCODER_PATH, 'rb') as f:
                self.intent_encoder = pickle.load(f)
            
            with open(LABEL_MAPPINGS_PATH, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                self.slot_labels = label_data['slot_labels']
                self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            
            print(" Encoders and mappings loaded")
        except Exception as e:
            print(f" Error loading encoders: {e}")
            raise
        
        # Set models to evaluation mode
        self.intent_model.eval()
        self.ner_model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.intent_model = self.intent_model.cuda()
            self.ner_model = self.ner_model.cuda()
            print(" Models moved to GPU")
        
        print(" NLU processor ready!")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent and entities for Vietnamese text
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            Dictionary containing intent and entities
        """
        # Intent prediction
        intent_result = self._predict_intent(text)
        
        # NER prediction  
        entities = self._predict_entities(text)
        
        return {
            'text': text,
            'intent': intent_result['intent'],
            'intent_confidence': intent_result['confidence'],
            'entities': entities,
            'timestamp': time.time()
        }
    
    def _predict_intent(self, text: str) -> Dict[str, Any]:
        """Predict intent for the input text"""
        # Tokenize input
        intent_inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=128
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            intent_inputs = {k: v.cuda() for k, v in intent_inputs.items()}
        
        # Predict
        with torch.no_grad():
            intent_outputs = self.intent_model(**intent_inputs)
            intent_logits = intent_outputs.logits
            predicted_intent_id = torch.argmax(intent_logits, dim=-1).item()
            intent_confidence = torch.softmax(intent_logits, dim=-1).max().item()
        
        # Convert ID to intent label
        predicted_intent = self.intent_encoder.inverse_transform([predicted_intent_id])[0]
        
        return {
            'intent': predicted_intent,
            'confidence': float(intent_confidence)
        }
    
    def _predict_entities(self, text: str) -> List[Dict[str, str]]:
        """Predict named entities (slots) for the input text"""
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # Prepare inputs for NER model
        ner_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=False
        )
        
        # Move to GPU if available  
        if torch.cuda.is_available():
            ner_inputs = {k: v.cuda() for k, v in ner_inputs.items()}
        
        # Predict
        with torch.no_grad():
            ner_outputs = self.ner_model(**ner_inputs)
            predictions = torch.argmax(ner_outputs.logits, dim=-1).squeeze().tolist()
        
        # Convert predictions to entities
        entities = self._extract_entities(tokens, predictions)
        
        return entities
    
    def _extract_entities(self, tokens: List[str], predictions: List[int]) -> List[Dict[str, str]]:
        """Extract entities from token predictions using BIO tagging"""
        entities = []
        current_entity = None
        current_tokens = []
        
        # Ensure predictions list matches tokens length
        if len(predictions) > len(tokens):
            predictions = predictions[1:len(tokens)+1]  # Skip [CLS] token
        elif len(predictions) < len(tokens):
            predictions = predictions[:len(tokens)]
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if pred_id >= len(self.id2label):
                continue
                
            label = self.id2label[pred_id]
            
            if label.startswith('B-'):
                # Beginning of new entity
                if current_entity and current_tokens:
                    # Save previous entity
                    entities.append({
                        'type': current_entity,
                        'filler': self.tokenizer.convert_tokens_to_string(current_tokens).strip()
                    })
                
                current_entity = label[2:]  # Remove 'B-' prefix
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continuation of current entity
                current_tokens.append(token)
                
            else:
                # Outside entity or different entity
                if current_entity and current_tokens:
                    entities.append({
                        'type': current_entity,
                        'filler': self.tokenizer.convert_tokens_to_string(current_tokens).strip()
                    })
                current_entity = None
                current_tokens = []
        
        # Don't forget the last entity
        if current_entity and current_tokens:
            entities.append({
                'type': current_entity,
                'filler': self.tokenizer.convert_tokens_to_string(current_tokens).strip()
            })
        
        return entities

def main():
    """Main daemon loop"""
    print(" Starting Vietnamese NLU daemon...")
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Initialize NLU processor
    try:
        nlu_processor = VNSLUProcessor()
    except Exception as e:
        print(f" Failed to initialize NLU processor: {e}")
        return
    
    print(f" Monitoring {TXT_DIR} for transcription files...")
    
    # Main processing loop
    while True:
        try:
            # Check for new transcription files
            if not os.path.exists(TXT_DIR):
                time.sleep(1)
                continue
                
            for filename in os.listdir(TXT_DIR):
                if not filename.endswith(".txt"):
                    continue
                
                input_path = os.path.join(TXT_DIR, filename)
                
                # Read transcription
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if not text:
                        print(f"  Empty transcription: {filename}")
                        os.remove(input_path)
                        continue
                    
                    # Process with NLU
                    result = nlu_processor.predict(text)
                    
                    # Save results
                    output_filename = filename.replace(".txt", "_nlu.json")
                    output_path = os.path.join(OUT_DIR, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    # Remove processed transcription
                    os.remove(input_path)
                    
                    # Log results
                    print(f" NLU processed: {filename}")
                    print(f"    Text: {text}")
                    print(f"    Intent: {result['intent']} (confidence: {result['intent_confidence']:.3f})")
                    print(f"     Entities: {result['entities']}")
                    print(f"    Saved: {output_path}")
                    print("-" * 60)
                    
                except Exception as e:
                    print(f" Error processing {filename}: {e}")
                    # Don't remove file on error, might be temporary issue
                    continue
        
        except KeyboardInterrupt:
            print("\n NLU daemon stopped by user")
            break
        except Exception as e:
            print(f" Unexpected error in main loop: {e}")
            time.sleep(5)  # Wait before retrying
        
        time.sleep(1)

if __name__ == "__main__":
    main()