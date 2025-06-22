# daemon_nlu.py
import os
import time
import queue
import threading
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import numpy as np
import json

# Configuration
INTENT_MODEL_PATH = "models/phobert/intent_classifier_final"
NER_MODEL_PATH = "models/phobert/ner_model_final"
INTENT_ENCODER_PATH = "models/phobert/intent_encoder.pkl"
NLU_RESULTS_DIR = "data/nlu_results"
TRANSCRIPTION_DIR = "data/transcriptions"

# Create output directory
os.makedirs(NLU_RESULTS_DIR, exist_ok=True)

# Shared queue for transcriptions
transcription_queue = queue.Queue()

# File counter for unique filenames
file_counter = 0

# Initialize Intent Classifier
try:
    print("🔄 Loading Intent Classifier model...")
    intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
    intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH)
    intent_model.eval()
    with open(INTENT_ENCODER_PATH, 'rb') as f:
        intent_encoder = pickle.load(f)
    print("✅ Intent Classifier loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Intent Classifier: {e}")
    exit(1)

# Initialize NER model with pipeline
try:
    print("🔄 Loading NER model...")
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    
    # Create pipeline for NER
    ner_pipeline = pipeline(
        "token-classification",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple"
    )
    print("✅ NER model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load NER model: {e}")
    exit(1)

def process_intent(text):
    """Process text to predict intent using PhoBERT Intent Classifier"""
    try:
        inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = intent_model(**inputs)
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_intent = intent_encoder.classes_[predicted_id]
        confidence = torch.softmax(logits, dim=-1)[0, predicted_id].item()
        return {"intent": predicted_intent, "confidence": confidence}
    except Exception as e:
        print(f"❌ Intent prediction error: {e}")
        return {"intent": "unknown", "confidence": 0.0}

def process_ner(text):
    """Process text to extract named entities using PhoBERT NER with pipeline"""
    try:
        # Use pipeline for NER
        results = ner_pipeline(text)
        
        # Format entities to match the original structure but with score
        entities = []
        for result in results:
            entities.append({
                'entity': result['entity_group'],
                'value': result['word'],
                'score': round(result['score'], 4)
            })
        
        return entities
    except Exception as e:
        print(f"❌ NER error: {e}")
        return []

def monitor_transcription_files():
    """Monitor transcription files and process them with NLU"""
    global file_counter
    processed_files = set()
    
    while True:
        try:
            if os.path.exists(TRANSCRIPTION_DIR):
                transcription_files = [f for f in os.listdir(TRANSCRIPTION_DIR)
                                     if f.endswith('.txt') and f not in processed_files]
                
                for filename in transcription_files:
                    file_path = os.path.join(TRANSCRIPTION_DIR, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if text:
                            print(f"📜 Processing transcription: {filename} ('{text}')")
                            start_time = time.time()
                            
                            # Process intent and NER
                            intent_result = process_intent(text)
                            ner_result = process_ner(text)
                            
                            latency = time.time() - start_time
                            print(f"🤖 Intent: {intent_result['intent']} (Confidence: {intent_result['confidence']:.2f})")
                            print(f"📍 Entities: {ner_result}")
                            print(f"⏱️ NLU Latency: {latency:.2f}s")
                            
                            # Save NLU result with unique timestamp and counter
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            file_counter += 1
                            nlu_result_path = f"{NLU_RESULTS_DIR}/nlu_{timestamp}_{file_counter:03d}.json"
                            with open(nlu_result_path, 'w', encoding='utf-8') as f:
                                json.dump({
                                    "transcription": text,
                                    "intent": intent_result,
                                    "entities": ner_result
                                }, f, ensure_ascii=False, indent=2)
                            print(f"💾 NLU result saved: {nlu_result_path}")
                        
                        processed_files.add(filename)
                    
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
                        processed_files.add(filename)  # Skip problematic files
                        
            time.sleep(0.5)  # Check every 500ms
            
        except KeyboardInterrupt:
            print("🛑 Stopping NLU file monitor...")
            break
        except Exception as e:
            print(f"❌ File monitor error: {e}")
            time.sleep(1)

def nlu_queue_processor():
    """Process transcriptions from queue"""
    global file_counter
    print("🎧 NLU queue processor started...")
    
    while True:
        try:
            text = transcription_queue.get(timeout=1.0)
            
            start_time = time.time()
            intent_result = process_intent(text)
            ner_result = process_ner(text)
            latency = time.time() - start_time
            
            print(f"🎤 Queue Transcription: '{text}'")
            print(f"🤖 Intent: {intent_result['intent']} (Confidence: {intent_result['confidence']:.2f})")
            print(f"📍 Entities: {ner_result}")
            print(f"⏱️ NLU Latency: {latency:.2f}s")
            
            # Save NLU result with unique timestamp and counter
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
            file_counter += 1
            nlu_result_path = f"{NLU_RESULTS_DIR}/nlu_queue_{timestamp}_{file_counter:03d}.json"
            with open(nlu_result_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "transcription": text,
                    "intent": intent_result,
                    "entities": ner_result
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 Queue NLU result saved: {nlu_result_path}")
            
            transcription_queue.task_done()
            
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("🛑 Stopping NLU queue processor...")
            break
        except Exception as e:
            print(f"❌ NLU queue error: {e}")
            continue

def main():
    """Main NLU service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NLU Service')
    parser.add_argument('--method', choices=['1', '2', '3'],
                       help='Processing method: 1=File monitoring, 2=Queue processing, 3=Both')
    args = parser.parse_args()
    
    print("🚀 Starting NLU service...")
    print(f"📁 Monitoring directory: {TRANSCRIPTION_DIR}")
    print(f"💾 Saving NLU results to: {NLU_RESULTS_DIR}")
    
    if args.method:
        processing_method = args.method
        print(f"🎯 Using method {processing_method} (from command line)")
    else:
        print("Choose processing method:")
        print("1. File monitoring (recommended)")
        print("2. Queue processing")
        print("3. Both")
        processing_method = input("Enter (1/2/3): ").strip()
        
        while processing_method not in ['1', '2', '3']:
            print("❌ Invalid choice, please enter 1, 2, or 3")
            processing_method = input("Enter (1/2/3): ").strip()
    
    threads = []
    
    if processing_method in ['1', '3']:
        file_thread = threading.Thread(target=monitor_transcription_files)
        file_thread.daemon = True
        file_thread.start()
        threads.append(file_thread)
        print("✅ File monitoring thread started")
    
    if processing_method in ['2', '3']:
        queue_thread = threading.Thread(target=nlu_queue_processor)
        queue_thread.daemon = True
        queue_thread.start()
        threads.append(queue_thread)
        print("✅ Queue processing thread started")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping NLU service...")
        print("✅ NLU service stopped")

if __name__ == "__main__":
    main()