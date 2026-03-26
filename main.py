import pandas as pd
import warnings
import os
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH
from src.preprocessing import load_arff_to_dataframe, preprocess_dataframe
from src.vector_store import VectorStore
from src.inference import RAGInference
from src.models_classic import ClassicModels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. SILENCER: Hide KAN/Torch/TensorFlow warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    print("🚀 Starting IDS Project: Full Dataset Mode")
    
    # 2. Load Data
    train_df_raw = load_arff_to_dataframe(TRAIN_DATA_PATH)
    test_df_raw = load_arff_to_dataframe(TEST_DATA_PATH)
    
    train_df = preprocess_dataframe(train_df_raw)
    test_df = preprocess_dataframe(test_df_raw)

   
    # 3. FORCE FULL DATASET INDEXING
    vdb = VectorStore()
    
    # We are commenting out the 'if' check to force it to run once
    print(f"🚀 FORCING INDEX of {len(train_df)} rows...")
    vdb.add_to_index(train_df) 
    print("✅ Full Indexing Complete!")
   

    # 4. Train Classic Models & KAN
    classic = ClassicModels()
    X_train, y_train, X_test, y_test = classic.prepare_data(train_df, test_df)
    
    # Using a 5k subset for training KAN/ML to keep it fast, but using FULL data for RAG
    classic.train_traditional(X_train.head(10000), y_train[:10000])
    classic.train_kan(X_train.head(5000), y_train[:5000], steps=3)

    # 5. RAG Evaluation Loop
    rag = RAGInference()
    y_true, y_rf, y_xgb, y_kan, y_rag = [], [], [], [], []
    
    # Open a log file to save AI explanations
    with open("ai_forensics_report.txt", "w") as f:
        f.write("=== IDS RAG FORENSIC LOG ===\n\n")

    print("\n🧐 Starting Comparative Evaluation (50 Samples)...")
    test_samples = test_df.sample(50, random_state=42)

    for idx, row in test_samples.iterrows():
        actual = row['label']
        # Predict with ML models
        preds = classic.get_predictions(pd.DataFrame([row.drop('label')]))
        
        # Predict with RAG
        similar_cases = vdb.query_similar_cases(row)
        analysis = rag.generate_analysis(row, similar_cases)
        rag_label = "anomaly" if "anomaly" in analysis.lower() else "normal"

        # Store Results
        y_true.append(actual)
        y_rf.append(preds['rf'][0])
        y_xgb.append(preds['xgb'][0])
        y_kan.append(preds['kan'][0])
        y_rag.append(rag_label)

        # Log AI reasoning to file
        with open("ai_forensics_report.txt", "a") as f:
            f.write(f"Sample {len(y_true)} | Actual: {actual} | RAG: {rag_label}\n")
            f.write(f"Reasoning: {analysis}\n")
            f.write("-" * 50 + "\n")

        print(f"✅ Processed {len(y_true)}/50 | Actual: {actual} | RAG: {rag_label}")

    # 6. Final Results & Metrics
    results = []
    models = [("Random Forest", y_rf), ("XGBoost", y_xgb), ("KAN", y_kan), ("RAG-IDS", y_rag)]
    
    for name, y_pred in models:
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label='anomaly', zero_division=0),
            "Recall": recall_score(y_true, y_pred, pos_label='anomaly', zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, pos_label='anomaly', zero_division=0)
        })

    report_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 FINAL COMPARATIVE STUDY RESULTS")
    print("="*50)
    print(report_df.to_string(index=False))
    
    # Save results for your plotting script
    report_df.to_csv("ids_model_comparison.csv", index=False)
    print("\n💾 Metrics saved to 'ids_model_comparison.csv'")
    print("📄 AI Explanations saved to 'ai_forensics_report.txt'")

if __name__ == "__main__":
    main()
    