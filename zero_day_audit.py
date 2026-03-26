import pandas as pd
import warnings
import os
from src.vector_store import VectorStore
from src.inference import RAGInference
from src.config import TEST_DATA_PATH
from src.preprocessing import load_arff_to_dataframe, preprocess_dataframe, row_to_string

warnings.filterwarnings("ignore")

def run_zero_day_audit():
    print("🕵️ Starting Zero-Day Vulnerability Audit...")
    
    # 1. Load the Test Data
    test_df_raw = load_arff_to_dataframe(TEST_DATA_PATH)
    test_df = preprocess_dataframe(test_df_raw)
    
    # 2. Initialize RAG Components
    vdb = VectorStore()
    rag = RAGInference()
    
    # 3. Find ALL anomalies in the test set
    anomalies = test_df[test_df['label'] != 'normal']
    
    if anomalies.empty:
        print("❌ No anomalies found in the test set to audit!")
        return

    # 4. Pick the first unique attack type found
    target_attack = anomalies.iloc[0]['label']
    sample_row = anomalies.iloc[0]
    
    print(f"\n🧪 TARGETING ATTACK TYPE: {target_attack.upper()}")
    print(f"🔬 This will be treated as a 'Zero-Day' (Hidden from Database)")
    print("-" * 50)
    
    # 5. THE FILTERED QUERY (The "Blindfold")
    # Query the collection but EXCLUDE the target attack type from matches
    results = vdb.collection.query(
        query_texts=[row_to_string(sample_row)],
        n_results=5,
        where={"label": {"$ne": str(target_attack)}} 
    )
    
    # 6. Generate Forensic Reasoning
    print("🧠 AI is analyzing the 'Unknown' pattern...")
    analysis = rag.generate_analysis(sample_row, results)
    
    # 7. Output the Audit Report
    print("\n📝 --- ZERO-DAY FORENSIC REPORT ---")
    print(f"ACTUAL ATTACK TYPE: {target_attack}")
    print(f"RAG DECISION: {'ANOMALY' if 'anomaly' in analysis.lower() else 'NORMAL'}")
    print("\nAI REASONING:")
    print(analysis)
    print("-" * 50)

    with open("zero_day_report.txt", "w") as f:
        f.write(f"ZERO-DAY AUDIT REPORT\nTarget: {target_attack}\n\nAnalysis:\n{analysis}")
    print("💾 Report saved to 'zero_day_report.txt'")

if __name__ == "__main__":
    run_zero_day_audit()