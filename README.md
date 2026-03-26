# 🛡️ RAG-IDS: Intrusion Detection with LLM Forensics

An advanced Intrusion Detection System leveraging **Retrieval-Augmented Generation (RAG)** and **Kolmogorov-Arnold Networks (KAN)** to provide explainable security alerts.

## 📊 Performance at a Glance
Our model prioritize **Recall (Threat Detection)** to ensure zero-day attacks are caught before they breach the perimeter.

| Model | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Random Forest | 0.78 | 0.60 | 0.73 |
| **RAG-IDS (Ours)** | **0.62** | **0.96** | **0.71** |

## 🚀 Key Features
* **125k Records Indexed:** Full NSL-KDD training set processed into ChromaDB.
* **Explainable AI:** Generates forensic reports explaining *why* a connection is flagged.
* **Zero-Day Audit:** Proven detection of unseen attack vectors via heuristic reasoning.


## 📈 Visualizations
![Model Comparison](./model_comparison_bar.png)
![Confusion Matrix](./confusion_matrices_grid.png)