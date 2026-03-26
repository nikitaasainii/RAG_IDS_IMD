import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load the .env file from the root directory
# This pulls your GROQ_API_KEY safely into the script
load_dotenv()

# 2. Project Root Directory (Calculates the path to your RAG_IDS_Comparison folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. API & Model Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"  # The latest Llama 3.1 version on Groq
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# 4. File Paths
# We use .arff here since your previous code snippet was for ARFF files
TRAIN_DATA_PATH = BASE_DIR / "data" / "KDDTrain+.arff"
TEST_DATA_PATH = BASE_DIR / "data" / "KDDTest+.arff"
CHROMA_PATH = BASE_DIR / "chroma_db"

# 5. NSL-KDD Column Names (Standard 43-column structure)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]