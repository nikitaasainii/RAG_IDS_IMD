import pandas as pd
from src.vector_store import VectorStore
from src.inference import RAGInference
from src.preprocessing import row_to_string

def test_custom_packet():
    # 1. Initialize your Project's "Brain"
    vdb = VectorStore()
    rag = RAGInference()

    # 2. Create a "Base" Normal Packet (Template)
    # We use common 'Normal' values so we only have to change the 'Attack' features
    custom_packet = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 200, 'dst_bytes': 500, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0, 'count': 1, 'srv_count': 1,
        'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 1, 'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }

    # 3. 🛠️ MODIFY the packet to simulate an attack
    # Let's simulate a 'Neptune' (DoS) attack by spiking the error rates
    print("\n💉 Injecting Malicious Features (Simulating DoS)...")
    custom_packet['service'] = 'private'
    custom_packet['serror_rate'] = 1.0
    custom_packet['srv_serror_rate'] = 1.0
    custom_packet['count'] = 250  # High connection count

    # 4. Convert to Series (Matches your preprocessing format)
    manual_row = pd.Series(custom_packet)

    # 5. RUN THE BUILT-IN INFERENCE
    print("🔎 Searching 125k records for similar behavior...")
    context = vdb.query_similar_cases(manual_row)
    
    print("🧠 AI Reasoning in progress...")
    report = rag.generate_analysis(manual_row, context)

    print("\n🛑 --- CUSTOM PACKET ANALYSIS ---")
    print(report)
    print("-" * 50)

if __name__ == "__main__":
    test_custom_packet()