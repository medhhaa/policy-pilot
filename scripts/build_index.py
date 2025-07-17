import os, sys
# Add the project root (one level up) to Python’s search path
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from policy_pilot.retrieval import build_faiss_index




def main():
    build_faiss_index(limit = 5)  

if __name__ == "__main__":
    main()