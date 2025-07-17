"""
build_index.py

Simple entry point to build the FAISS index without using `-c`.
"""
from policy_pilot.retrieval import build_faiss_index

def main():
    build_faiss_index(limit = 200)

if __name__ == "__main__":
    main()
