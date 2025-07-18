# Policy Pilot

A Retrieval-Augmented Generation (RAG) pipeline for compliance policy analysis and guidance. Policy Pilot helps organizations navigate complex regulatory requirements by providing intelligent, contextual answers from official compliance documents.

## Use Cases

- **Healthcare Organizations**: HIPAA compliance guidance for patient data handling and security requirements
- **Financial Services**: SOX compliance for internal controls and financial reporting automation
- **E-commerce & Tech**: GDPR and CCPA compliance for data privacy and user rights management
- **Payment Processing**: PCI-DSS requirements for secure credit card transaction handling
- **Compliance Teams**: Quick access to specific regulatory requirements without manual document searches
- **Legal & Risk Management**: Automated compliance gap analysis and policy recommendations

## Architecture

Policy Pilot implements a complete RAG pipeline with the following components:

```
PDFs → Text Extraction → Chunking → Embeddings → Vector Store → LLM Agent → Streamlit UI
```

### Core Components

1. **Document Processing**: Extracts text from compliance PDFs using pdfplumber with pdfminer fallback
2. **Text Chunking**: Splits documents into ~750 character chunks for optimal retrieval
3. **Embedding Generation**: Uses SentenceTransformers (all-MiniLM-L6-v2) for local embedding
4. **Vector Storage**: FAISS index for efficient similarity search
5. **LLM Agent**: Gemini 2.5 Pro for generating compliance guidance
6. **Web Interface**: Streamlit app for user interaction

## Supported Regulations

- **GDPR** (General Data Protection Regulation)
- **HIPAA** (Health Insurance Portability and Accountability Act)
- **SOX** (Sarbanes-Oxley Act)
- **CCPA** (California Consumer Privacy Act)
- **PCI-DSS** (Payment Card Industry Data Security Standard)

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/policy-pilot.git
   cd policy-pilot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download compliance documents**
   ```bash
   python scripts/pdf_fetcher.py
   ```

4. **Process and chunk documents**
   ```bash
   python scripts/pdf_chunker.py
   ```

5. **Build vector index**
   ```bash
   python scripts/build_index.py
   ```

6. **Launch the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## Project Structure

```
policy-pilot/
├── chunks/
│   └── chunks.json              # Processed document chunks
├── pdfs/
│   ├── CCPA.pdf                 # Downloaded compliance documents
│   ├── GDPR.pdf
│   ├── HIPAA.pdf
│   ├── PCI-DSS.pdf
│   └── SOX.pdf
├── policy_pilot/
│   ├── __init__.py
│   ├── embed_utils.py           # Embedding generation utilities
│   ├── llm_agent.py             # Gemini LLM integration
│   └── retrieval.py             # FAISS vector store operations
├── scripts/
│   ├── build_index.py           # Vector index construction
│   ├── list_models.py           # Available model listing
│   ├── pdf_chunker.py           # Document processing
│   ├── pdf_fetcher.py           # PDF downloading
│   └── test_llm_agent.py        # Testing utilities
├── streamlit_app.py             # Web interface
├── vector_store/
│   ├── faiss.index              # Vector similarity index
│   └── id_map.json              # Chunk ID mappings
└── requirements.txt
```

## Customization

### Adding New Regulations

1. Update `REGULATIONS` dictionary in `pdf_fetcher.py`
2. Run the document processing pipeline
3. Rebuild the vector index

### Adjusting Chunk Size

Modify `MAX_CHARS` in `pdf_chunker.py` to optimize for your use case:
- Smaller chunks: More precise retrieval, may lose context
- Larger chunks: More context, potentially less precise

### Tuning Retrieval

Adjust `top_k` parameter in queries to balance between comprehensive context (higher k) and focused responses (lower k).


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: Policy Pilot provides guidance based on publicly available regulatory documents. Always consult with qualified legal professionals for specific compliance decisions.