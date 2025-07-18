import streamlit as st
import os
import sys

# Add the project root to Python's search path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from policy_pilot.retrieval import query_faiss
from policy_pilot.llm_agent import answer_query

# Page configuration
st.set_page_config(
    page_title="Policy Pilot",
    page_icon="🛡️",
    layout="centered"
)

# Simple CSS
st.markdown("""
<style>
    .answer-box {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }

    .confidence-badge {
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        font-weight: bold;
    }

    .confidence-high { background-color: #d4edda; color: #155724; }
    .confidence-medium { background-color: #fff3cd; color: #856404; }
    .confidence-low { background-color: #f8d7da; color: #721c24; }

    /* Reduce padding/margins around the whole app */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.title("Policy Pilot")
st.markdown("AI-powered regulatory compliance guidance")

# Main query input
query = st.text_area(
    "Ask your compliance question:",
    placeholder="e.g., What HIPAA encryption requirements apply to patient data?",
    height=80
)

# Simple search button
if st.button("Get Answer", type="primary"):
    if query:
        with st.spinner("Analyzing..."):
            # Retrieve and generate answer
            context_chunks = query_faiss(query, top_k=25)
            response = answer_query(query, context_chunks)
            
            # # Display answer immediately
            # confidence = response["confidence"]
            # confidence_class = f"confidence-{confidence.lower()}"
            
            # st.markdown(f"""
            # <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
            #     <span class="confidence-badge {confidence_class}">Confidence: {confidence}</span>
            #     <span style="color: #666; font-size: 0.9em;">{len(response['sources'])} regulations • {response['context_chunks']} sources</span>
            # </div>
            # """, unsafe_allow_html=True)
            
            # # Main answer
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(response["answer"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Collapsible source details
            with st.expander("View source context"):
                for i, chunk in enumerate(context_chunks, 1):
                    st.markdown(f"**{chunk['id']}** (Score: {chunk['score']:.2f})")
                    st.markdown(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                    if i < len(context_chunks):
                        st.markdown("---")
    else:
        st.error("Please enter a question.")

# Example questions
st.markdown("---")
st.markdown("**Example questions:**")
examples = [
    "What training must be provided under CCPA?",
    "What rights do users have over their personal data?",
    "Who must be trained in handling private customer information?",
    "How should a company secure cardholder data?"
]

for example in examples:
    if st.button(example, key=f"example_{examples.index(example)}"):
        st.rerun()

# Footer
st.markdown("---")
st.caption("Supported: GDPR, HIPAA, SOX, CCPA, PCI-DSS | Always consult legal professionals for specific compliance decisions.")