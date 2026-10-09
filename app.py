from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

from src.config import load_config
from src.document_loader import load_documents
from src.chunking import create_chunks
from src.embedding import create_embedding_model, create_embeddings
from src.retrieval import create_faiss_index, retrieve
from src.llm import create_llm_client, generate_answer


# =========================================================
# Environment
# =========================================================

load_dotenv()


# =========================================================
# Configuration
# =========================================================

config = load_config("config.yaml")

documents_directory = Path(
    config["documents"]["directory"]
)

documents_pattern = config["documents"]["file_pattern"]

embedding_model_name = config["embedding"]["model"]

top_k = config["retrieval"]["top_k"]

llm_model_name = config["llm"]["model"]

llm_temperature = config["llm"]["temperature"]

chunk_size = config["chunking"]["chunk_size"]

chunk_overlap = config["chunking"]["overlap"]

base_url = os.getenv("LLM_BASE_URL")

api_key = os.getenv("LLM_API_KEY")


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Travel RAG v2",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Travel Policy Assistant")

st.caption(
    "RAG demo with document selection and conflict awareness"
)


# =========================================================
# Cached resources (models and client only)
# =========================================================

@st.cache_resource
def get_embedding_model():
    return create_embedding_model(embedding_model_name)


@st.cache_resource
def get_llm_client():
    return create_llm_client(
        base_url=base_url,
        api_key=api_key,
    )


# =========================================================
# Build index for a given set of documents
# =========================================================

@st.cache_data
def build_index(selected_docs, chunk_size_val, chunk_overlap_val):
    documents = []
    for doc_name in selected_docs:
        path = documents_directory / doc_name
        text = path.read_text(encoding="utf-8")
        documents.append({"source": doc_name, "text": text})

    chunks = create_chunks(
        documents,
        chunk_size=chunk_size_val,
        overlap=chunk_overlap_val,
    )

    embeddings = create_embeddings(
        get_embedding_model(), chunks
    )

    index = create_faiss_index(embeddings)

    return documents, chunks, index


# =========================================================
# Sidebar
# =========================================================

all_documents = sorted(
    p.name for p in documents_directory.glob(documents_pattern)
)

with st.sidebar:

    st.header("RAG Settings")

    chunk_size_selected = st.slider(
        "Chunk size",
        min_value=100,
        max_value=2000,
        value=chunk_size,
        step=100,
    )

    chunk_overlap_selected = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=chunk_size_selected,
        value=min(chunk_overlap, chunk_size_selected),
        step=50,
    )

    top_k_selected = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=top_k,
    )

    temperature_selected = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(llm_temperature),
        step=0.1,
    )

    st.divider()

    st.subheader("Corpus")

    selected_documents = st.multiselect(
        "Documents to include",
        options=all_documents,
        default=all_documents,
    )

    st.write(
        f"**{len(selected_documents)} / {len(all_documents)}" 
        f" documents selected**"
    )

    st.divider()

    st.write(
        f"**Embedding:** "
        f"`{embedding_model_name}`"
    )

    llm_client = get_llm_client()

    available_models = [
        model.id for model in llm_client.models.list().data
    ]

    llm_model_selected = st.selectbox(
        "LLM Model",
        options=available_models,
        index=available_models.index(llm_model_name)
            if llm_model_name in available_models
            else 0,
    )

    st.divider()

    st.subheader("System Prompt")

    default_system_prompt = """You are a corporate travel policy assistant.
Answer the user's question using ONLY the information
contained in the context below.

IMPORTANT RULES:
1. If the retrieved documents contain conflicting information
   (e.g. different numbers, different limits, or different rules
   for the same topic), you MUST:
   - Explicitly state that the documents contain conflicting
     information.
   - Present BOTH (or all) versions clearly.
   - Attribute each version to its source document.
   - Do NOT silently choose one over the other.
   - Recommend that the user contact the Travel & Expense team
     for clarification.

2. If the answer cannot be found in the context, say:
   "I don't know based on the provided documents."

3. Do not invent information."""

    system_prompt = st.text_area(
        "Edit the system prompt to add constraints",
        value=default_system_prompt,
        height=300,
        key="system_prompt",
    )

    with st.expander("Prompt constraint examples"):
        st.markdown("""
**Ignore personal travel reports:**
```
IMPORTANT: Ignore any documents that are personal travel
reports, personal accounts, or contain personal information
about specific individuals. Only use official corporate
policy documents and procedural guides.
```

**Prefer the most recent document:**
```
If two documents conflict, prefer the one that is explicitly
marked as an update or supplement.
```

**Escalate on conflict:**
```
If the documents contain contradictory information, do not
attempt to resolve it. Tell the user that conflicting
information was found and they should contact the Travel
& Expense team.
```
""")


# =========================================================
# Build index based on selection
# =========================================================

if not selected_documents:
    st.warning("Select at least one document to use the assistant.")
    st.stop()

documents, chunks, index = build_index(
    tuple(selected_documents),
    chunk_size_selected,
    chunk_overlap_selected,
)

embedding_model = get_embedding_model()


# =========================================================
# Main interface
# =========================================================

question = st.text_input(
    "Ask a question about business travel",
    placeholder=(
        "e.g. What is the maximum meal allowance?"
    ),
)


# =========================================================
# Process question
# =========================================================

if question:

    with st.spinner("Searching documents..."):

        try:

            # -------------------------------------------------
            # Retrieve relevant chunks
            # -------------------------------------------------

            results = retrieve(
                question=question,
                embedding_model=embedding_model,
                index=index,
                chunks=chunks,
                top_k=top_k_selected,
            )

            # -------------------------------------------------
            # Generate answer with conflict-aware prompt
            # -------------------------------------------------

            context = "\n\n".join(
                f"[Source: {result['source']}]\n"
                f"{result['text']}"
                for result in results
            )

            prompt = f"""You are a corporate travel policy assistant.
Answer the user's question using ONLY the information
contained in the context below.

IMPORTANT RULES:
1. If the retrieved documents contain conflicting information
   (e.g. different numbers, different limits, or different rules
   for the same topic), you MUST:
   - Explicitly state that the documents contain conflicting
     information.
   - Present BOTH (or all) versions clearly.
   - Attribute each version to its source document.
   - Do NOT silently choose one over the other.
   - Recommend that the user contact the Travel & Expense team
     for clarification.

2. If the answer cannot be found in the context, say:
   "I don't know based on the provided documents."

3. Do not invent information.

Context:
----------------
{context}
----------------

Question:
{question}
"""

            response = llm_client.chat.completions.create(
                model=llm_model_selected,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                temperature=temperature_selected,
            )

            answer = response.choices[0].message.content

        except Exception as e:

            st.error(
                "Something went wrong. "
                "Make sure LM Studio is running "
                "and the model is loaded."
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # Answer
    # =====================================================

    st.subheader("Answer")

    st.write(answer)


    # =====================================================
    # Retrieved documents
    # =====================================================

    st.subheader("Retrieved Documents")

    for i, result in enumerate(
        results,
        start=1,
    ):

        with st.expander(
            f"{i}. {result['source']} "
            f"— similarity "
            f"{result['score']:.3f}"
        ):

            st.write(
                f"**Chunk:** "
                f"{result['chunk_id']}"
            )

            st.write(
                result["text"]
            )


# =========================================================
# Example questions
# =========================================================

else:

    st.info(
        "Ask a question to search "
        "the travel documents."
    )

    st.subheader(
        "Try these questions"
    )

    example_questions = [
        "What is the maximum hotel rate I can book without manager approval?",
        "Do I need manager approval to book the Central Paris Hotel?",
        "What is the maximum daily meal allowance?",
        "After how many hours of flight can I book business class?",
        "How many checked bags does the company reimburse?",
        "How many days do I have to submit my expense claim?",
        "What is the tip limit at restaurants?",
        "What is the fare difference cap for choosing a nonstop flight?",
    ]

    for example in example_questions:

        st.markdown(
            f"- {example}"
        )