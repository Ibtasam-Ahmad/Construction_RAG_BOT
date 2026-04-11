import fitz  # PyMuPDF
import base64
import streamlit as st
import anthropic
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- MUST BE FIRST ----------
st.set_page_config(page_title="Construction RAG", layout="wide")

# Constants
EMBEDDING_DIM = 384
DEFAULT_MODEL = "claude-sonnet-4-6"

# ---------- Embedding Model (cached globally, shared across sessions) ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# ---------- System Prompt & First Query ----------
SYSTEM_PROMPT = """You are a Construction Bot with expertise in all aspects of construction and related topics. You can analyze and assist with:  
- Construction plans, blueprints, sketches, and specifications.  
- Materials selection, quantities, and costs, including concrete, steel, wood, and other building components.  
- Structural analysis, dimensions, load calculations, and engineering details.  
- Mechanical, Electrical, and Plumbing (MEP) systems.  
- Project phases, timelines, and schedules.  
- Construction codes, regulations, and standards compliance.  

When presented with a construction plan page, extract ALL relevant measurements, dimensions, material specifications, and notes. Be thorough and precise. If the page does not contain measurable data, state that clearly.

You will only respond to construction‑related queries. If asked about anything unrelated, reply: "I am a Construction Bot and can only assist with construction-related topics."
"""

FIRST_QUERY = """
Please review this construction plan page and provide a comprehensive extraction of all square footage / area measurements for:
1. Sheetrock
2. Concrete (slabs, foundations, etc.)
3. Roofing (with subtypes: Shingle, Modified bitumen, TPO, Metal R panel, Standing seam)
4. Structural steel

Also include any other relevant dimensions, material quantities, and specifications shown on this page.
If the page does not contain such data, respond with "No measurable data on this page."
"""

# ---------- PDF Processing ----------
def pdf_to_images(uploaded_file, output_dir, dpi=120):
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    image_paths = []
    for i in range(len(pdf_document)):
        page = pdf_document.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(output_dir, f'page_{i}.jpg')
        pix.save(img_path)
        image_paths.append(img_path)
    pdf_document.close()
    return image_paths

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ---------- Page Analysis (Prompt 1) ----------
def analyze_page_for_index(image_b64, page_num, client, model):
    try:
        message = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": FIRST_QUERY}
                ]
            }]
        )
        summary = message.content[0].text
        return page_num, summary, image_b64
    except Exception as e:
        return page_num, f"Error analyzing page: {str(e)}", image_b64

# ---------- Build Vector Store (in‑memory) ----------
def build_vector_store_from_data(page_data):
    texts = [f"Page {num}: {summary}" for num, summary, _ in page_data]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    metadata = [{"page_num": num, "summary": summary, "image": img} for num, summary, img in page_data]
    return index, metadata

# ---------- RAG Query (Prompt 2) ----------
def rag_query(user_query, index, metadata, client, model, k=5):
    q_emb = embedder.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(q_emb.astype(np.float32), k)
    relevant = [metadata[i] for i in indices[0] if i != -1]
    if not relevant:
        return "No relevant pages found.", []

    content_parts = []
    for item in relevant:
        content_parts.append({
            "type": "text",
            "text": f"--- Page {item['page_num']} Summary ---\n{item['summary']}"
        })
        content_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": item['image'],
            },
        })

    content_parts.append({
        "type": "text",
        "text": f"User Query: {user_query}\n\nUsing the above page summaries and images, provide a detailed, accurate answer. Include specific measurements and page references."
    })

    message = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=2048,
        messages=[{"role": "user", "content": content_parts}]
    )
    answer = message.content[0].text
    return answer, relevant

# ---------- Detailed Analysis (Prompt 3) ----------
def detailed_analysis(user_query, all_page_data, client, model):
    def process_single(page_num, image_b64):
        try:
            msg = client.messages.create(
                model=model,
                system=SYSTEM_PROMPT,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                        {"type": "text", "text": f"Page {page_num}:\n{user_query}"}
                    ]
                }]
            )
            return page_num, msg.content[0].text
        except Exception as e:
            return page_num, f"Error: {str(e)}"

    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(all_page_data)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single, num, img): num for num, _, img in all_page_data}
        for i, future in enumerate(as_completed(futures)):
            page_num, response = future.result()
            results[page_num] = response
            progress_bar.progress((i + 1) / total)
            status_text.write(f"✓ Page {page_num} processed ({i+1}/{total})")

    status_text.empty()
    progress_bar.empty()

    sorted_pages = sorted(results.items())
    report = f"# 📄 Detailed Page‑by‑Page Analysis for Query:\n> {user_query}\n\n"
    for num, resp in sorted_pages:
        report += f"## Page {num}\n{resp}\n\n---\n\n"
    return report

# ---------- Streamlit UI ----------
st.title("🏗️ Construction Plan Analysis (RAG + Detailed)")

# ---------- Sidebar: API Configuration ----------
with st.sidebar:
    st.header("🔑 API Configuration")
    user_api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your Anthropic API key (starts with 'sk-ant-')."
    )
    user_model = st.text_input(
        "Model Name",
        value=DEFAULT_MODEL,
        help="Claude model to use (e.g., claude-sonnet-4-6)."
    )

    st.divider()
    st.subheader("Analysis Mode")
    detailed_mode = st.toggle(
        "🔍 Detailed Analysis (process all pages)",
        value=False,
        help="When ON, each query will be run against EVERY page and a full page-by-page report is shown."
    )

    if st.button("🗑️ Clear Chat & Upload New PDF"):
        # Clear session state related to PDF/index
        for key in ["index_ready", "faiss_index", "metadata", "all_page_data", "chat_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Validate API key
if not user_api_key:
    st.warning("Please enter your Claude API key in the sidebar.")
    st.stop()

client = anthropic.Anthropic(api_key=user_api_key)

# Initialize session state for this session
if 'index_ready' not in st.session_state:
    st.session_state.index_ready = False
    st.session_state.faiss_index = None
    st.session_state.metadata = None
    st.session_state.all_page_data = []   # (page_num, summary, image_b64)
    st.session_state.chat_history = []

# PDF Upload (only if index not ready)
if not st.session_state.index_ready:
    uploaded_file = st.file_uploader("Upload a construction PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("📄 Processing PDF and building vector index... This may take a few minutes."):
            with tempfile.TemporaryDirectory() as tmpdir:
                image_paths = pdf_to_images(uploaded_file, tmpdir, dpi=100)
                total_pages = len(image_paths)
                st.info(f"PDF has {total_pages} pages.")

                page_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    for i, path in enumerate(image_paths):
                        img_b64 = encode_image(path)
                        futures.append(executor.submit(analyze_page_for_index, img_b64, i, client, user_model))

                    for idx, future in enumerate(as_completed(futures)):
                        page_num, summary, img_b64 = future.result()
                        page_data.append((page_num, summary, img_b64))
                        progress_bar.progress((idx + 1) / total_pages)
                        status_text.write(f"✓ Page {page_num} indexed ({idx+1}/{total_pages})")

                status_text.empty()
                progress_bar.empty()

                page_data.sort(key=lambda x: x[0])
                st.session_state.all_page_data = page_data

                # Build FAISS index in memory
                index, metadata = build_vector_store_from_data(page_data)
                st.session_state.faiss_index = index
                st.session_state.metadata = metadata
                st.session_state.index_ready = True

        st.success(f"✅ Indexing complete! {total_pages} pages processed.")
        st.rerun()
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "images" in msg:
                for img_b64, page_num in msg["images"]:
                    st.caption(f"Page {page_num}")
                    st.image(base64.b64decode(img_b64), use_column_width=True)   # <-- FIXED

    # Chat input
    user_query = st.chat_input("Ask about materials, measurements, or details...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if detailed_mode:
                    response = detailed_analysis(
                        user_query,
                        st.session_state.all_page_data,
                        client,
                        user_model
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    response, relevant_pages = rag_query(
                        user_query,
                        st.session_state.faiss_index,
                        st.session_state.metadata,
                        client,
                        user_model
                    )
                    st.markdown(response)
                    if relevant_pages:
                        st.divider()
                        st.caption("📄 Referenced pages:")
                        for item in relevant_pages:
                            st.caption(f"Page {item['page_num']}")
                            st.image(base64.b64decode(item['image']), use_column_width=True)   # <-- FIXED

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "images": [(item['image'], item['page_num']) for item in relevant_pages]
                    })
        st.rerun()