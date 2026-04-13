# ============================================================
#  Construction Plan Analyzer Pro
#  Compatible with Streamlit 1.27+
#  Fixes:
#    - No Streamlit calls inside background threads (ScriptRunContext)
#    - No width="stretch" (uses use_container_width for old versions)
#    - No st.write_stream (uses manual streaming for old versions)
#    - Correct model names
# ============================================================

import fitz  # PyMuPDF
import base64
import streamlit as st
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import hashlib
import time
from datetime import datetime

# ---------- MUST BE FIRST ----------
st.set_page_config(page_title="Construction Plan Analyzer Pro", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 384
MAX_WORKERS   = 10
BATCH_DELAY   = 0.5

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Construction Plan Analyzer. Your task is to extract EVERY measurable and descriptive value from the provided construction drawing page, **page by page**. For each page, produce a structured output that lists each extracted value in a separate row.

**Mandatory extraction categories (include ALL that are visible):**

1. **Dimensions & Measurements** – every linear dimension (length, width, height, depth, spacing), area (room, slab, site), volume, angle, slope.
2. **Materials** – concrete grades (e.g., 4000 psi), rebar sizes and spacing (e.g., 16mmØ @ 300mm CRS), steel sections, masonry types, roofing, finishes.
3. **Structural Elements** – foundation depth/width, column size/spacing, beam dimensions, slab thickness, reinforcement details (bar size, spacing, count).
4. **Architectural Elements** – room names, floor levels, ceiling heights, door/window sizes, stair tread/riser dimensions.
5. **Site Data** – lot boundaries, setbacks, driveway dimensions, parking spaces, landscape areas, utility locations.
6. **MEP Systems** – pipe sizes, downpipe locations, manhole references, tank dimensions (if visible).
7. **Textual Data** – drawing number, sheet number, scale, date, revision, general notes, abbreviations.

**Output format (MUST follow):**

For each page, start with:

## Page <page_number> – <drawing_title> (sheet <sheet_no>)

Then a table with columns:

| Parameter | Value | Unit | Drawing Reference | Notes |
|-----------|-------|------|-------------------|-------|

**Rules:**
- One row per **discrete value** (e.g., one row for “Bedroom 1 width”, another row for “Bedroom 1 length”).
- If a value is missing or illegible, write `[missing]` in the Value column and explain in Notes.
- Convert all imperial values to metric in parentheses (e.g., `12'-0" (3.658m)`).
- Include the drawing reference (e.g., “A-2.03”, “Detail 3”) if visible.
- If a page contains no construction data, output: `No construction data found on this page.`

**Example row:**
| Parameter | Value | Unit | Drawing Reference | Notes |
|-----------|-------|------|-------------------|-------|
| Bedroom 1 – width | 12'-0" (3.658m) | feet (m) | A-2.03 | from garage level plan |
| Slab thickness | 150mm | mm | S-08 | typical upper roof slab |

Now analyse the provided page and extract EVERY value accordingly."""

FIRST_QUERY = """Analyse this construction plan page **page by page**. Extract **every single measurable and descriptive value** as per the system prompt.

Follow this strict structure:

## Page <number> – <title> (sheet <ref>)

| Parameter | Value | Unit | Drawing Reference | Notes |
|-----------|-------|------|-------------------|-------|

**Required extractions (if present on this page):**
- All linear dimensions (including dimensions inside detail circles)
- All area values (room areas, slab areas, site areas)
- All material specifications (rebar: size, spacing, grade; concrete: MPa/psi; masonry: block type)
- All structural element sizes (beam width/depth, column width/depth, foundation thickness)
- All architectural room labels with their measured dimensions
- All elevation heights and level differences
- All pipe diameters and plumbing fixture references
- Any text note that contains a numerical value or specification

**If a dimension is shown but the text is unclear, state "unclear" and describe its location (e.g., "dimension near north wall").**

Proceed now."""

# ── Embedding model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# ── PDF helpers ───────────────────────────────────────────────────────────────
def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """Returns True if the PDF appears to be image-based (scanned)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts, img_ratios = [], []
        for i in range(min(3, len(doc))):
            page = doc[i]
            texts.append(len(page.get_text().strip()))
            area = page.rect.width * page.rect.height
            img_area = 0
            for img in page.get_images():
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    img_area += pix.width * pix.height
                except Exception:
                    pass
            img_ratios.append(img_area / area if area > 0 else 0)
        doc.close()
        return float(np.mean(texts)) < 100 or float(np.mean(img_ratios)) > 0.7
    except Exception:
        return True


def pdf_page_to_b64(pdf_bytes: bytes, page_num: int, dpi: int = 300) -> str:
    """Rasterise one PDF page to a base64-encoded JPEG string."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    jpg = pix.tobytes("jpeg")
    doc.close()
    return base64.b64encode(jpg).decode("utf-8")


def extract_single_page_pdf_b64(pdf_bytes: bytes, page_num: int) -> str:
    """Return a base64-encoded single-page PDF."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst = fitz.open()
    dst.insert_pdf(src, from_page=page_num, to_page=page_num)
    data = dst.tobytes()
    src.close()
    dst.close()
    return base64.b64encode(data).decode("utf-8")


# ── Page analysis (pure functions – no Streamlit calls) ───────────────────────
def _analyse_page_vision(pdf_bytes: bytes, page_num: int, dpi: int,
                          api_key: str, model: str,
                          local_cache: dict) -> Dict:
    """Analyse one page via vision. Uses local_cache dict (not st.session_state)."""
    cache_key = hashlib.md5(
        f"{hashlib.md5(pdf_bytes[:10000]).hexdigest()}_{page_num}_{dpi}_initial".encode()
    ).hexdigest()

    if cache_key in local_cache:
        return {"page_num": page_num, "content": local_cache[cache_key],
                "success": True, "cache_key": cache_key, "from_cache": True}

    try:
        client = anthropic.Anthropic(api_key=api_key)
        img_b64 = pdf_page_to_b64(pdf_bytes, page_num, dpi)
        msg = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            # max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": FIRST_QUERY}
            ]}]
        )
        text = msg.content[0].text
        return {"page_num": page_num, "content": text,
                "success": True, "cache_key": cache_key, "from_cache": False}
    except Exception as e:
        return {"page_num": page_num, "content": f"Error: {e}",
                "success": False, "cache_key": cache_key, "from_cache": False}


def _analyse_page_native(pdf_bytes: bytes, page_num: int,
                          api_key: str, model: str) -> Dict:
    """Analyse one page via native PDF document upload."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        page_b64 = extract_single_page_pdf_b64(pdf_bytes, page_num)
        msg = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            # max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": page_b64}},
                {"type": "text", "text": FIRST_QUERY}
            ]}]
        )
        return {"page_num": page_num, "content": msg.content[0].text,
                "success": True, "cache_key": None, "from_cache": False}
    except Exception as e:
        return {"page_num": page_num, "content": f"Error: {e}",
                "success": False, "cache_key": None, "from_cache": False}


def _worker(args: tuple) -> Dict:
    """Thread worker – NO Streamlit calls allowed here."""
    page_num, pdf_bytes, use_vision, dpi, api_key, model, local_cache = args
    if use_vision:
        return _analyse_page_vision(pdf_bytes, page_num, dpi, api_key, model, local_cache)
    else:
        return _analyse_page_native(pdf_bytes, page_num, api_key, model)


def process_all_pages(pdf_bytes: bytes, page_numbers: List[int],
                       use_vision: bool, dpi: int,
                       api_key: str, model: str) -> List[Dict]:
    """
    Parallel page processing.
    Reads from (and writes to) st.session_state.analysis_cache ONLY in the
    main thread – never inside workers.
    """
    # Snapshot the cache into a plain dict to pass safely to threads
    local_cache: dict = dict(st.session_state.get("analysis_cache", {}))

    total   = len(page_numbers)
    results = []

    progress_bar = st.progress(0)
    status_box   = st.empty()

    thread_args = [
        (pn, pdf_bytes, use_vision, dpi, api_key, model, local_cache)
        for pn in page_numbers
    ]

    processed = 0
    for batch_start in range(0, total, MAX_WORKERS):
        batch_end  = min(batch_start + MAX_WORKERS, total)
        batch_args = thread_args[batch_start:batch_end]

        status_box.write(f"Processing pages {batch_start + 1}–{batch_end} of {total}…")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_worker, a): a for a in batch_args}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                processed += 1
                progress_bar.progress(processed / total)

                # Write new results back to cache in the MAIN thread
                if res.get("cache_key") and not res.get("from_cache") and res["success"]:
                    if "analysis_cache" not in st.session_state:
                        st.session_state.analysis_cache = {}
                    st.session_state.analysis_cache[res["cache_key"]] = res["content"]
                    local_cache[res["cache_key"]] = res["content"]

        if batch_end < total:
            time.sleep(BATCH_DELAY)

    status_box.empty()
    progress_bar.empty()
    return results


# ── Vector store ──────────────────────────────────────────────────────────────
def build_vector_store(pages_data: List[Dict]):
    good = [p for p in pages_data if p["success"]]
    if not good:
        return None, []

    texts      = [f"Page {p['page_num']}: {p['content']}" for p in good]
    embeddings = embedder.encode(texts, convert_to_numpy=True).astype(np.float32)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    metadata = [{"page_num": p["page_num"], "content": p["content"]} for p in good]
    return index, metadata


def get_relevant_pages(query: str, index, metadata: List[Dict], k: int = 5) -> List[Dict]:
    if index is None or not metadata:
        return []
    k = min(k, len(metadata))
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, idxs = index.search(q_emb, k)
    return [metadata[i] for i in idxs[0] if 0 <= i < len(metadata)]


# ── Streaming generators (no Streamlit calls inside) ─────────────────────────
def gen_rag_fast(query: str, index, metadata: List[Dict],
                  client, model: str, k: int):
    if index is None or not metadata:
        yield "Error: index not ready."
        return

    pages = get_relevant_pages(query, index, metadata, k)
    if not pages:
        yield "No relevant pages found for this query."
        return

    context = "\n\n".join(
        f"--- PAGE {p['page_num']} ---\n{p['content']}" for p in pages
    )
    prompt = (
        f"Based on the following construction plan extracts, answer the question.\n"
        f"Cite page numbers.\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Give a concise answer with specific measurements and drawing references."
    )
    try:
        with client.messages.stream(
            model=model, 
            # max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk
    except Exception as e:
        yield f"\n\nError: {e}"


def gen_rag_deep(query: str, relevant_pages: List[Dict],
                  pdf_bytes: bytes, client, model: str, dpi: int = 150):
    if not relevant_pages:
        yield "No relevant pages found."
        return

    blocks = [{"type": "text", "text": (
        f"User Query: {query}\n\n"
        f"Analysing {len(relevant_pages)} relevant page(s). "
        "Answer accurately with measurements and page references."
    )}]
    for p in relevant_pages:
        blocks.append({"type": "text",
                        "text": f"\n--- PAGE {p['page_num']} (prior extract) ---\n{p['content'][:300]}…"})
        try:
            img_b64 = pdf_page_to_b64(pdf_bytes, p["page_num"], dpi)
            blocks.append({"type": "image",
                            "source": {"type": "base64",
                                       "media_type": "image/jpeg",
                                       "data": img_b64}})
        except Exception as e:
            blocks.append({"type": "text", "text": f"[Image unavailable: {e}]"})

    blocks.append({"type": "text", "text": (
        "\nProvide a detailed answer with specific measurements. "
        "If the image contradicts the prior extract, trust the image."
    )})
    try:
        with client.messages.stream(
            model=model, system=SYSTEM_PROMPT, 
            # max_tokens=4096,
            messages=[{"role": "user", "content": blocks}]
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk
    except Exception as e:
        yield f"\n\nError: {e}"


def gen_detailed(query: str, pages_data: List[Dict], pdf_bytes: bytes,
                  client, model: str, use_vision: bool, dpi: int):
    """Yields the *fully accumulated* markdown string (for st.empty().markdown)."""
    accumulated = (
        f"# 📋 Detailed Analysis Report\n"
        f"**Query:** {query}  \n"
        f"**Pages:** {len(pages_data)}  \n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    )
    yield accumulated

    for p in pages_data:
        pn     = p["page_num"]
        header = f"## Page {pn}\n\n"
        body   = ""
        try:
            if use_vision:
                img_b64  = pdf_page_to_b64(pdf_bytes, pn, dpi)
                messages = [{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": f"Page {pn}: {query}"}
                ]}]
            else:
                page_b64 = extract_single_page_pdf_b64(pdf_bytes, pn)
                messages = [{"role": "user", "content": [
                    {"type": "document",
                     "source": {"type": "base64", "media_type": "application/pdf", "data": page_b64}},
                    {"type": "text", "text": f"Page {pn}: {query}"}
                ]}]

            with client.messages.stream(
                model=model, system=SYSTEM_PROMPT,
                # max_tokens=2048, 
                messages=messages
            ) as stream:
                for chunk in stream.text_stream:
                    body += chunk
                    yield accumulated + header + body + "\n\n---\n\n"

            accumulated += header + body + "\n\n---\n\n"
        except Exception as e:
            accumulated += header + f"Error: {e}\n\n---\n\n"
            yield accumulated


# ── Streaming display helper (works on all Streamlit versions) ────────────────
def stream_to_placeholder(generator, placeholder):
    """
    Consumes a text-chunk generator and renders it live using st.empty().
    Returns the full accumulated string.
    Falls back gracefully if st.write_stream is unavailable.
    """
    full = ""
    for chunk in generator:
        full += chunk
        placeholder.markdown(full + "▌")
    placeholder.markdown(full)
    return full


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏗️ Construction Plan Analyzer Pro")
st.markdown(
    "Upload architectural PDFs for comprehensive extraction of "
    "measurements, materials, and specifications."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-…")

    model = st.selectbox(
        "Claude Model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"],
        index=0
    )

    st.subheader("Processing Mode")
    processing_mode = st.radio(
        "PDF Processing Mode",
        ["Auto-Detect (Recommended)",
         "Vision Only (For Scanned Plans)",
         "Native PDF (For Digital Plans)"]
    )

    st.subheader("Analysis Mode")
    analysis_mode = st.radio("Query Mode",
                              ["RAG (Smart Retrieval)", "Detailed (All Pages)"])

    use_deep_rag = False
    if analysis_mode == "RAG (Smart Retrieval)":
        st.subheader("RAG Strategy")
        rag_mode     = st.radio("Choose RAG Mode",
                                ["🚀 Fast (Text Only)", "🔍 Deep (Visual Analysis)"])
        use_deep_rag = rag_mode == "🔍 Deep (Visual Analysis)"

    with st.expander("Advanced Options"):
        dpi_setting  = st.slider("Image DPI", 100, 200, 300)
        max_pages    = st.number_input("Max Pages to Process", 1, 600, 100)
        context_pages = st.slider("Context Pages (RAG)", 1, 10, 3)

    # Clear button — use_container_width works on all versions
    if st.button("🗑️ Clear Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── API key gate ──────────────────────────────────────────────────────────────
if not api_key:
    st.warning("Please enter your Anthropic API key in the sidebar.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# ── Session state defaults ────────────────────────────────────────────────────
defaults = dict(
    processed=False, pages_data=[], index=None, metadata=[],
    chat_history=[], is_scanned=False, pdf_bytes=None,
    file_hash=None, analysis_cache={}, pending_query=None
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
#  UPLOAD & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.processed:
    uploaded = st.file_uploader("📄 Upload Construction PDF", type=["pdf"])

    if uploaded:
        t0        = time.time()
        pdf_bytes = uploaded.read()
        file_hash = hashlib.md5(pdf_bytes).hexdigest()

        # Reset if a different file is uploaded
        if st.session_state.file_hash != file_hash:
            st.session_state.analysis_cache = {}

        st.session_state.pdf_bytes  = pdf_bytes
        st.session_state.file_hash  = file_hash

        with st.spinner("🔍 Analysing PDF structure…"):
            scanned = is_scanned_pdf(pdf_bytes)
            st.session_state.is_scanned = scanned

            if processing_mode == "Auto-Detect (Recommended)":
                use_vision = scanned
            elif processing_mode == "Vision Only (For Scanned Plans)":
                use_vision = True
            else:
                use_vision = False

            mode_label = "Vision" if use_vision else "Native PDF"
            st.info(f"{'📸 Scanned' if scanned else '📄 Digital'} PDF detected → using **{mode_label}** mode.")

            doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = min(len(doc), int(max_pages))
            doc.close()

            pages_data = process_all_pages(
                pdf_bytes    = pdf_bytes,
                page_numbers = list(range(total_pages)),
                use_vision   = use_vision,
                dpi          = dpi_setting,
                api_key      = api_key,
                model        = model,
            )
            pages_data.sort(key=lambda x: x["page_num"])
            st.session_state.pages_data = pages_data

            index, metadata = build_vector_store(pages_data)
            st.session_state.index    = index
            st.session_state.metadata = metadata
            st.session_state.processed = True

            elapsed   = time.time() - t0
            successful = sum(1 for p in pages_data if p["success"])
            st.success(f"✅ Processed {successful}/{total_pages} pages in {elapsed:.1f}s")

            with st.expander("📊 Processing details"):
                st.write(f"- Successful pages: {successful}/{total_pages}")
                st.write(f"- Index entries: {len(metadata)}")
                st.write(f"- Elapsed: {elapsed:.1f}s")

            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Status:** ✅ Ready")
        st.markdown(f"**Pages indexed:** {len(st.session_state.metadata)}")

    st.markdown("---")
    st.subheader("💬 Ask About Your Construction Plans")

    # ── Render history ────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("pages"):
                st.caption("Referenced: " + ", ".join(f"p{p+1}" for p in msg["pages"]))
                if not msg.get("fast_mode"):
                    cols = st.columns(min(len(msg["pages"]), 3))
                    for i, pn in enumerate(msg["pages"][:3]):
                        with cols[i]:
                            try:
                                img_b64 = pdf_page_to_b64(
                                    st.session_state.pdf_bytes, pn, dpi=200)
                                st.image(base64.b64decode(img_b64),
                                         caption=f"Page {pn}",
                                         use_container_width=True)
                            except Exception:
                                pass

    # ── Determine active query ────────────────────────────────────────────────
    current_query: Optional[str] = None
    user_input = st.chat_input("Ask about measurements, materials, elevations…")
    if user_input:
        current_query = user_input
    if st.session_state.pending_query:
        current_query = st.session_state.pending_query
        st.session_state.pending_query = None

    # ── Process query ─────────────────────────────────────────────────────────
    if current_query:
        st.session_state.chat_history.append({"role": "user", "content": current_query})
        with st.chat_message("user"):
            st.markdown(current_query)

        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
            pages_used: List[int] = []
            fast_mode     = False

            try:
                if not st.session_state.pdf_bytes:
                    full_response = "❌ PDF data missing. Please re-upload."
                    placeholder.error(full_response)

                elif analysis_mode == "Detailed (All Pages)":
                    full_response = stream_to_placeholder(
                        gen_detailed(
                            current_query,
                            st.session_state.pages_data,
                            st.session_state.pdf_bytes,
                            client, model,
                            use_vision=st.session_state.is_scanned,
                            dpi=dpi_setting
                        ),
                        placeholder
                    )
                    pages_used = [p["page_num"] for p in st.session_state.pages_data]

                elif use_deep_rag:
                    rel_pages  = get_relevant_pages(
                        current_query, st.session_state.index,
                        st.session_state.metadata, k=context_pages)
                    pages_used = [p["page_num"] for p in rel_pages]
                    full_response = stream_to_placeholder(
                        gen_rag_deep(
                            current_query, rel_pages,
                            st.session_state.pdf_bytes,
                            client, model, dpi=150
                        ),
                        placeholder
                    )

                else:  # Fast RAG
                    fast_mode  = True
                    rel_pages  = get_relevant_pages(
                        current_query, st.session_state.index,
                        st.session_state.metadata, k=context_pages)
                    pages_used = [p["page_num"] for p in rel_pages]
                    full_response = stream_to_placeholder(
                        gen_rag_fast(
                            current_query,
                            st.session_state.index,
                            st.session_state.metadata,
                            client, model, k=context_pages
                        ),
                        placeholder
                    )

            except Exception as e:
                full_response = f"❌ Error: {e}"
                placeholder.error(full_response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "pages": pages_used,
                "fast_mode": fast_mode
            })

        st.rerun()

    # ── Quick action buttons ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Quick Extraction")

    quick_queries = {
        "📐 All Measurements": (
            "Extract ALL numerical measurements, dimensions, and areas from the entire document. "
            "Present in tables by category: Site, Building, Rooms, Structural."
        ),
        "🧱 Materials": (
            "List all material specs: concrete grades, steel sizes, timber dims, "
            "roofing materials, and finishes with quantities where shown."
        ),
        "🏗️ Structural": (
            "Extract all structural info: foundation depths, column sizes, beam spans, "
            "slab thicknesses, and reinforcement details."
        ),
        "📏 Elevations": (
            "Summarise all elevation drawings: heights, floor levels, "
            "and all vertical dimensions."
        ),
    }

    cols = st.columns(len(quick_queries))
    for col, (label, query) in zip(cols, quick_queries.items()):
        with col:
            # use_container_width is available in all Streamlit versions
            if st.button(label, use_container_width=True, key=f"qb_{label}"):
                st.session_state.pending_query = query
                st.rerun()

# ── Help ──────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How to Use"):
    st.markdown("""
**Modes:**
- **Fast RAG** — retrieves relevant pages from the pre-extracted text, answers in ~1-2 s
- **Deep RAG** — re-opens the actual page images for the relevant pages, more accurate (~5-10 s)
- **Detailed** — analyses every page sequentially, best for full-document reports

**Tips:**
- Use Fast mode for quick lookups; Deep mode for precise measurements
- 300 DPI gives a good balance of quality and speed
- Clear Session resets everything so you can upload a new PDF
""")