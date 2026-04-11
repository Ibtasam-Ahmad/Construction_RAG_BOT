Here’s a **client-facing execution & testing README/manual** for your project. It’s written so a non-developer (or QA tester) can follow it step by step.

---

# 🏗️ Construction Plan Analysis Tool

### Client Execution & Testing Manual

---

## 📌 1. Overview

This application is a **Construction RAG (Retrieval-Augmented Generation) tool** built using:

* Streamlit (UI)
* FAISS (vector search)
* SentenceTransformers (text embeddings)
* Anthropic Claude API (AI analysis)

### 🎯 Purpose

The system allows users to:

* Upload construction PDF drawings/plans
* Automatically extract measurements & specifications
* Ask questions about materials, dimensions, and quantities
* Perform:

  * ⚡ Fast search (RAG mode)
  * 🔍 Full document deep analysis (Detailed mode)

---

## 🖥️ 2. System Requirements

Ensure the following are installed:

* Python **3.9+**
* pip (Python package manager)
* Internet connection (for API calls)

### 📦 Required Python Libraries

Install dependencies:

```bash
pip install streamlit pymupdf anthropic faiss-cpu sentence-transformers numpy
```

---

## 🔑 3. API Requirement

You MUST have:

* Claude API key from Anthropic

Format:

```
sk-ant-xxxxxxxxxxxx
```

---

## ▶️ 4. How to Run the Application

Navigate to the project folder and run:

```bash
streamlit run app.py
```

(Replace `app.py` with your file name if different)

### 🌐 Access the App

Once started, open:

```
http://localhost:8501
```

---

## 🧭 5. User Interface Guide

### 🔹 Sidebar (Left Panel)

#### 1. API Configuration

* Enter your **Claude API Key**
* Enter model name (default):

  ```
  claude-sonnet-4-6
  ```

#### 2. Analysis Mode Toggle

| Mode | Description                     |
| ---- | ------------------------------- |
| OFF  | ⚡ Fast RAG (recommended)        |
| ON   | 🔍 Detailed (slower, full scan) |

#### 3. Reset Button

* Clears:

  * Uploaded PDF
  * Index
  * Chat history

---

## 📄 6. Upload & Indexing Process

### Step-by-step:

1. Upload a **construction PDF**
2. System will:

   * Convert each page → image
   * Analyze each page using AI
   * Extract:

     * Square footage
     * Materials
     * Dimensions
3. Build a searchable vector index

### ⏳ Expected Time

| PDF Size  | Time       |
| --------- | ---------- |
| 5 pages   | ~30–60 sec |
| 20 pages  | ~2–5 min   |
| 50+ pages | 5–15 min   |

---

## 💬 7. How to Use (Testing Scenarios)

### ✅ Test Case 1: Basic Query

**Input:**

```
How much concrete is used in this project?
```

**Expected Output:**

* Concrete quantities
* Page references
* Supporting images

---

### ✅ Test Case 2: Material Breakdown

**Input:**

```
List all roofing types and their areas
```

**Expected Output:**

* Roofing categories:

  * Shingle
  * TPO
  * Metal
* Area measurements

---

### ✅ Test Case 3: No Data Case

**Input:**

```
What is the steel tonnage?
```

**Expected Output:**

```
No measurable data on this page.
```

OR clearly states absence

---

### ✅ Test Case 4: Detailed Mode

Enable:

```
🔍 Detailed Analysis = ON
```

**Input:**

```
Extract all dimensions from all pages
```

**Expected Output:**

* Page-by-page report
* Each page separately analyzed

---

## ⚙️ 8. System Behavior

### 🔹 RAG Mode (Default)

* Searches top **5 relevant pages**
* Faster response
* Uses embeddings + similarity search

### 🔹 Detailed Mode

* Scans **ALL pages**
* Slower but exhaustive
* Best for audits

---

## 📊 9. What the System Extracts

The AI is trained to extract:

* 📐 Dimensions (length, width, height)
* 🧱 Materials:

  * Concrete
  * Steel
  * Sheetrock
  * Roofing
* 📏 Area calculations
* 📝 Notes & annotations

---

## ⚠️ 10. Known Limitations

* Poor-quality scans → reduced accuracy
* Handwritten notes may not be detected
* Very large PDFs may take longer
* Non-construction queries → rejected

Example:

```
"I am a Construction Bot and can only assist with construction-related topics."
```

---

## 🧪 11. QA Testing Checklist

| Test           | Pass Criteria             |
| -------------- | ------------------------- |
| PDF upload     | File loads without error  |
| Index creation | All pages processed       |
| RAG query      | Relevant pages returned   |
| Detailed mode  | All pages analyzed        |
| Image display  | Correct page images shown |
| Error handling | Graceful messages         |

---

## 🐞 12. Common Errors & Fixes

### ❌ API Key Error

**Issue:**

```
Authentication failed
```

**Fix:**

* Verify API key format
* Ensure no extra spaces

---

### ❌ Module Not Found

**Fix:**

```bash
pip install -r requirements.txt
```

---

### ❌ FAISS Issues (Windows)

Use:

```bash
pip install faiss-cpu
```

---

## 🔄 13. Reset / Restart

To restart fresh:

* Click:

  ```
  🗑️ Clear Chat & Upload New PDF
  ```

OR

* Restart app:

```bash
CTRL + C
streamlit run app.py
```

---

## 📈 14. Performance Tips

* Use **RAG mode** for quick queries
* Use **Detailed mode** only when needed
* Keep PDFs under **100 pages** for optimal performance

---

## 📬 15. Support

If issues occur, provide:

* PDF sample
* Query used
* Screenshot of error
* Logs from terminal

---

## ✅ Summary

This tool enables:

* Automated construction plan understanding
* Fast material estimation
* AI-powered document querying
