# Retrieval-Augmented Generation (RAG)

Source: https://huggingface.co/blog/ngxson/make-your-own-rag

### Step 1 — Install Python 3.11 via Homebrew
<pre>
brew install python@3.11 
</pre>

### Step 2 — Verify It Installed
<pre>
python3.11 --version
</pre>

### Step 3 — Create Fresh Clean Venv
<pre>
deactivate
cd ~/Desktop/rag
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate  
</pre>

### Step 4 — Install Everything Fresh
<pre>
pip install ollama rank_bm25 streamlit chromadb
</pre>

### Load models
<pre>
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
ollama pull llama3.2:3b 
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
</pre>




### Four Ways To Run
<pre>
  
# Normal chatbot loop
python3 rag_app6.py

# Agent mode (terminal)
python3 rag_app6.py --agent

# Benchmark
python3 rag_app6.py --benchmark

# Streamlit UI (includes both chat + agent mode toggle)
<!-- streamlit run rag_app6.py -->
python3 -m streamlit run rag_app6.py
</pre>





