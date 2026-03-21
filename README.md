# Retrieval-Augmented Generation (RAG)

Source: https://huggingface.co/blog/ngxson/make-your-own-rag

## Step 1 — Install Python 3.11 via Homebrew
<pre>
brew install python@3.11 
</pre>

## Step 2 — Verify It Installed
<pre>
python3.11 --version
</pre>

# Step 3 — Create Fresh Clean Venv
<pre>
deactivate
cd ~/Desktop/rag
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate  
</pre>

# Step 4 — Install Everything Fresh
<pre>
pip install ollama rank_bm25 streamlit chromadb
</pre>

# Load models
<pre>
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
</pre>

<pre>
# Terminal chatbot
python3 rag_app.py

# Streamlit UI
streamlit run rag_app.py
</pre>


Four Ways To Run
<pre>
# Normal chatbot loop
python3 rag_app.py

# Agent mode (terminal)
python3 rag_app.py --agent

# Benchmark
python3 rag_app.py --benchmark

# Streamlit UI (includes both chat + agent mode toggle)
streamlit run rag_app.py

</pre>

<pre>
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
</pre>

For final query:
<pre>
python demo.py
</pre>


Four ways to run
<pre>
  # Normal chatbot loop
python3 rag_app.py

# Agent mode (terminal)
python3 rag_app.py --agent

# Benchmark
python3 rag_app.py --benchmark

# Streamlit UI (includes both chat + agent mode toggle)
streamlit run rag_app.py
</pre>
