# Retrieval-Augmented Generation (RAG)

Source: https://huggingface.co/blog/ngxson/make-your-own-rag

Let's install the ollama package:
<pre>
pip install ollama
pip install chromadb rank_bm25 streamlit
</pre>

if pip does not work use brew for mac

After installed, open a terminal and run the following command to download the required models:

<pre>
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
</pre>

For final query:
<pre>
python demo.py
</pre>


For Demo App:
<pre>
python rag_app.py
# Normal chatbot
python3 rag_app.py

# Benchmarking mode
python3 rag_app.py --benchmark
</pre>



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


Step 1 — Install Python 3.11 via Homebrew
bashbrew install python@3.11
Step 2 — Verify It Installed
bashpython3.11 --version
Should show Python 3.11.x
Step 3 — Create Fresh Clean Venv
bashdeactivate
cd ~/Desktop/rag
python3.11 -m venv rag_env_311
source rag_env_311/bin/activate
Step 4 — Install Everything Fresh
bashpip install ollama rank_bm25 streamlit chromadb
Step 5 — Run
bash# Terminal chatbot
python3 rag_app.py

# Streamlit UI
streamlit run rag_app.py



Agent Tools Available
ToolWhat It Doesrag_search(query)Searches your documentscalculator(expression)Evaluates mathsummarise(text)Summarises long textfinish(answer)Returns final answer
In chat mode you can also type agent: your question to invoke agent inline.
