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



Agent Tools Available
ToolWhat It Doesrag_search(query)Searches your documentscalculator(expression)Evaluates mathsummarise(text)Summarises long textfinish(answer)Returns final answer
In chat mode you can also type agent: your question to invoke agent inline.
