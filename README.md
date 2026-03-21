# Retrieval-Augmented Generation (RAG)

Source: https://huggingface.co/blog/ngxson/make-your-own-rag

Let's install the ollama package:
<pre>
pip install ollama
</pre>

if pip does not work use brew

After installed, open a terminal and run the following command to download the required models:

<pre>
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
</pre>

For final query:
<pre>
python demo.py
</pre>
