# demo_simple_rag

Source: https://huggingface.co/blog/ngxson/make-your-own-rag

Let's install the ollama package:
<pre>
pip install ollama
</pre>
After installed, open a terminal and run the following command to download the required models:

<pre>
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
</pre>

<pre>
python demo.py
</pre>
