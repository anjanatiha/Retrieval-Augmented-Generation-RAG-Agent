import ollama
from flashrank import Ranker, RerankRequest

# Load the dataset
dataset = []
with open('cat-facts.txt', 'r') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=10):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Reranker setup
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

def rerank(query, chunks, top_n=3):
    passages = [
        {"id": i, "text": chunk}
        for i, (chunk, similarity) in enumerate(chunks)
    ]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    # ✅ Keep similarity score with reranked chunks
    return [(chunks[r["id"]][0], chunks[r["id"]][1]) for r in results[:top_n]]

# Single question
input_query = input('Ask me a question: ')

# Retrieve broadly
retrieved_knowledge = retrieve(input_query, top_n=10)

# ✅ Show before reranking with similarity
print('\nRetrieved knowledge (before reranking):')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

# Rerank
reranked_chunks = rerank(input_query, retrieved_knowledge, top_n=3)

# ✅ Show after reranking with similarity
print('\nReranked knowledge (after reranking):')
for chunk, similarity in reranked_chunks:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

# Build context
context = '\n'.join([f' - {chunk}' for chunk, similarity in reranked_chunks])

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context}
'''

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
)

print('\nChatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print()
