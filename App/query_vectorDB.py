from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load env files
load_dotenv()
similarity_search_index = int(os.getenv("similarity_search_index"))
chorma_DB_path = os.getenv("CHROMA_DB_PATH")
similarity_margin_value = float(os.getenv("SIMILARITY_MARGIN_VALUE"))

# Define embedding model (Local embedding model through hugging face)
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

# Define LLM model (Local embedding model through hugging face)
llm_model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name)
text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)


# Vector Embedding the question and performing the similarity search and get the top most results
def query_database(question: str):
    db = Chroma(
        persist_directory = chorma_DB_path,
        embedding_function = embedding_model
    )
    search_results = db.similarity_search_with_relevance_scores(question, similarity_search_index)
    filtered_results = [res for res in search_results if res[1] > similarity_margin_value]
    print("\n\n",search_results)
    print("\n\n",filtered_results)
    return filtered_results

# Query using the LLM
def query_question(question: str) -> str:
    print("Question is", question)
    results = query_database(question)
    if not results:
        return "No relevant information found in documents. Try another question."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = f"""
You are a helpful internal analytics assistant specialized in Finance and Project Management.
Answer the given question using only the provided context. And only answer the relevant question.

Context:
{context_text}

Question:
{question}
"""
    # llm = ChatOpenAI(model="gpt-4o-mini")
    # response = llm.invoke(prompt)
    # answer_text = response.content
    # sources_text = format_sources(results)
    # return f"{answer_text}\n\n📂 Sources Utilized:\n{sources_text}"

    answer_text = local_llm_inference(prompt)
    sources_text = format_sources(results)
    return f"{answer_text}\n\n📂 Sources Utilized:\n{sources_text}"


# Define a local LLM inference
def local_llm_inference(prompt: str, max_tokens: int = 200):
    outputs = text_gen(prompt, max_length=max_tokens, do_sample=True)
    return outputs[0]["generated_text"]


# -Format Sources to show in the output
def format_sources(search_results):
    sources = [
        os.path.basename(doc.metadata.get("source", ""))
        for doc, _ in search_results
        if doc.metadata.get("source")
    ]
    unique_sources = list(dict.fromkeys(sources))
    return "\n".join([f"    • {s}" for s in unique_sources]) if unique_sources else "No sources available"
