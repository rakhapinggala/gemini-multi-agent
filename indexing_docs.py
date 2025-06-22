import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  

# Konfigurasi embedding Gemini
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Folder dan output
DATA_PATHS = {
    "finance": ("Lang-Learning/data/data_finance", "Lang-Learning/vectorstores/finance_index"),
    "hr": ("Lang-Learning/data/data_hr", "Lang-Learning/vectorstores/hr_index"),
    "it": ("Lang-Learning/data/data_it", "Lang-Learning/vectorstores/it_index"),
    "investor": ("Lang-Learning/data/data_invest", "Lang-Learning/vectorstores/invest_indexx"),
}

def build_index(doc_path, out_path):
    all_docs = []
    for file in os.listdir(doc_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(doc_path, file))
            docs = loader.load()
            all_docs.extend(docs)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(splits, embedding)
    vectorstore.save_local(out_path)
    print(f"✅ Index saved to {out_path}")

# Proses semua dokumen
for label, (doc_path, out_path) in DATA_PATHS.items():
    build_index(doc_path, out_path)
print("✅ All indexes built successfully.")

