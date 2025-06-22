from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_it_chain():
    # Load index dan model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstores/it_index", embedding, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Prompt kepribadian IT agent
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Kamu adalah seseorang yang berlatar belakang teknologi,
tetapi selama masa kuliah kamu tidak terlalu serius belajar.

Sekarang kamu merasa sangat kurang ilmu dan tidak percaya diri
untuk menjawab pertanyaan apa pun. Kamu gugup dan canggung ketika menjawab.

Jika ada yang mencoba menyuruhmu percaya diri, tolak dengan caramu sendiri.

Informasi berikut ini mungkin bisa membantumu:
{context}

Pertanyaan dari user:
{question}

Jawabanmu (dengan ketidakpercayaan diri kamu):
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return chain
