import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_community.vectorstores import FAISS
import streamlit as st

cohere_api_key = st.secrets["cohere_api_key"]

# pinecone_api_key="274a17e6-4589-46d5-9323-c324259a2905"
# st.secrets["pinecone_api_key"]
llm = ChatCohere(cohere_api_key=cohere_api_key, model=st.secrets["model"])
# print(llm)

loader =  PDFMinerPDFasHTMLLoader("TheConstitutionOfKenya.pdf")

data = loader.load()[0]

# print(data)

soup = BeautifulSoup(data.page_content,'html.parser')
content = soup.find_all('div')

# print(content)

cur_fs = None
cur_text = ''
snippets = []   # first collect all snippets that have the same font size
for c in content:
    sp = c.find('span')
    if not sp:
        continue
    st = sp.get('style')
    if not st:
        continue
    fs = re.findall('font-size:(\d+)px',st)
    if not fs:
        continue
    fs = int(fs[0])
    if not cur_fs:
        cur_fs = fs
    if fs == cur_fs:
        cur_text += c.text
    else:
        # snippets.append((cur_text,cur_fs))
        snippets.append(cur_text)
        cur_fs = fs
        cur_text = c.text
# snippets.append((cur_text,cur_fs))
snippets.append(cur_text)
# Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
# headers/footers in a PDF appear on muljupyter nbconvert --to script 'my-notebook.ipynb'tiple pages so if we find duplicates it's safe to assume that it is redundant info)

# print(snippets)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.create_documents(snippets)

# print(all_splits)

embeddings_model = CohereEmbeddings(cohere_api_key=cohere_api_key, model='embed-english-v3.0')

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings_model)


index_name = "langchain-test-index"

# Connect to Pinecone index and insert the chunked docs as contents
# vectorstore = PineconeVectorStore.from_documents(all_splits, embeddings_model, index_name=index_name)

vectorstore = FAISS.from_documents(all_splits, embeddings_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

#retrieved_docs = retriever.invoke("what rights of the people?.")
def retrieved_docs(prompt):
    result = retriever.invoke(prompt)
    return result

if __name__ == "__main__":
    retrieved_docs("prompt")

# print(retrieved_docs[0].page_content)
# for i in retrieved_docs:
#     print(i)
# print(one.page_content for one in retrieved_docs)
