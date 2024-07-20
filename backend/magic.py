import re
import logging
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from decouple import config
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatCohere()
embeddings_model = CohereEmbeddings(model='embed-english-light-v3.0')

# Preprocess the document and store embeddings
def preprocess_and_store_embeddings(file_path):
    try:
        loader =  PDFMinerPDFasHTMLLoader(file_path)
        data = loader.load()[0]

        soup = BeautifulSoup(data.page_content,'html.parser')
        content = soup.find_all('div')

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
                snippets.append(cur_text)
                cur_fs = fs
                cur_text = c.text
        snippets.append(cur_text)
        # Note: The above logic is very straightforward can be found on the docs. 
        # # print(snippets)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.create_documents(snippets)
        # print(all_splits)
        # embeddings_model = CohereEmbeddings(model='embed-english-light-v3.0')

        vectorstore = FAISS.from_documents(all_splits, embeddings_model)
        # vectorstore.save_local(folder_path=vector_store_path,index_name="index")
        logger.info("Document embeddings stored successfully.")

        print("vectorstore.index.ntotal : ", vectorstore.index.ntotal)
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error preprocessing document: {e}")
        raise

# Query the vector store for relevant chunks
def query_vector_store(query, vector_store, top_k=5):
    embedding = embeddings_model.embed_query(query)
    return vector_store.similarity_search_by_vector(embedding, top_k)

if __name__ == "__main__":
    vector_store = preprocess_and_store_embeddings("constitution.pdf")

    query = "What are the rights of the people?"
    result = query_vector_store(query, vector_store, top_k=5)
    print("result: ", result)
