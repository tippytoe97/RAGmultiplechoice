from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

class Retriever:
    def __init__(self):
        self.huggingface_embeddings = None
        pass

    def loadDocuments(self, data_dir):
        '''
        Load the PDFs using Langchain's PDF Directory Loader (see imported package above)
        
        Args:
            data_dir: String path of folder location of PDFs to load

        Returns:
            documents: list of langchain documents
        '''
        loader = PyPDFDirectoryLoader(data_dir)
        documents = loader.load()
        return documents

    def splitDocuments(self, documents, chunk_size=700, chunk_overlap=50):
        '''
        Split the loaded documents into smaller chunks. 
        
        Args:
            documents: list of langchain documents
            chunk_size: int, number of characters in a chunk
            chunk_overlap: int, number of characters overlapping between adjacent chunks
        
        Returns:
            document_chunks: list of langchain document chunks
        '''
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        document_chunks = splitter.split_documents(documents)
        return document_chunks

    def createRetriever(self, document_chunks, num_chunks_to_return=4):
        '''
        Initialize the embedding and retriever model using the FAISS vectorstore and huggingface embeddings

        Args:
            document_chunks: list of langchain document chunks
            num_chunks_to_return: int, number of chunks to retrieve per query

        Returns:
            retriever: langchain VectorStoreRetriever
        '''
        self.huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # Replace None with HuggingFace Embeddings in this variable

        # Create FAISS index from embeddings
        vectorstore = FAISS.from_documents(document_chunks, self.huggingface_embeddings)

        # Create retriever with specified number of chunks to return
        retriever = vectorstore.as_retriever(search_kwargs={"k": num_chunks_to_return})

        return retriever