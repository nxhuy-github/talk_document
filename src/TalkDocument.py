from langchain import document_loaders as dl
from langchain import text_splitter as ts
from langchain import embeddings
from langchain import vectorstores as vs
from langchain import retrievers
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.docstore.document import Document
from typing import Any

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TalkDocument:
    DS_TYPE_LIST = ["WEB", "TXT", "PDF"]
    SPLIT_TYPE_LIST = ["CHARACTER", "TOKEN"]
    EMBEDDING_TYPE_LIST = ["HF", "OPENAI"]
    VECTORSTORE_TYPE_LIST = ["FAISS", "CHROMA", "SVM"]
    CHAIN_TYPE_LIST = ["stuff", "map_reduce", "map_rerank", "refine"]
    REPO_ID_DEFAULT = "declare-lab/flan-alpaca-large"

    def __init__(
            self,
            data_source_path:str = None,
            data_text:str = None,
            HF_API_TOKEN:str = None,
            OPENAI_KEY:str = None
        ) -> None:
        self.data_source_path = data_source_path
        self.data_text = data_text
        self.HF_API_TOKEN = HF_API_TOKEN
        self.OPENAI_KEY = OPENAI_KEY

        self.document = None
        self.document_splitted = None
        self.embedding_model = None
        self.embedding_type = None
        self.db = None
        self.llm = None
        self.chain = None
        self.repo_id = TalkDocument.REPO_ID_DEFAULT

        if not self.data_source_path and not self.data_text:
            print("You must provide either data_source_path or data_text or both")

    def get_document(self, data_source_type:str = "TXT") -> list[Document]:
        data_source_type = data_source_type.upper() if data_source_type.upper() in TalkDocument.DS_TYPE_LIST else TalkDocument.DS_TYPE_LIST[0]

        if data_source_type == "WEB":
            loader = dl.WebBaseLoader(self.data_source_path)
            self.document = loader.load()

        else:
            if self.data_text:
                self.document = self.data_text
            else:
                if data_source_type == "TXT" and self.data_source_path:
                    loader = dl.TextLoader(self.data_source_path)
                    self.document = loader.load()
                if data_source_type == "PDF" and self.data_source_path:
                    loader = dl.PyPDFLoader(self.data_source_path)
                    self.document = loader.load()

        return self.document
    
    def get_split(self, split_type:str ="character", chunk_size:int = 200, chunk_overlap:int = 10) -> list[Document] | list[str]:
        split_type = split_type.upper() if split_type.upper() in TalkDocument.SPLIT_TYPE_LIST else TalkDocument.SPLIT_TYPE_LIST[0]

        if self.document:
            if split_type == "CHARACTER":
                text_splitter = ts.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            elif split_type == "TOKEN":
                text_splitter = ts.TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # If you input a string as a document, we'll perform a split_text.
            if self.data_text:
                try:
                    self.document_splitted = text_splitter.split_text(self.document)
                except Exception as err:
                    logger.exception(f"Error when split data text: {err}")

            # If you upload a document, we'll do a split_documents.
            elif self.data_source_path:
                try:
                    self.document_splitted = text_splitter.split_documents(self.document)
                except Exception as err:
                    logger.exception(f"Error when split data source: {err}")

        return self.document_splitted
    
    def get_embedding(
            self, 
            embedding_type:str = "HF", 
            OPENAI_KEY:str = None
        ) -> embeddings.HuggingFaceEmbeddings | embeddings.OpenAIEmbeddings:
        if not self.embedding_model:
            embedding_type = embedding_type.upper() if embedding_type.upper() in TalkDocument.EMBEDDING_TYPE_LIST else TalkDocument.EMBEDDING_TYPE_LIST[0]

            if embedding_type == "HF":
                self.embedding_model = embeddings.HuggingFaceEmbeddings()

            elif embedding_type == "OPENAI":
                self.OPENAI_KEY = self.OPENAI_KEY if self.OPENAI_KEY else OPENAI_KEY
                if self.OPENAI_KEY:
                    self.embedding_model = embeddings.OpenAIEmbeddings(openai_api_key=self.OPENAI_KEY)
                else:
                    logger.exception("You need an OpenAI key")

        self.embedding_type = embedding_type
        return self.embedding_model

    def get_storage(
            self, 
            vectorstore_type:str = "FAISS", 
            embedding_type:str = "HF", 
            OPENAI_KEY:str = None
        ) -> vs.FAISS | vs.Chroma | retrievers.SVMRetriever:
        self.embedding_type = embedding_type.upper() if embedding_type.upper() in TalkDocument.EMBEDDING_TYPE_LIST else TalkDocument.EMBEDDING_TYPE_LIST[0]
        vectorstore_type = vectorstore_type.upper() if vectorstore_type.upper() in TalkDocument.VECTORSTORE_TYPE_LIST else TalkDocument.VECTORSTORE_TYPE_LIST[0]

        # Init embedding model
        self.get_embedding(self.embedding_type, OPENAI_KEY)

        # Init vector store model
        if vectorstore_type == "FAISS":
            model_vectorstore = vs.FAISS
        elif vectorstore_type == "CHROMA":
            model_vectorstore = vs.Chroma
        elif vectorstore_type == "SVM":
            model_vectorstore = retrievers.SVMRetriever


        if self.data_text:
            try:
                self.db = model_vectorstore.from_texts(self.document_splitted, self.embedding_model)
            except Exception as err:
                logger.exception(f"Error when create vector store for raw text: {err}")

        elif self.data_source_path:
            try:
                self.db = model_vectorstore.from_documents(self.document_splitted, self.embedding_model)
            except Exception as err:
                logger.exception(f"Error when create vector store for documents: {err}")

        return self.db
    
    def get_search(self, question:str, with_score:bool = False) -> list[tuple[Document, float]] | list[Document] | Any:
        relevant_docs = None

        if self.db and "SVM" not in str(type(self.db)):
            if with_score:
                relevant_docs = self.db.similarity_search_with_relevance_scores(question)
            else:
                relevant_docs = self.db.similarity_search(question)

        elif self.db:
            relevant_docs = self.db.get_relevant_documents(question)

        return relevant_docs
    
    def create_db_document(
            self,
            data_source_type:str = "TXT",
            split_type:str = "character",
            chunk_size:int = 200,
            embedding_type:str = "HF",
            chunk_overlap:int = 10,
            OPENAI_KEY:str = None,
            vectorstore_type:str = "FAISS"
        ) -> vs.FAISS | vs.Chroma | retrievers.SVMRetriever:
        self.get_document(data_source_type=data_source_type)
        self.get_split(split_type=split_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        db = self.get_storage(vectorstore_type=vectorstore_type, embedding_type=embedding_type, OPENAI_KEY=OPENAI_KEY)

        return db
    
    def do_question(
            self,
            question:str,
            repo_id:str = "declare-lab/flan-alpaca-large", 
            chain_type:str = "stuff", # This means that the answer is the first solution found by the LLM
            # relevant_docs:list[tuple[Document, float]] | list[Document] | Any = None,
            with_score:bool = False,
            temperature:int = 0,
            max_length:int = 300,
            language:str ="English"
    ) -> dict[str, Any] :
        # We restore the most important parts (splits) related to the question
        relevant_docs = self.get_search(question=question, with_score=with_score)
        
        if relevant_docs:
            # We define the LLM that we want to use, 
            # we must introduce the repo id since we are using huggingface.
            chain_type = chain_type.lower() if chain_type.lower() in TalkDocument.CHAIN_TYPE_LIST else TalkDocument.CHAIN_TYPE_LIST[0]

            if (self.repo_id != repo_id) or (self.llm is None):
                self.repo_id = repo_id

                # We created the LLM
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id, 
                    huggingfacehub_api_token=self.HF_API_TOKEN, 
                    model_kwargs={
                        "temperature": temperature, 
                        "max_length": max_length
                    }
                )
            
            # We prepare a prompt that includes the question, how we want the answer, and those text elements
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            If the question is similar to [Talk me about the document], 
            the response should be a summary commenting on the most important points about the document
            
            
            {context}
            Question: {question}
            """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            PROMPT = PROMPT + f" The Answer have to be in {language} language"

            # We create the chain, chain_type= "stuff".
            # "stuff" means that the answer is the first solution found by the LLM
            self.chain = self.chain if self.chain else load_qa_chain(llm=self.llm, chain_type=chain_type, prompt=PROMPT)

            # We pass this prompt to our Intelligent Language Model (LLM)
            response = self.chain({
                "input_documents": relevant_docs,
                "question": question
            }, return_only_outputs=True)

            return response
        else:
            return {"output_text": "ERROR: Something went wrong and the query could not be performed. Check the data source and its access"}

