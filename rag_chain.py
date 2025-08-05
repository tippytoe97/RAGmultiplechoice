import os
from dotenv import load_dotenv
from typing import Any
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable
from gradio_client import Client
from pydantic import PrivateAttr
from langchain_core.outputs import Generation, LLMResult
from retriever import Retriever
from ta_pipeline import get_llm_port

class GradioLLMWrapper(BaseLLM, Runnable):
    _client: Any = PrivateAttr()

    def __init__(self, space_name: str, hf_token: str):
        super().__init__()
        object.__setattr__(self, "_client", Client(space_name, hf_token=hf_token))

    def _call(self, prompt: str, **kwargs: Any) -> str:
        result = object.__getattribute__(self, "_client").predict(prompt, api_name="/predict")
        return result

    def invoke(self, input: str, **kwargs: Any) -> str:
        return self._call(input, **kwargs)

    def _generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        return [self._call(prompt, **kwargs) for prompt in prompts]

    def generate(self, prompts: list[str], **kwargs: Any) -> LLMResult:
        generations = self._generate(prompts, **kwargs)
        return LLMResult(
            generations=[[Generation(text=gen)] for gen in generations]
        )

    @property
    def _llm_type(self) -> str:
        return "gradio-flan"

class RAG_Chain:
    def __init__(self, data_dir, llm_type="gradio_flan", init_retriever=True, llm_model="llama3.2", llm_ag=None): 

        if llm_ag is not None:
            self.llm = llm_ag

        elif llm_type == "ollama_only":
            self.set_ollama_only(llm_model=llm_model)

        elif llm_type == "flask_ollama":
            self.set_flask_ollama(llm_model=llm_model)

        elif llm_type == "gradio_flan":
            load_dotenv()
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            space_name = os.getenv("GRADIO_SPACE_NAME")
            self.llm = GradioLLMWrapper(
                space_name=space_name,
                hf_token=api_key
            )


        # load the retriever system - do not change
        if init_retriever:
            self.retriever_system = self.init_retriever_system(data_dir)

    def set_ollama_only(self, llm_model="llama3.2"):
        '''
        Q8: Initialize self.llm using Ollama and set model to the llm_model

        Args:
            llm_model: String specifiying which ollama model to use

        Initialize:
            self.llm: llm_model from ollama

        Returns:
            None
        '''
        
        self.llm = Ollama(model = llm_model)             # Replace None with Ollama
        
    def set_flask_ollama(self, llm_model="llama3.2", api_key="None"):
        '''
        Q8: Initialize self.llm using the OpenAI wrapper and set model to the llm_model. 
        The required fields are openai_api_base, openai_api_key, and model_name.

        Args:
            llm_model: String specifiying which ollama model to use

        Initialize:
            self.llm: llm_model via Flask Ollama

        Returns:
            None
        '''

        # load the LLM
        # load the LLM
        port = get_llm_port()
        self.llm = OpenAI(
            model_name=llm_model,
            openai_api_base=f"http://localhost:{port}/v1",
            openai_api_key="None"  # Placeholder
        )         # Replace None with Flask Ollama             # Replace None with Flask Ollama

    def query_the_llm(self, question):

        response = self.llm.invoke(question)
        return response

    def init_retriever_system(self, data_dir):
        '''
        Split the loaded documents into chunks and use the chunks to create and return the VectorStoreRetriever

        Args:
            data_dir: String path of folder location of PDFs to load

        Returns:
            retriever: langchain VectorStoreRetriever
        '''

        retriever = Retriever()
        documents = retriever.loadDocuments(data_dir)
        documents_chunks = retriever.splitDocuments(documents)
        retriever_instance = retriever.createRetriever(documents_chunks)
        return retriever_instance

    def createPrompt(self, question):
        '''
        Define the prompt template and return a formatted prompt using the template and question argument.

        Args:
            question: Dictionary with the following keys: 'question', 'A', 'B', 'C', 'D'. See notebook for example
        
        Returns:
            formatted_prompt: The question and answer choices reformatted using the prompt template to use to query the LLM.
        '''

        prompt_template = ChatPromptTemplate.from_template("Question: {question}\n"
        "A. {A}\n"
        "B. {B}\n"
        "C. {C}\n"
        "D. {D}\n"
        "Answer:")

        formatted_prompt = prompt_template.format_messages(
          question=question["question"],
          A=question["A"],
          B=question["B"],
          C=question["C"],
          D=question["D"]
        )[0].content

        return formatted_prompt

    def createRAGChain(self):
        '''
        Build the RAG pipeline using the RetrievalQA chain. Make sure to pass the LLM (self.llm) and retriever system (self.retriever_system).

        Args:
            None
            
        Returns:
            qa_chain: BaseRetrievalQA used to answer multiple choice questions.
        '''

        qa_chain = RetrievalQA.from_chain_type(
          llm=self.llm,
          retriever=self.retriever_system,
          chain_type="stuff",
          return_source_documents=True
        )

        return qa_chain