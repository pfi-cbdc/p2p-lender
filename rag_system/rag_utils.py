import os
import json
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import chromadb
from django.conf import settings

class RAGSystem:
    def __init__(self):
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize vector store
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create persistent directory for ChromaDB
        self.persist_directory = "chroma_db"
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

    def process_json_data(self, json_data: Dict[str, Any]) -> List[Document]:
        """Convert JSON data into LangChain Documents"""
        documents = []
        
        def extract_text_from_json(obj, path=""):
            """Recursively extract text from JSON structure"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    extract_text_from_json(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    extract_text_from_json(item, current_path)
            elif isinstance(obj, (str, int, float, bool)):
                # Create a document for each text field
                content = f"Field: {path}\nValue: {obj}"
                metadata = {"source": "json_upload", "field_path": path}
                documents.append(Document(page_content=content, metadata=metadata))
        
        extract_text_from_json(json_data)
        return documents

    def build_knowledge_base(self, json_data: Dict[str, Any]) -> str:
        """Build vector database from JSON data"""
        try:
            # Process JSON into documents
            documents = self.process_json_data(json_data)
            
            if not documents:
                return "No text content found in JSON data"
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the database
            self.vector_store.persist()
            
            # Clear memory when new data is uploaded
            self.memory.clear()
            
            return f"Successfully built knowledge base with {len(texts)} text chunks"
            
        except Exception as e:
            return f"Error building knowledge base: {str(e)}"

    def query_knowledge_base(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base with a question"""
        if not self.vector_store:
            return {"error": "Knowledge base not initialized. Please upload JSON data first."}
        
        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Get conversation history
            chat_history = self.memory.chat_memory.messages
            history_text = ""
            if chat_history:
                history_text = "\n".join([f"Q: {msg.content}" if i % 2 == 0 else f"A: {msg.content}" 
                                        for i, msg in enumerate(chat_history)])
            
            # Create prompt with context and history
            qa_prompt_template = """
            Use the following context and chat history to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=qa_prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            # Format the prompt
            formatted_prompt = PROMPT.format(
                context=context,
                chat_history=history_text,
                question=question
            )
            
            # Get answer from LLM
            result = self.llm.invoke(formatted_prompt)
            
            # Store in memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(result.content)
            
            return {
                "answer": result.content,
                "question": question,
                "sources": self._get_sources(question),
                "conversation_history": self.get_conversation_history()
            }
            
        except Exception as e:
            return {"error": f"Error querying knowledge base: {str(e)}"}

    def _get_sources(self, question: str) -> List[Dict[str, Any]]:
        """Get relevant sources for a question"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(question, k=3)
            sources = []
            for doc in docs:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return sources
        except Exception as e:
            return []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        try:
            chat_history = self.memory.chat_memory.messages
            history = []
            
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    history.append({
                        "question": chat_history[i].content,
                        "answer": chat_history[i + 1].content
                    })
            
            return history
        except Exception as e:
            return []

    def clear_conversation_history(self) -> Dict[str, str]:
        """Clear the conversation history"""
        try:
            self.memory.clear()
            return {"message": "Conversation history cleared successfully"}
        except Exception as e:
            return {"error": f"Error clearing conversation history: {str(e)}"}

    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get summary of the knowledge base"""
        if not self.vector_store:
            return {"error": "Knowledge base not initialized"}
        
        try:
            # Get all documents
            all_docs = self.vector_store.get()
            
            return {
                "total_documents": len(all_docs['documents']) if all_docs['documents'] else 0,
                "total_chunks": len(all_docs['embeddings']) if all_docs['embeddings'] else 0,
                "status": "active",
                "conversation_count": len(self.get_conversation_history())
            }
        except Exception as e:
            return {"error": f"Error getting summary: {str(e)}"}

# Global RAG system instance
rag_system = RAGSystem()

def process_json_with_rag(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to process JSON with RAG system"""
    try:
        # Build knowledge base
        build_result = rag_system.build_knowledge_base(json_data)
        
        if "Error" in build_result:
            return {"error": build_result}
        
        # Get summary
        summary = rag_system.get_knowledge_base_summary()
        
        return {
            "status": "success",
            "message": build_result,
            "summary": summary,
            "endpoint": "/api/rag/query/",
            "instructions": "Use the query endpoint to ask questions about your uploaded data"
        }
        
    except Exception as e:
        return {"error": f"Error processing JSON with RAG: {str(e)}"}

def query_rag_system(question: str) -> Dict[str, Any]:
    """Query the RAG system with a question"""
    return rag_system.query_knowledge_base(question)

def get_conversation_history() -> List[Dict[str, str]]:
    """Get conversation history"""
    return rag_system.get_conversation_history()

def clear_conversation_history() -> Dict[str, str]:
    """Clear conversation history"""
    return rag_system.clear_conversation_history() 