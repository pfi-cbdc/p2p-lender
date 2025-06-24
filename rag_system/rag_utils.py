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
import requests
from PyPDF2 import PdfReader
import re
import spacy

class RAGSystem:
    def __init__(self):
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        # Use RecursiveCharacterTextSplitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        # Custom prompt template for RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a financial assistant helping with loan data. 
Answer the following question using only the provided context.

Context:
{context}

Question:
{question}

Answer:"""
        )
        # spaCy NLP model (load once)
        self.nlp = spacy.load("en_core_web_sm")
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

        self.full_json_data = None  # Store the full JSON data

    def is_structured_question(self, question: str) -> bool:
        """Detect if the question is structured and should use the full JSON context."""
        patterns = [
            r"list of", r"how many", r"filter by", r"who has", r"compare"
        ]
        return any(re.search(pat, question, re.IGNORECASE) for pat in patterns)

    def structured_query_handler(self, question: str) -> str:
        """Use the full JSON data for structured questions."""
        if not self.full_json_data:
            return "No JSON data available for structured queries."
        prompt = f"""
You are a smart assistant for a loan application system. Here is the full data:
{json.dumps(self.full_json_data)}

Question: {question}
Answer:"""
        result = self.llm.invoke(prompt)
        return result.content

    def process_json_data(self, json_data: Dict[str, Any]) -> List[Document]:
        """Convert JSON data into one Document per applicant."""
        documents = []
        applicants = json_data.get("applicants", [])
        for entry in applicants:
            content = "\n".join([f"{k}: {v}" for k, v in entry.items()])
            documents.append(Document(page_content=content, metadata=entry))
        return documents

    def fetch_api_data(self, api_url: str) -> dict:
        """Fetch JSON data from the given API URL."""
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            # Wrap in 'applicants' for compatibility
            if isinstance(data, list):
                return {"applicants": data}
            return data
        except Exception as e:
            return {"error": f"Error fetching API data: {str(e)}"}

    def build_knowledge_base(self, json_data: dict = None) -> str:
        """Build vector database from API data (optionally also from provided JSON data)."""
        try:
            # Fetch API data
            api_url = "https://rag-test-x7m8.onrender.com/api/borrowers"
            api_data = self.fetch_api_data(api_url)
            if "error" in api_data:
                return api_data["error"]
            api_documents = self.process_json_data(api_data)

            # Optionally process user-uploaded JSON data
            user_documents = []
            if json_data:
                self.full_json_data = json_data  # Store for structured queries
                user_documents = self.process_json_data(json_data)
            else:
                self.full_json_data = api_data  # Fallback to API data for structured queries
            # Combine all documents
            all_documents = api_documents + user_documents
            if not all_documents:
                return "No text content found in API or JSON data"

            # Split documents into chunks
            texts = self.text_splitter.split_documents(all_documents)

            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vector_store.persist()
            self.memory.clear()

            # Set up retriever and QA chain with custom prompt
            self.retriever = self.vector_store.as_retriever()
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.prompt_template}
            )
            return f"Successfully built knowledge base with {len(texts)} text chunks from API and JSON data"
        except Exception as e:
            return f"Error building knowledge base: {str(e)}"

    def _extract_fields_from_question(self, question: str) -> list:
        """Extract likely field names from the question using spaCy noun chunks (up to 2 words, lowercased)."""
        doc = self.nlp(question)
        return [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 2]

    def _parse_income(self, income_str):
        """Parse income string to numeric bounds (min, max)."""
        if not isinstance(income_str, str):
            return (0, 0)
        s = income_str.lower().replace(",", "")
        if "upto" in s:
            nums = re.findall(r'\d+', s)
            return (0, int(nums[0])) if nums else (0, 0)
        elif "-" in s:
            nums = [int(x) for x in re.findall(r'\d+', s)]
            return (nums[0], nums[1]) if len(nums) == 2 else (0, 0)
        elif s.isdigit():
            val = int(s)
            return (val, val)
        return (0, 0)

    def _filter_json_by_fields(self, fields: list, question: str) -> list:
        """Filter self.full_json_data['applicants'] by logic inferred from the question and fields, with income range parsing."""
        if not self.full_json_data or "applicants" not in self.full_json_data:
            return []
        filtered = []
        # Extract numeric bounds from question for income
        q = question.lower().replace(",", "")
        q_nums = [int(x) for x in re.findall(r'\d+', q)]
        q_min, q_max = (None, None)
        if "over" in q or ">" in q:
            q_min = q_nums[0] if q_nums else 0
            q_max = float('inf')
        elif "under" in q or "<" in q:
            q_min = 0
            q_max = q_nums[0] if q_nums else 0
        elif len(q_nums) == 2:
            q_min, q_max = q_nums[0], q_nums[1]
        for entry in self.full_json_data["applicants"]:
            risk = entry.get("risk_category", "").lower()
            income_str = entry.get("monthly_income", "0")
            income_min, income_max = self._parse_income(income_str)
            match = True
            if "risk" in fields and "high" in q and "high" in risk:
                match = True
            elif "risk" in fields and "low" in q and "low" in risk:
                match = True
            elif "risk" in fields:
                match = False
            if "income" in fields and (q_min is not None or q_max is not None):
                # Check overlap between applicant's income range and query range
                if q_min is not None and income_max < q_min:
                    match = False
                if q_max is not None and income_min > q_max:
                    match = False
            if match:
                filtered.append(entry)
        return filtered

    def query_knowledge_base(self, question: str, debug: bool = False, source_page: int = 1, source_page_size: int = 3) -> Dict[str, Any]:
        """Query the knowledge base with a question, using hybrid pipeline and evaluation. Supports debug trace and paginated sources."""
        # Hybrid: if question is both structured and unstructured, combine logic
        is_structured = self.is_structured_question(question)
        matched_fields = self._extract_fields_from_question(question)
        filtered_json = []
        explanation = ""
        if is_structured and matched_fields:
            filtered_json = self._filter_json_by_fields(matched_fields, question)
        # If hybrid, combine filtered JSON and RAG context
        if is_structured and matched_fields and filtered_json:
            # Compose context for LLM: filtered JSON + top RAG chunks
            context = json.dumps(filtered_json, indent=2)
            rag_docs = self._get_sources(question)
            # Pagination for sources
            total_chunks = len(rag_docs)
            start = (source_page - 1) * source_page_size
            end = start + source_page_size
            paged_rag_docs = rag_docs[start:end]
            rag_context = "\n\n".join([doc["content"] for doc in paged_rag_docs])
            full_context = f"Structured Results:\n{context}\n\nRAG Results:\n{rag_context}"
            prompt = self.prompt_template.format(context=full_context, question=question)
            result = self.llm.invoke(prompt)
            answer = result.content
            confidence = "high" if len(filtered_json) > 0 else "medium"
            question_type = "hybrid"
            sources = paged_rag_docs
            explanation = f"Filtered {len(filtered_json)} applicants based on {', '.join(matched_fields)}, plus {len(paged_rag_docs)} vector-matched documents (page {source_page})."
        elif is_structured:
            answer = self.structured_query_handler(question)
            confidence = "medium"
            question_type = "structured"
            sources = [{"content": "Full JSON used", "metadata": {}}]
            explanation = f"Used structured query handler with fields: {', '.join(matched_fields)}."
        else:
            if not self.qa_chain:
                return {"error": "Knowledge base not initialized. Please upload JSON data first."}
            try:
                rag_docs = self._get_sources(question)
                total_chunks = len(rag_docs)
                start = (source_page - 1) * source_page_size
                end = start + source_page_size
                paged_rag_docs = rag_docs[start:end]
                result = self.qa_chain({"query": question})
                answer = result["result"]
                confidence = "medium"
                question_type = "unstructured"
                sources = paged_rag_docs
                explanation = f"Used RAG with top {len(paged_rag_docs)} vector-matched documents (page {source_page})."
            except Exception as e:
                return {"error": f"Error querying knowledge base: {str(e)}"}
        # Store in memory
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)
        # Evaluation output
        output = {
            "answer": answer,
            "question": question,
            "question_type": question_type,
            "confidence": confidence,
            "matched_fields": matched_fields,
            "sources": sources,
            "source_page": source_page,
            "source_page_size": source_page_size,
            "conversation_history": self.get_conversation_history()
        }
        if debug:
            output["explanation"] = explanation
            if is_structured and matched_fields:
                output["filtered_json_count"] = len(filtered_json)
            if 'total_chunks' in locals():
                output["top_chunks"] = len(sources)
        return output

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