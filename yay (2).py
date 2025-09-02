import os
import streamlit as st
import PyPDF2
import requests
import json
from typing import List, Dict, Optional
import re
from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF document"""
    content: str
    page_number: int
    chunk_index: int
    
class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def _init_(self):
        self.max_chunk_size = 1000  # Maximum characters per chunk
        self.overlap_size = 100     # Overlap between chunks for context
    
    def extract_text_from_pdf(self, pdf_file) -> List[DocumentChunk]:
        """Extract text from PDF file and split into chunks"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            chunks = []
            chunk_index = 0
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Clean and process text
                    cleaned_text = self._clean_text(text)
                    
                    # Split text into chunks
                    page_chunks = self._split_text_into_chunks(
                        cleaned_text, page_num + 1, chunk_index
                    )
                    chunks.extend(page_chunks)
                    chunk_index += len(page_chunks)
            
            logger.info(f"Extracted {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
    
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
        return text.strip()
    
    def _split_text_into_chunks(self, text: str, page_num: int, start_chunk_index: int) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_index = start_chunk_index
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # If we're not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    page_number=page_num,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap_size)
        
        return chunks

class GraniteAI:
    """Interface for IBM Granite AI model"""
    
    def _init_(self, api_key: str = None, model_id: str = "ibm-granite/granite-4.0-tiny-preview"):
        self.api_key = api_key or os.getenv("IBM_API_KEY")
        self.model_id = model_id
        self.base_url = "https://api.watsonx.ai/v1"  # Update with actual Watson API endpoint
        
        if not self.api_key:
            logger.warning("No API key provided. Using mock responses for demonstration.")
    
    def generate_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate answer using Granite model based on context"""
        try:
            # Prepare context from chunks
            context = self._prepare_context(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            if self.api_key:
                # Make API call to IBM Watson/Granite
                response = self._call_granite_api(prompt)
                return response
            else:
                # Mock response for demonstration
                return self._generate_mock_response(question, context_chunks)
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        """Prepare context text from document chunks"""
        context_parts = []
        for chunk in chunks[:5]:  # Limit to top 5 relevant chunks
            context_parts.append(f"[Page {chunk.page_number}] {chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for the AI model"""
        prompt = f"""Based on the following document content, please answer the question accurately and concisely. Only use information from the provided context.

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _call_granite_api(self, prompt: str) -> str:
        """Make API call to IBM Granite model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        # Note: Update this URL with the actual IBM Watson API endpoint
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
    
    def _generate_mock_response(self, question: str, chunks: List[DocumentChunk]) -> str:
        """Generate a mock response for demonstration purposes"""
        if not chunks:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Simple keyword matching for demonstration
        question_lower = question.lower()
        relevant_content = []
        
        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            # Simple relevance check
            if any(word in chunk_lower for word in question_lower.split() if len(word) > 3):
                relevant_content.append(f"From page {chunk.page_number}: {chunk.content[:200]}...")
        
        if relevant_content:
            return f"Based on the document content, here's what I found:\n\n" + "\n\n".join(relevant_content[:2])
        else:
            return "I couldn't find specific information related to your question in the uploaded document."

class StudyMateApp:
    """Main StudyMate application"""
    
    def _init_(self):
        self.pdf_processor = PDFProcessor()
        self.ai_model = GraniteAI()
        self.document_chunks = []
        self.chat_history = []
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="StudyMate - AI PDF Q&A",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 StudyMate: AI-Powered PDF Q&A System")
        st.markdown("Upload your PDF and ask questions about its content!")
        
        # Sidebar for file upload and settings
        with st.sidebar:
            st.header("📁 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document to start asking questions"
            )
            
            if uploaded_file is not None:
                if st.button("Process PDF", type="primary"):
                    with st.spinner("Processing PDF..."):
                        try:
                            self.document_chunks = self.pdf_processor.extract_text_from_pdf(uploaded_file)
                            st.success(f"✅ Processed {len(self.document_chunks)} text chunks from your PDF!")
                            st.session_state.pdf_processed = True
                            st.session_state.document_chunks = self.document_chunks
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {str(e)}")
            
            # Display document info
            if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
                st.info(f"📄 Document loaded with {len(st.session_state.document_chunks)} text segments")
                
                # Show sample content
                if st.expander("Preview Document Content"):
                    if st.session_state.document_chunks:
                        sample_chunk = st.session_state.document_chunks[0]
                        st.text_area(
                            f"Sample from Page {sample_chunk.page_number}:",
                            sample_chunk.content[:300] + "...",
                            height=150,
                            disabled=True
                        )
        
        # Main chat interface
        if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
            self._render_chat_interface()
        else:
            st.info("👆 Please upload and process a PDF file to start asking questions.")
            
            # Show example questions
            st.subheader("Example Questions You Can Ask:")
            examples = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What does the document say about [specific topic]?",
                "Explain the concept mentioned on page X",
                "What are the conclusions or recommendations?"
            ]
            
            for example in examples:
                st.write(f"• {example}")
    
    def _render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Ask Questions About Your Document")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
        
        # Question input
        question = st.chat_input("Ask a question about your document...")
        
        if question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Generate and display answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Find relevant chunks (simple implementation)
                    relevant_chunks = self._find_relevant_chunks(question, st.session_state.document_chunks)
                    
                    # Generate answer
                    answer = self.ai_model.generate_answer(question, relevant_chunks)
                    st.write(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    def _find_relevant_chunks(self, question: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Find relevant document chunks for the question (simple keyword matching)"""
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            # Simple relevance score based on word overlap
            overlap = len(question_words.intersection(chunk_words))
            if overlap > 0:
                scored_chunks.append((chunk, overlap))
        
        # Sort by relevance and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:5]]

def main():
    """Main entry point"""
    app = StudyMateApp()
    app.run()

if __name__ == "_main_":
    main()

# Requirements file content (save as requirements.txt):
"""
streamlit>=1.28.0
PyPDF2>=3.0.1
requests>=2.31.0
python-dotenv>=1.0.0
"""

# Setup instructions:
"""
1. Install required packages:
   pip install -r requirements.txt

2. Set up IBM Watson API key (optional for full functionality):
   export IBM_API_KEY="your_api_key_here"
   
   Or create a .env file with:
   IBM_API_KEY=your_api_key_here

3. Run the application:
   streamlit run studymate.py

4. Open your browser to http://localhost:8501

Note: This implementation includes a mock AI response system for demonstration.
To use the actual IBM Granite model, you'll need to:
- Sign up for IBM Watson/watsonx.ai
- Get your API credentials
- Update the API endpoint in the GraniteAI class
"""
import os
import streamlit as st
import PyPDF2
import requests
import json
from typing import List, Dict, Optional
import re
from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF document"""
    content: str
    page_number: int
    chunk_index: int
    
class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def _init_(self):
        self.max_chunk_size = 1000  # Maximum characters per chunk
        self.overlap_size = 100     # Overlap between chunks for context
    
    def extract_text_from_pdf(self, pdf_file) -> List[DocumentChunk]:
        """Extract text from PDF file and split into chunks"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            chunks = []
            chunk_index = 0
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Clean and process text
                    cleaned_text = self._clean_text(text)
                    
                    # Split text into chunks
                    page_chunks = self._split_text_into_chunks(
                        cleaned_text, page_num + 1, chunk_index
                    )
                    chunks.extend(page_chunks)
                    chunk_index += len(page_chunks)
            
            logger.info(f"Extracted {len(chunks)} chunks from {len(pdf_reader.pages)} pages")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
        return text.strip()
    
    def _split_text_into_chunks(self, text: str, page_num: int, start_chunk_index: int) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_index = start_chunk_index
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # If we're not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    page_number=page_num,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap_size)
        
        return chunks

class GraniteAI:
    """Interface for IBM Granite AI model"""
    
    def _init_(self, api_key: str = None, model_id: str = "ibm-granite/granite-4.0-tiny-preview"):
        self.api_key = api_key or os.getenv("IBM_API_KEY")
        self.model_id = model_id
        self.base_url = "https://api.watsonx.ai/v1"  # Update with actual Watson API endpoint
        
        if not self.api_key:
            logger.warning("No API key provided. Using mock responses for demonstration.")
    
    def generate_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate answer using Granite model based on context"""
        try:
            # Prepare context from chunks
            context = self._prepare_context(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            if self.api_key:
                # Make API call to IBM Watson/Granite
                response = self._call_granite_api(prompt)
                return response
            else:
                # Mock response for demonstration
                return self._generate_mock_response(question, context_chunks)
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        """Prepare context text from document chunks"""
        context_parts = []
        for chunk in chunks[:5]:  # Limit to top 5 relevant chunks
            context_parts.append(f"[Page {chunk.page_number}] {chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for the AI model"""
        prompt = f"""Based on the following document content, please answer the question accurately and concisely. Only use information from the provided context.

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _call_granite_api(self, prompt: str) -> str:
        """Make API call to IBM Granite model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        # Note: Update this URL with the actual IBM Watson API endpoint
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
    
    def _generate_mock_response(self, question: str, chunks: List[DocumentChunk]) -> str:
        """Generate a mock response for demonstration purposes"""
        if not chunks:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Simple keyword matching for demonstration
        question_lower = question.lower()
        relevant_content = []
        
        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            # Simple relevance check
            if any(word in chunk_lower for word in question_lower.split() if len(word) > 3):
                relevant_content.append(f"From page {chunk.page_number}: {chunk.content[:200]}...")
        
        if relevant_content:
            return f"Based on the document content, here's what I found:\n\n" + "\n\n".join(relevant_content[:2])
        else:
            return "I couldn't find specific information related to your question in the uploaded document."

class StudyMateApp:
    """Main StudyMate application"""
    
    def _init_(self):
        self.pdf_processor = PDFProcessor()
        self.ai_model = GraniteAI()
        self.document_chunks = []
        self.chat_history = []
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="StudyMate - AI PDF Q&A",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 StudyMate: AI-Powered PDF Q&A System")
        st.markdown("Upload your PDF and ask questions about its content!")
        
        # Sidebar for file upload and settings
        with st.sidebar:
            st.header("📁 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document to start asking questions"
            )
            
            if uploaded_file is not None:
                if st.button("Process PDF", type="primary"):
                    with st.spinner("Processing PDF..."):
                        try:
                            self.document_chunks = self.pdf_processor.extract_text_from_pdf(uploaded_file)
                            st.success(f"✅ Processed {len(self.document_chunks)} text chunks from your PDF!")
                            st.session_state.pdf_processed = True
                            st.session_state.document_chunks = self.document_chunks
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {str(e)}")
            
            # Display document info
            if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
                st.info(f"📄 Document loaded with {len(st.session_state.document_chunks)} text segments")
                
                # Show sample content
                if st.expander("Preview Document Content"):
                    if st.session_state.document_chunks:
                        sample_chunk = st.session_state.document_chunks[0]
                        st.text_area(
                            f"Sample from Page {sample_chunk.page_number}:",
                            sample_chunk.content[:300] + "...",
                            height=150,
                            disabled=True
                        )
        
        # Main chat interface
        if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
            self._render_chat_interface()
        else:
            st.info("👆 Please upload and process a PDF file to start asking questions.")
            
            # Show example questions
            st.subheader("Example Questions You Can Ask:")
            examples = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What does the document say about [specific topic]?",
                "Explain the concept mentioned on page X",
                "What are the conclusions or recommendations?"
            ]
            
            for example in examples:
                st.write(f"• {example}")
    
    def _render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Ask Questions About Your Document")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
        
        # Question input
        question = st.chat_input("Ask a question about your document...")
        
        if question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Generate and display answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Find relevant chunks (simple implementation)
                    relevant_chunks = self._find_relevant_chunks(question, st.session_state.document_chunks)
                    
                    # Generate answer
                    answer = self.ai_model.generate_answer(question, relevant_chunks)
                    st.write(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    def _find_relevant_chunks(self, question: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Find relevant document chunks for the question (simple keyword matching)"""
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            # Simple relevance score based on word overlap
            overlap = len(question_words.intersection(chunk_words))
            if overlap > 0:
                scored_chunks.append((chunk, overlap))
        
        # Sort by relevance and return top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:5]]

def main():
    """Main entry point"""
    app = StudyMateApp()
    app.run()

if __name__ == "_main_":
    main()

# Requirements file content (save as requirements.txt):
"""
streamlit>=1.28.0
PyPDF2>=3.0.1
requests>=2.31.0
python-dotenv>=1.0.0
"""

# Setup instructions:
"""
1. Install required packages:
   pip install -r requirements.txt

2. Set up IBM Watson API key (optional for full functionality):
   export IBM_API_KEY="your_api_key_here"
   
   Or create a .env file with:
   IBM_API_KEY=your_api_key_here

3. Run the application:
   streamlit run studymate.py

4. Open your browser to http://localhost:8501

Note: This implementation includes a mock AI response system for demonstration.
To use the actual IBM Granite model, you'll need to:
- Sign up for IBM Watson/watsonx.ai
- Get your API credentials
- Update the API endpoint in the GraniteAI class
"""