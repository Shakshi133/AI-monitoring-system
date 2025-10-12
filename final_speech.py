import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
import os
from pydub import AudioSegment
import io
import tempfile
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import speech_recognition as sr
from gtts import gTTS
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data
nltk.download('punkt')

# Check for GPU availability
if not torch.cuda.is_available():
    st.error("GPU not detected. This application requires a GPU to run.")
    st.stop()

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
groq_api_key = os.getenv('GROQ_API')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "recorded_text" not in st.session_state:
    st.session_state.recorded_text = ""
if "vocabulary_metrics" not in st.session_state:
    st.session_state.vocabulary_metrics = {
        'unique_words': set(),
        'total_words': 0,
        'sessions': []
    }
if "sentence_metrics" not in st.session_state:
    st.session_state.sentence_metrics = {
        'simple': 0,
        'compound': 0,
        'complex': 0,
        'compound_complex': 0
    }
if "error_metrics" not in st.session_state:
    st.session_state.error_metrics = {
        'grammar': 0,
        'spelling': 0,
        'punctuation': 0,
        'word_choice': 0
    }

def analyze_sentence_complexity(text):
    """Analyze the complexity of sentences in the text"""
    sentences = sent_tokenize(text)
    for sentence in sentences:
        # Count clauses by looking for coordinating conjunctions and subordinating conjunctions
        conjunctions = len(re.findall(r'\b(and|but|or|nor|for|yet|so|because|although|if|when|while)\b', sentence.lower()))
        
        if conjunctions == 0:
            st.session_state.sentence_metrics['simple'] += 1
        elif conjunctions == 1:
            if re.search(r'\b(because|although|if|when|while)\b', sentence.lower()):
                st.session_state.sentence_metrics['complex'] += 1
            else:
                st.session_state.sentence_metrics['compound'] += 1
        else:
            st.session_state.sentence_metrics['compound_complex'] += 1

def update_metrics(user_input):
    """Update all metrics based on user input"""
    # Update vocabulary metrics
    words = set(re.findall(r'\b\w+\b', user_input.lower()))
    st.session_state.vocabulary_metrics['unique_words'].update(words)
    st.session_state.vocabulary_metrics['total_words'] += len(words)
    
    # Add session data
    st.session_state.vocabulary_metrics['sessions'].append({
        'session': len(st.session_state.vocabulary_metrics['sessions']) + 1,
        'unique_words': len(st.session_state.vocabulary_metrics['unique_words']),
        'total_words': st.session_state.vocabulary_metrics['total_words']
    })
    
    # Update sentence complexity metrics
    analyze_sentence_complexity(user_input)
    
    # Simple error detection (this could be made more sophisticated)
    # Currently just checking for common patterns
    text_lower = user_input.lower()
    if any(word in text_lower for word in ['is', 'are', 'was', 'were']) and "i" in text_lower:
        st.session_state.error_metrics['grammar'] += 1
    if len(re.findall(r'[.!?]', user_input)) < len(sent_tokenize(user_input)):
        st.session_state.error_metrics['punctuation'] += 1

def create_dashboard():
    """Create and display the learning analytics dashboard"""
    st.subheader("Learning Analytics Dashboard")
    
    # Create three columns for the dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Vocabulary Growth Chart
        if st.session_state.vocabulary_metrics['sessions']:
            df_vocab = pd.DataFrame(st.session_state.vocabulary_metrics['sessions'])
            fig_vocab = px.line(df_vocab, x='session', y=['unique_words', 'total_words'],
                              title='Vocabulary Growth Over Time',
                              labels={'value': 'Word Count', 'session': 'Session Number'})
            st.plotly_chart(fig_vocab)
    
    with col2:
        # Sentence Complexity Pie Chart
        sentence_data = pd.DataFrame({
            'Type': ['Simple', 'Compound', 'Complex', 'Compound-Complex'],
            'Count': [st.session_state.sentence_metrics['simple'],
                     st.session_state.sentence_metrics['compound'],
                     st.session_state.sentence_metrics['complex'],
                     st.session_state.sentence_metrics['compound_complex']]
        })
        fig_sentence = px.pie(sentence_data, values='Count', names='Type',
                            title='Sentence Complexity Distribution')
        st.plotly_chart(fig_sentence)
    # 'Spelling',
    # Error Analysis Bar Chart
    error_data = pd.DataFrame({
        'Category': ['Grammar', 'Punctuation', 'Word Choice'],
        'Errors': [st.session_state.error_metrics['grammar'],
                #   st.session_state.error_metrics['spelling'],
                  st.session_state.error_metrics['punctuation'],
                  st.session_state.error_metrics['word_choice']]
    })
    fig_errors = px.bar(error_data, x='Category', y='Errors',
                       title='Common Mistakes Analysis')
    st.plotly_chart(fig_errors)

def load_documents():
    """Load and process PDF documents"""
    loader1 = PyPDFLoader(r'E:\LANGCHAIN\TENSES (1).pdf')
    loader2 = PyPDFLoader(r'E:\LANGCHAIN\senetnce structure.pdf')
    loader3 = PyPDFLoader(r'E:\LANGCHAIN\84669-pet-vocabulary-list.pdf')

    data1 = loader1.load()
    data2 = loader2.load()
    data3 = loader3.load()
    return data1 + data2 + data3

def initialize_rag_chain():
    """Initialize the RAG chain with necessary components"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=groq_api_key)

    persist_directory = "chroma_db"
    os.makedirs(persist_directory, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
    )

    data = load_documents()
    vectorstore = Chroma.from_documents(
        documents=data,
        embedding=embeddings,
        client=chroma_client,
        collection_name="english_learning"
    )

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})

    # Create context-aware retriever
    context_prompt = """Given a chat history and the latest user question \
    Which might reference context in the chat history \
    formulate a standalone question which can be understood \
    without the chat history."""

    context_template = ChatPromptTemplate.from_messages([
        ("system", context_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_chain = create_history_aware_retriever(llm, retriever, context_template)

    # Create QA chain with the same prompt as before
    qa_prompt = """You are an assistant designed to help children aged 1 to 8 enhance their communication and language skills. \
    Using the following context and the conversation provided by the user, analyze and provide feedback across multiple aspects to improve the child's language abilities:

Context: {context}
1. If the child asks anything related to maths like tables, maths problems, etc., then do not give any output related to the aspects below. Only focus on providing the correct solution to the math query and avoid including vocabulary growth, sentence complexity, communication patterns, or sentiment analysis.

2. *Simple Conversations*: If the child is just maintaining a simple conversation, do not analyze vocabulary, sentence complexity, communication patterns, or sentiment. Simply respond naturally and keep the conversation flowing without providing additional analysis.


if a user uses a keyword concise then make sure you give a conscise feedback

3. *Vocabulary Growth Detection: Identify and list all the **unique words* used by the child, one by one. \
Instead of just providing a count, please list every unique word used by the child in the input. \
make sure you use the {context} to answer the vocab
Encourage vocabulary growth by suggesting new words to incorporate and praising the child for using a variety of words. \
Track how the child's vocabulary expands over time and provide feedback on their progress.

4. *Sentence Complexity Analysis*: Evaluate the child's sentence structure, sentence length, and grammatical complexity. \
Provide suggestions for enhancing sentence complexity, such as using more descriptive adjectives, conjunctions, or compound sentences. \
Keep the language simple and clear, suitable for the child's understanding.

5. *Communication Patterns*: Analyze the flow of conversation, noting whether the child is appropriately taking turns, responding to questions, and initiating conversation. \
Provide advice on improving conversational skills, focusing on active listening, and initiating dialogue.

6. *Sentiment Analysis*: Analyze the child's emotional tone in the conversation. Track if the child's emotional responses or expressions are improving over time. \
Provide guidance on expressing feelings in a positive or constructive manner.

7.at end ask him if he want any drills or exercises releated 

Question: {input}"""


    # Context: {context}
    # Question: {input}"""

    qa_template = ChatPromptTemplate.from_messages([
        ("system", qa_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_template)
    return create_retrieval_chain(history_chain, qa_chain)

def process_audio_input(audio_data, recognizer):
    """Process audio input and convert to text"""
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Speech Recognition service; {e}")
        return None

def text_to_speech(text):
    """Convert text to speech"""
    try:
        temp_dir = "temp_audio"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_file = os.path.join(temp_dir, "temp_audio.mp3")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_file)
        
        with open(temp_file, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def convert_to_wav(audio_file):
    """Convert uploaded audio to WAV format"""
    try:
        audio_bytes = audio_file.read()
        audio_file.seek(0)
        
        file_extension = audio_file.name.split('.')[-1].lower()
        
        if file_extension in ['mp3', 'm4a']:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_extension)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                audio.export(temp_wav.name, format='wav')
                return temp_wav.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_wav.write(audio_bytes)
                return temp_wav.name
    except Exception as e:
        st.error(f"Error converting audio file: {e}")
        return None

def handle_user_input(user_input, rag_chain):
    """Process user input and generate response"""
    # Update metrics
    update_metrics(user_input)
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        if st.button("🔊", key=f"listen_user_{len(st.session_state.messages)}"):
            audio_bytes = text_to_speech(user_input)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            langchain_history = []
            for msg in st.session_state.chat_history:
                langchain_history.append(HumanMessage(content=msg[0]))
                langchain_history.append(AIMessage(content=msg[1]))
            
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": langchain_history
            })
            
            response_text = response['answer']
            st.markdown(response_text)
            
            if st.button("🔊", key=f"listen_assistant_{len(st.session_state.messages)}"):
                audio_bytes = text_to_speech(response_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
    
    # Update session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.chat_history.append((user_input, response_text))

def main():
    st.title("English Learning Assistant")
    st.write("I'm here to help children improve their English skills!")
    
    # Initialize RAG chain
    rag_chain = initialize_rag_chain()
    
    # Add dashboard button in sidebar
    with st.sidebar:
        if st.button("Show Dashboard"):
            create_dashboard()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if st.button("🔊", key=f"listen_{message['role']}_{id(message)}"):
                audio_bytes = text_to_speech(message["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
    
    # Show recorded text if available
    if st.session_state.recorded_text:
        st.info(f"Recorded text: {st.session_state.recorded_text}")
        if st.button("Send recorded text"):
            handle_user_input(st.session_state.recorded_text, rag_chain)
            st.session_state.recorded_text = ""
            st.rerun()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"], key="audio_uploader")
    
    # Handle file upload
    if uploaded_file:
        try:
            wav_path = convert_to_wav(uploaded_file)
            if wav_path:
                try:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(wav_path) as source:
                        audio = recognizer.record(source)
                        text = process_audio_input(audio, recognizer)
                        if text:
                            st.session_state.recorded_text = text
                            st.rerun()
                finally:
                    os.unlink(wav_path)
        except Exception as e:
            st.error("Please upload a valid audio file")
    
    # Input area with buttons
    # col1, col2 = st.columns([6, 1])
    
    # with col

    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.chat_input("Message", key=f"chat_input_{len(st.session_state.messages)}")
    
    with col2:
        if not st.session_state.recording:
            if st.button("🎤"):
                st.session_state.recording = True
                try:
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        st.write("Recording...")
                        audio = recognizer.listen(source, timeout=5)
                        text = process_audio_input(audio, recognizer)
                        if text:
                            st.session_state.recorded_text = text
                            st.session_state.recording = False
                            st.rerun()
                except Exception as e:
                    st.error(f"Error recording audio: {e}")
                    st.session_state.recording = False
        else:
            st.button("⏹️")
    
    # Process user input
    if user_input:
        handle_user_input(user_input, rag_chain)
        st.rerun()


# Main function
if __name__ == "__main__":
    main()
