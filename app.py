import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import base64
from PIL import Image
import os

# Load environment variables
load_dotenv()

st.title("AI-Coding-Agent")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Initialize Pinecone
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

# Connect to your Pinecone index
pinecone_index = pc.Index("codebase-rag")

def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace="https://github.com/CoderAgent/SecureAgent"
    )

    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.
    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama3-8b-8192",  # Updated to consistent model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

def perform_multimodal_rag(query, image_analysis=None):
    """
    Perform RAG with both text query and image analysis if available.
    """
    # Combine query with image analysis if available
    enhanced_query = query
    if image_analysis:
        enhanced_query = f"""
        Image Context:
        {image_analysis}
        
        User Query:
        {query}
        """
    
    # Get embeddings for the enhanced query
    raw_query_embedding = get_huggingface_embeddings(enhanced_query)

    # Query Pinecone for relevant code context
    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace="https://github.com/CoderAgent/SecureAgent"
    )

    # Extract contexts from matches
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    
    # Create augmented query with both code context and image analysis
    augmented_query = "<CONTEXT>\n" + \
                        "\n\n-------\n\n".join(contexts[:10]) + \
                        "\n-------</CONTEXT>\n\n" + \
                        (f"IMAGE ANALYSIS:\n{image_analysis}\n\n" if image_analysis else "") + \
                        "MY QUESTION:\n" + \
                        query

    system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.
    Answer questions about the codebase and any provided images, based on the code context and image analysis provided. 
    Always consider all of the context provided when forming a response.
    """

    # Get LLM response
    llm_response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

def encode_image_from_upload(uploaded_file):
    """
    Encode uploaded image to base64.
    """
    try:
        bytes_data = uploaded_file.getvalue()
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        return base64_string
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def analyze_image(image_base64):
    """
    Analyze an uploaded image using Groq's vision model.
    """
    if image_base64 is None:
        return None
    
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and describe what you see, including any code, diagrams, or technical content."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        
        if chat_completion.choices:
            return chat_completion.choices[0].message.content
        return None
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

# Streamlit chat interface setup
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.2-11b-32768"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Add image upload widget
uploaded_file = st.file_uploader("Upload an image of code or technical diagram", type=['png', 'jpg', 'jpeg'])
image_analysis = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing image...'):
        image_base64 = encode_image_from_upload(uploaded_file)
        if image_base64:
            image_analysis = analyze_image(image_base64)
            if image_analysis:
                st.success('Image analyzed successfully!')

with st.chat_message("assistant"):
    st.write("Hello! I can answer any questions you have about the Secure Agent Codebase! What can I help you with.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the codebase or uploaded image:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Get the RAG response with image analysis
        rag_response = perform_multimodal_rag(prompt, image_analysis)
        
        # Create the message placeholder
        message_placeholder = st.empty()
        full_response = ""
        
        # Create messages array including image analysis if available
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful coding assistant. Use the provided context from both the codebase and any uploaded images to provide comprehensive answers."
            }
        ]
        
        # Add image analysis context if available
        if image_analysis:
            messages.append({
                "role": "assistant",
                "content": f"Image Analysis Context:\n{image_analysis}"
            })
        
        # Add RAG response and chat history
        messages.extend([
            {"role": "assistant", "content": rag_response},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ])
        
        # Stream the response
        stream = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            stream=True,
        )

        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        # Update with final response
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})