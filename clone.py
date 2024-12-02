from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from git import Repo
from langchain.schema import Document
from pinecone import Pinecone
import os
import ast
import re

def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory."""
    repo_name = repo_url.split("/")[-1]
    repo_path = f"/content/{repo_name}"
    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

path = clone_repository("https://github.com/CoderAgent/SecureAgent")

SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

def extract_python_functions(content, file_path):
    """Extract functions from Python code using AST."""
    functions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Get the source code lines for the function
                func_lines = content.split('\n')[node.lineno-1:node.end_lineno]
                func_code = '\n'.join(func_lines)
                
                # Get docstring if it exists
                docstring = ast.get_docstring(node) or ""
                
                functions.append({
                    "name": node.name,
                    "content": func_code,
                    "docstring": docstring,
                    "file_path": file_path,
                    "language": "python"
                })
    except Exception as e:
        print(f"Error parsing Python file {file_path}: {str(e)}")
    return functions

def extract_js_functions(content, file_path):
    """Extract functions from JavaScript/TypeScript code using regex."""
    functions = []
    
    # Patterns for different function declarations
    patterns = [
        (r'function\s+(\w+)\s*\([^)]*\)\s*{([^}]*)}', 'function'),
        (r'const\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*{([^}]*)}', 'const function'),
        (r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{([^}]*)}', 'arrow function'),
        (r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*([^;]+)', 'arrow function')
    ]
    
    for pattern, func_type in patterns:
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        for match in matches:
            func_name = match.group(1)
            func_code = match.group(0)
            functions.append({
                "name": func_name,
                "content": func_code,
                "file_path": file_path,
                "language": "javascript",
                "type": func_type
            })
    
    return functions

def get_file_content(file_path, repo_path):
    """Extract functions from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, repo_path)
        extension = os.path.splitext(file_path)[1]
        
        if extension == '.py':
            return extract_python_functions(content, rel_path)
        elif extension in {'.js', '.jsx', '.ts', '.tsx'}:
            return extract_js_functions(content, rel_path)
        else:
            return []
            
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []

def get_main_files_content(repo_path: str):
    """Get functions from all supported files in the repository."""
    all_functions = []

    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    functions = get_file_content(file_path, repo_path)
                    all_functions.extend(functions)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return all_functions

# Initialize Pinecone
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

# Connect to your Pinecone index
pinecone_index = pc.Index("codebase-rag")

# Extract all functions
functions = get_main_files_content(path)

# Create documents for each function
documents = []
for func in functions:
    # Create a formatted content string
    content = f"""
    File: {func['file_path']}
    Function: {func['name']}
    Language: {func['language']}
    {func.get('docstring', '')}

    Code:
    {func['content']}
        """.strip()
        
    doc = Document(
        page_content=content,
        metadata={
            "source": func['file_path'],
            "function_name": func['name'],
            "language": func['language'],
            "type": func.get('type', 'function')
        }
    )
    documents.append(doc)
    print(f"Added function: {func['name']} from {func['file_path']}")

# Create the vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(),
    index_name="codebase-rag",
    namespace="https://github.com/CoderAgent/SecureAgent"
)