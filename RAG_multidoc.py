#!/usr/bin/env python
# coding: utf-8

# Importations de bibliothèques
import logging
import sys
import tkinter as tk
from tkinter import filedialog
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex, Settings, StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pyvis.network import Network
from IPython.display import display, HTML

# Configuration de la journalisation
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Définition des constantes
HF_TOKEN = "HUGGINGFACE_API_TOKEN" # VOTRE_API_HUGGINGFACE
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def setup_llm():
    """Configurer l'API d'inférence HuggingFace."""
    return HuggingFaceInferenceAPI(
        model_name=MODEL_NAME,
        token=HF_TOKEN
    )

def setup_embed_model():
    """Configurer le modèle d'embeddings."""
    return LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

def read_pdfs(pdf_files):
    """Lire le texte à partir des fichiers PDF.
    
    Args:
        pdf_files (list): Liste des chemins vers les fichiers PDF.
    
    Returns:
        list: Liste des documents extraits des PDF.
    """
    doc = SimpleDirectoryReader(input_files=pdf_files)
    text = doc.load_data()
    return text

def setup_service_context(llm):
    """Configurer le contexte de service pour le LLM."""
    Settings.llm = llm
    Settings.chunk_size = 1200

def setup_storage_context():
    """Configurer le contexte de stockage."""
    graph_store = SimpleGraphStore()
    return StorageContext.from_defaults(graph_store=graph_store)

def construct_knowledge_graph_index(documents, storage_context, embed_model):
    """Construire l'index du graphe de connaissances à partir des documents.
    
    Args:
        documents (list): Liste des documents.
        storage_context (StorageContext): Contexte de stockage.
        embed_model (LangchainEmbedding): Modèle d'embeddings.
    
    Returns:
        KnowledgeGraphIndex: Index du graphe de connaissances.
    """
    return KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=5,
        storage_context=storage_context,
        embed_model=embed_model,
        include_embeddings=True
    )
    

def create_query_engine(index):
    """Créer un moteur de requêtes à partir de l'index du graphe de connaissances.
    
    Args:
        index (KnowledgeGraphIndex): Index du graphe de connaissances.
    
    Returns:
        QueryEngine: Moteur de requêtes.
    """
    return index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
    )
    
def visualize_knowledge_graph(index, output_html="Knowledge_graph.html"):
    """Visualiser le graphe de connaissances et sauvegarder en fichier HTML.
    
    Args:
        index (KnowledgeGraphIndex): Index du graphe de connaissances.
        output_html (str): Nom du fichier HTML de sortie.
    """
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    net.show("graph.html")
    net.save_graph(output_html)
    display(HTML(filename=output_html))
    print("Le graphe de connaissances a été sauvegardé et affiché avec succès.")


def generate_response(query_engine, query):
    """Générer une réponse à la requête en utilisant le moteur de requêtes.
    
    Args:
        query_engine (QueryEngine): Moteur de requêtes.
        query (str): La requête utilisateur.
    
    Returns:
        Response: La réponse générée.
    """
    message_template = f"""<|system|> Please don't go out of context, respond according to the information you gather from the PDF documents if you don't know, say I don't know, there's no point in talking nonsense.
    Please provide a clear and well-structured answer in the form of an article, i.e. introduction, state of the art, conclusion.
    Please start your answer with this sentence based on my knowledge gained from the documents. then begins the introduction.
    </s>
    <|user|>
    Question: {query}
    Helpful Answer:
    </s>"""
    response = query_engine.query(message_template)
    return response

def save_response(response, filename="response.md"):
    """Enregistrer la réponse dans un fichier markdown.
    
    Args:
        response (Response): La réponse générée.
        filename (str): Nom du fichier de sortie.
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.response.split("<|assistant|>")[-1].strip())
    print("Le document a été sauvegardé avec succès")

def main(pdf_files, query):
    """Fonction principale pour exécuter tout le processus.
    
    Args:
        pdf_files (list): Liste des chemins vers les fichiers PDF.
        query (str): La requête utilisateur.
    """
    llm = setup_llm()
    embed_model = setup_embed_model()
    documents = read_pdfs(pdf_files)
    setup_service_context(llm)
    storage_context = setup_storage_context()
    index = construct_knowledge_graph_index(documents, storage_context, embed_model)
    query_engine = create_query_engine(index)
    response = generate_response(query_engine, query)
    save_response(response)
    visualize_knowledge_graph(index)

if __name__ == "__main__":
    # Ouvrir une fenêtre de dialogue pour sélectionner les fichiers PDF
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale
    pdf_files = filedialog.askopenfilenames(title="Sélectionnez des fichiers PDF", filetypes=[("PDF files", "*.pdf")])
    if not pdf_files:
        print("Aucun fichier sélectionné. Sortie...")
        sys.exit()

    # Demander à l'utilisateur d'entrer sa requête
    query = input("Entrez votre requête : ")

    # Exécuter le processus principal
    main(pdf_files, query)
