
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
import os

def summarizer(filename):
    
    """
    This function takes an amazon transcript and returns a JSON containing "Point Clefs", "Objectifs", and "Sujets".
    The JSON is output in the Results folder.
    Args
        filename: the filename (excluding the .txt) of the file to be summarised
    Returns
        ../Results/result_{filename}.txt
    """
    
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name='gpt-4', openai_api_key=OPENAI_API_KEY)

    # Map
    map_template = """Le suivant est un ensemble de documents
    {docs}
    Sur la base de cette liste de docs, veuillez identifier les thèmes principaux 
    Réponse utile :"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
    # Reduce
    reduce_template = """ Vous êtes un expert de renommée mondiale en pédagogie.
    Voici un résumé détaillé d'une leçon. Vous devez en distiller l'essence selon les critères suivants:
    - Points Clés: Commencez par un paragraphe introductif concis résumant l'ensemble de la leçon. 
    Ensuite, listez les 5 enseignements les plus importants de la leçon. 
    Chaque point clé doit être développé dans un bref paragraphe.
    - Objectifs de la leçon: Veuillez articuler les principaux objectifs de la leçon, reflétant votre expertise en matière de conception pédagogique.
    - Sujets: Essayez d'établir une structure exhaustive en réseau ou en arborescence qui encapsule les thèmes principaux et les sous-thèmes de la leçon. 
    Ceci devrait donner une vue claire et hiérarchisée du contenu de la leçon.

    Vous répondrez en format JSON, dont les clefs seront "Point Clefs", "Objectifs",  et "Sujets"
    Ensemble de résumés: {doc_summaries}
    Réponse en JSON:
    """
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name='gpt-4',
        chunk_size=1000, chunk_overlap=0
    )
    
    with open(f"../Resources/{filename}.txt", 'r') as file:
            data = json.load(file)
    text = data['results']['transcripts'][0]['transcript']
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    print("Loading...")
    result = map_reduce_chain.run(docs)
    with open(f"../Results/result_{filename}.txt", "w") as file:
        file.write(result)
    print(f"Results successfully written as text to result_{filename}.txt.")
    