from PyPDF2 import PdfReader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from constants import openai_key
from langchain.chains import LLMChain, SequentialChain
os.environ['GROQ_API_KEY'] = openai_key
llm = ChatGroq(api_key=os.environ['GROQ_API_KEY'], model_name="llama-3.1-8b-instant", temperature=0.1 )
template='''from the {text} give me the details like name of the customer, contact number of the customer, policy_number, insurance_company_name, type_of_policy, start_date of policy, expiry_date of policy, registration_number, engine_number, chassis_number, body_type, vehicle_make, model, manufacturing_year, total_premium_paid, address of the consumer, in the format of a json file, no other text needed, and no subdictonaries in associated with key just text'''
chain1=LLMChain(llm=llm, prompt=PromptTemplate(template=template,input_variables=['text']), output_key='dict')
def extract_details(chain=chain1, text=None):
    '''provide your input text'''
    try:    
        return json.loads(re.search(r"```json\n(.*?)\n```",chain.run(text), re.DOTALL).group(1))
        
    except  Exception as e:
        return e
