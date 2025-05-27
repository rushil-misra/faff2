import chromadb
import pandas as pd
import re
import hashlib
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction




chat_1 = pd.read_csv('chats/chat_1.csv')

chat_1_new = chat_1[['message_id','type','text_body','from_number','media_url','context','created_at']]

chat_1_new['created_at'] = pd.to_datetime(chat_1_new['created_at'])


client = chromadb.Client()
embedding_func = OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY')) 

client = chromadb.PersistentClient(path="./memory_db")
collection = client.get_or_create_collection(name="user_facts")

def user_facts(j,text):

    if bool(re.search(r"\b(I|my|me|we)\b", str(text), re.IGNORECASE)):
        extracted_fact = llm.invoke(f'your task is to determine if the following message gives any information about the user {text}. If it does, write it in simple manner. If it does not, simply write no').content
        if re.fullmatch(r"\b(no)\b", extracted_fact, re.IGNORECASE):
            print(j,'  ::  ',"Not useful")
            return None
        else:
            print(j,'  ::  ',extracted_fact)
            return extracted_fact
    
            
j = 0

for i in range(len(chat_1_new)):
    text = chat_1_new['text_body'][i]
    response = user_facts(j,text)
    j +=1
    if response is None:
        continue
    
    collection.add(
        ids=[f"user_fact_{chat_1_new['message_id'][i]}"],  
        documents=[response],
        metadatas=[{
            "message_id": str(chat_1_new['message_id'][i]),
            "from_number": str(chat_1_new['from_number'][i]),
            "text_body": str(text),
            "created_at": str(chat_1_new['created_at'][i])
        }]
    )


results = collection.get()

rows = []
for i in range(len(results["ids"])):
    row = results["metadatas"][i].copy()
    row["fact"] = results["documents"][i]
    row["id"] = results["ids"][i]
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("user_facts_export.csv", index=False)
print("Exported to user_facts_export.csv")