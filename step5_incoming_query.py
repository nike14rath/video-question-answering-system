import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


df = joblib.load('Dataframe_with_embeddings.joblib')
# print(df)

# finding the similarity here

# let's create a functions for creating the the embeddings here 


def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
        # "input": 
    })
    embedded = r.json()
    
    return embedded


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    # print(response)
    return response
    



# Asking the Questions for the Query 

incoming_query = input("Ask a Question:")

    
# taking the first values here that's why [0]
embedding = create_embeddings([incoming_query])
Query_vector = embedding['embeddings'][0]
print(embedding['embeddings'])




# let's find the similarities in between the embeddings
similarity = cosine_similarity(np.vstack(df['embedding']), [Query_vector]).flatten()

top_results = 5  # variable to get the top results that are matching with our program

# print(similarity)
max_index = similarity.argsort()[::-1][0:top_results]
print(max_index)

new_df = df.iloc[max_index]
print(new_df[['Video','start', 'end', "text"]])



prompt = f"""Here are the Vedios containting the text, Vedio, Time Stamp: {new_df[['Video','start', 'end', "text"]].to_json()}
The User asked this specific Questions related to Vedios,give the top three results in easy to navigate format if possible, you have the Answer how much content is taught, in which vedio title and at what Time Stamp(give the time in minutes formatW), Guide the User to that Particular Vedio. If User asked unrealted Questions tell them, that you can answer questions related to vedio only: {incoming_query}
"""


# let's check how the prompt will look like before it is going to LLM

with open("prompt.txt",'w') as f:
    f.write(prompt)
    
response = inference(prompt)['response']
print(response)


# saving the response in the txt file here
with open("response.txt",'w') as f:
    f.write(response)