import openai
import os
import base64
from time import sleep
from tqdm import tqdm

# Set your OpenAI API key
openai.api_key = ""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def generate_reranking(image_pairs, res_file_name, temperature=0.2):
    prompt = f"I have 2 images of outdoor scenes: A and B. In this task, you must not talk about weather, lighting, vehicles, people, or animals. Refrain from mentioning any elements that are not directly observable or are obscure. You must try your best to find all objects you see in A. For every object you see in A, you should try your best to find a corresponding object in B. Then, you must try your best to find all objects you see in B. For every object you see in B, you should try your best to find a corresponding object in A. In your response, just give me a numbered list of objects that you fail to find a match in B, and a numbered list of objects that you fail to find a match in A. Do not use prepositions."

    for pair in tqdm(image_pairs):
        query_image = encode_image(pair[0])
        db_image = encode_image(pair[1])
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role":"system",
                "content":[
                    {
                    "type":"text",
                        "text":f"You are an expert to analyze images. You need to read images carefully."
                    }
                ]
                },
                
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        
                        "url":f"data:image/png;base64,{query_image}"
                    },
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        
                        "url":f"data:image/png;base64,{db_image}"
                    },
                    },
                ],
                }
            ],
            max_tokens=4096,
            temperature=temperature
        )
        # folder_path = os.path.split(pair[0])[0]
        # result_file = os.path.join(folder_path, res_file_name)
        with open(res_file_name, 'a', encoding='utf-8') as f:
            f.write(os.path.basename(pair[0])+' '+os.path.basename(pair[1]) + "\n" + response.choices[0].message.content + "\n\n")
        sleep(1)
    # return result_file

f=open("/scratch/ds5725/alvpr/deep-visual-geo-localization-benchmark/matched_paths/db_q1_q_q3.txt")
image_pairs=[]
lines=f.readlines()
# i=0
for line in lines[100:200]:
    s=line.strip().split()
    image_pairs.append((s[0],s[1]))
    # i+=1
    # if i>=5:
    #     break
print(len(image_pairs))
# # Example usage
# image_pairs = [("D:\\cv\\alvpr\\LLM4VPR\\q_1.jpg", "D:\\cv\\alvpr\\LLM4VPR\\db_1.jpg")]
generate_reranking(image_pairs,"text_100_200.txt")
