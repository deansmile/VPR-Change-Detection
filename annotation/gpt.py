import openai
import os
import base64
from time import sleep
from tqdm import tqdm

# Set your OpenAI API key
openai.api_key = "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"

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

# t0_path="D:\\cv\\alvpr\\LLM4VPR\\test\\test\\t0"
# t1_path="D:\\cv\\alvpr\\LLM4VPR\\test\\test\\t1"
# f_fps=open("vl_cmu_cd_file_paths.txt")
# image_pairs=[]
# # i=0
# for line in f_fps.readlines():
#     s=line.strip()
#     image_pairs.append((t0_path+"\\"+s,t1_path+"\\"+s))
#     # i+=1
#     # if i>=5:
#     #     break
# print(len(image_pairs))

# source_txt_file = 'D:\\cv\\alvpr\\LLM4VPR\\db_q_test.txt'  # Ensure this path is correct
# base_image_folder = 'D:\\cv\\alvpr\\test_images\\test_images'

# # Initialize a list to store the image pairs
# image_pairs = []

# # Open the source text file and read the lines
# with open(source_txt_file, 'r') as file:
#     for line in file:
#         # Split the line into three strings
#         parts = line.strip().split()
#         if len(parts) >= 2:
#             # Get the image paths (first and second strings)
#             db_image_path = os.path.join(base_image_folder, os.path.basename(parts[0]))
#             query_image_path = os.path.join(base_image_folder, os.path.basename(parts[1]))
            
#             # Append the tuple of real image paths to the list
#             image_pairs.append((db_image_path, query_image_path))

# # Display the first 10 image pairs as a sample

# image_pairs is a list of tuples with each tuple storing the path of image pairs
image_pairs=[("db_1.jpg","q_1.jpg")]
generate_reranking(image_pairs,"text.txt")
