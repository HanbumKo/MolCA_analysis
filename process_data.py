import torch
import random
import string

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="sk-xx")

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)


def generate_random_text(length=10):
    characters = string.ascii_letters + string.digits
    random_text = ''.join(random.choice(characters) for _ in range(length))
    return random_text


data, slices = torch.load('data/PubChem324kV2/pretrain.pt')

text_new = []
for i, d in tqdm(enumerate(data.text)):
    random_text = generate_random_text(10)
    text_new.append("The molecule is a " + random_text)
    # if i == 10:
    #     break

data.text = text_new
torch.save((data, slices), 'data/PubChem324kV2_keyword_random/pretrain.pt')




