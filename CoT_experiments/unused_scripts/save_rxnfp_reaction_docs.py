import re
import os
keys_dict = {}
with open("CoT_experiments/keys.txt", "r") as f:
    keys = f.readlines()
    for key_line in keys:
        key_name, key_val = key_line.split("=")
        keys_dict[key_name.strip()] = key_val.strip()
        os.environ[key_name.strip()] = key_val.strip()
import json
import requests

from collections import deque, defaultdict
from bs4 import BeautifulSoup, Tag
from fake_useragent import UserAgent

user_agent = UserAgent()
headers = {'User-Agent': user_agent.random}

# Load CoT_experiments/data/rxnfp/rxnclass2name.json
with open("CoT_experiments/data/rxnfp/rxnclass2name.json", "r") as f:
    rxnclass2name = json.load(f)
with open("CoT_experiments/data/rxnfp/rxnclass2id.json", "r") as f:
    rxnclass2id = json.load(f)
reaction_types = [rxnclass2name[rxnclass] for rxnclass in rxnclass2id.keys()] # 50 reaction types


def preprocess(soup):
    # soup = BeautifulSoup(raw_html, 'html.parser')
    # Remove all scripts from the soup
    for script in soup.find_all('script'):
        script.extract()
    
    # Remove all style tags from the soup
    for style in soup.find_all('style'):
        style.extract()
    
    # Remove all images from the soup
    for img in soup.find_all('img'):
        img.extract()

    # Remove all figures from the soup
    for figure in soup.find_all('figure'):
        figure.extract()

    # Remove all audio from the soup
    for audio in soup.find_all('audio'):
        audio.extract()
    
    # Remove all video from the soup
    for video in soup.find_all('video'):
        video.extract()

    # Remove all table from the soup
    for table in soup.find_all('table'):
        table.extract()

    # Remove all button from the soup
    for button in soup.find_all('button'):
        button.extract()

    for a in soup.find_all('a'):
        a.unwrap()

    for span in soup.find_all('span'):
        span.unwrap()
    
    return soup


def find_tags_with_sentences_not_split_by_newlines(html):
    citation_words = ["PMID", "PMCID", "DOI", "ISBN", "ISSN", "Bibcode", "OCLC", "LCCN", "JSTOR", "S2CID", "doi: "]
    
    # HTML parsing
    soup = BeautifulSoup(html, 'html.parser')
    soup = preprocess(soup)
    
    # Initialize deque for BFS search
    queue = deque([soup])
    results = []

    while queue:
        current_node = queue.popleft()
        
        # Explore only if the current node is Tag
        if isinstance(current_node, Tag):
            text = current_node.get_text().strip()
            sentences = re.split(r'[.!?] ', text)
            # sentences = [sentence.strip() for sentence in sentences]
            
            # Find the case where two or more sentences are not split by newline characters
            if len(sentences) > 1 and all('\n' not in sentence and len(sentence) > 5 for sentence in sentences) and not any(cw in text for cw in citation_words):
                results.append(current_node)
                current_node.extract()
                continue
            
            # Add child nodes to the queue
            for child in current_node.children:
                queue.append(child)
    
    return results



for react_type in reaction_types:
    react_type_replaced = react_type.replace(" ", "_")
    if os.path.exists(f"CoT_experiments/data/reaction_docs/search_results/{react_type_replaced}.json"):
        results = json.load(open(f"CoT_experiments/data/reaction_docs/search_results/{react_type_replaced}.json", "r"))
    else:
        search_url = f"https://www.googleapis.com/customsearch/v1?key={keys_dict['SEARCH_API_KEY']}&cx={keys_dict['SEARCH_ENGINE_ID']}&q={react_type}&excludeTerms=filetype:pdf-filetype:ppt-filetype:doc-filetype:xls"
        response = requests.get(search_url, headers=headers)
        use_google_search = True
        results = response.json()
        # Save to f"CoT_experiments/data/reaction_docs/search_results/{react_type_replaced}.json"
        with open(f"CoT_experiments/data/reaction_docs/search_results/{react_type_replaced}.json", "w") as file:
            file.write(json.dumps(results, indent=4))

    contexts = []
    if "items" in results:
        for item in results['items']:
            url = item['link']
            url_replaced = url.replace(" ", "_").replace("/", "__").strip()
            url_domain = "/".join("".join(url.split("//")[1:]).split("/")[:-1])
            url_domain_replaced = url_domain.replace(" ", "_").replace("/", "__").strip()
            # GET request to url
            try:
                response = requests.get(url, headers=headers, timeout=2)
            except:
                continue
            if response.status_code != 200:
                continue
            else:
                raw_html = response.text
                try:
                    tags = find_tags_with_sentences_not_split_by_newlines(raw_html)
                except:
                    continue
                contexts += [tag.get_text().strip() for tag in tags]
    elif "error" in results:
        print(f"Error: {results['error']['message']}")
    
    context_text = "\n\n".join(contexts)
    # Save context_text to f"CoT_experiments/data/reaction_docs/chunks/{react_type_replaced}.txt"
    with open(f"CoT_experiments/data/reaction_docs/chunks/{react_type_replaced}.txt", "w") as f:
        f.write(context_text)
    print(f"Saved context_text to {react_type_replaced}.txt")