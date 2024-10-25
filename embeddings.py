import os
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
pdf_path = "[The Morgan Kaufmann Series in Computer Architecture and Design ] David A. Patterson, John L. Hennessy - Computer Organization and Design ARM Edition_ The Hardware Software Interface (Instructor's Edu Resource 1 of.pdf"

def text_formatter(text: str) -> str:
    """performs formatting on text"""
    clean_text = text.replace("\n"," ").strip()
    return clean_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({"page_number": page_number,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sent_count_raw": len(". "),
                                "page_token_count": len(text)/4,
                                "text": text})
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path = pdf_path)
pages_and_texts[:2]

from spacy.lang.en import English
#example using spacy
nlp = English()
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])

    # chunking sentences
num_sentence_chunks = 10
def split_list(input_list: list[str], slice_size: int = num_sentence_chunks) -> list[list[str]]:
  return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(item["sentences"])
    item["num_sentence_chunks"] = len(item["sentence_chunks"])

import re

pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        joined_chunk = " ".join(chunk).replace("\n", " ").strip()
        joined_chunk = re.sub(r'\.([A-Z])', r'.\1', joined_chunk)
        chunk_dict["chunk"] = joined_chunk

        chunk_dict["chunk_char_count"] = len(joined_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_chunk)/4# 1 token is ~4 char

        pages_and_chunks.append(chunk_dict)

df = pd.DataFrame(pages_and_chunks)
min_token_length = 15
for row in df[df["chunk_token_count"] <= min_token_length].sample(10).iterrows():
    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["chunk"]}')

pages_and_chunks_over_min = df[df["chunk_token_count"] > min_token_length].to_dict("records")
# pages_and_chunks_over_min = df[df["chunk_token_count"] > min_token_length].to_dict("records")
# pages_and_chunks_over_min[:2]
# print(random.sample(pages_and_chunks_over_min, k=1))
from sentence_transformers import SentenceTransformer
embedding_model= SentenceTransformer(model_name_or_path= 'all-mpnet-base-v2',
                                     device = 'cuda')
sentences = ["This is an example sentence related to embeddings ", "Each sentence is converted to embeddings", " i like segs!"]
embeddings = embedding_model.encode(sentences)
embedding_dict = dict(zip(sentences, embeddings))

# for sentence, embedding in embedding_dict.items():
#     print(f'Sentence: {sentence} | Embedding: {embedding}')
# Send the model to the GPU
embedding_model.to("cuda") # requires a GPU installed, for reference on my local machine, I'm using a NVIDIA RTX 4090
for item in tqdm(pages_and_chunks_over_min):
    item["embedding"] = embedding_model.encode(item["chunk"])
# Create embeddings one by one on the GPU
text_chunks = [item["chunk"] for item in pages_and_chunks_over_min]
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                               convert_to_tensor=True) # optional to return embeddings as tensor instead of array
# Save embeddings to file
text_chunk_embeddings
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

