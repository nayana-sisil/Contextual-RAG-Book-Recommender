import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

#getting the thumbnail,if not add default

books = pd.read_csv("./dataset/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

#dealing with description  and saving in vector db

books["tagged_description"].to_csv("tagged_description.txt",
                                   sep = "\n",
                                   index = False,
                                   header = False)

with open("tagged_description.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

books = re.split(r'(?=\d{13} )', raw_text)
books = [b.strip() for b in books if b.strip()]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

db_books = Chroma.from_texts(
    books,
    embedding=embeddings,
    collection_name="books_test"
)




