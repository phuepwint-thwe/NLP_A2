import streamlit as st
import torch
import pickle
from models.classes import LSTMLanguageModel
from library.utils import generate
import torchtext
from collections import Counter
from torchtext.vocab import Vocab

# Load Data
data = pickle.load(open('models/Data.pkl', 'rb'))

# Tokenizer
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Importing training data
Data = pickle.load(open('./models/Data.pkl', 'rb'))
vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Define Model Arguments
args = {
    "vocab_size": len(vocab),
    "emb_dim": 1024,  # Adjust embedding size
    "hid_dim": 1024,    # Hidden layer size
    "num_layers": 2,      # Number of LSTM layers
    "dropout_rate": 0.65,       # Dropout rate
}

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args['vocab_size'] = 8547 # This is the vocab size of the pretrained model
model = LSTMLanguageModel(**args).to(device)

# # Load Pretrained Weights (if available)
# try:
#     model.load_state_dict(torch.load('models/best-val-lstm_lm.pt', map_location=device))
#     st.write("Loaded pretrained model weights successfully!")
# except FileNotFoundError:
#     st.write("Pretrained weights not found. The model is untrained.")

# Streamlit App
# Streamlit App
st.title("A2 - LSTM Text Generator")
st.write("Please provide a prompt about the book **'War and Peace'** to generate a continuation of the text.")

# User input
prompt = st.text_input("Enter a prompt:", "")

# Parameters selection
temperature = st.selectbox("Select creativity level (temperature):", [0.5, 0.7, 0.75, 0.8, 1.0], index=2)
max_seq_len = st.selectbox("Select maximum sequence length:", [10,20,30], index=1)

# Generate button
if st.button("Generate the text!"):
    if prompt.strip():
        with st.spinner("Generating text..."):
            generation = generate(prompt.strip(), max_seq_len, temperature, model, tokenizer, vocab, device)
            generated_text = " ".join(generation)
        st.success("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")
