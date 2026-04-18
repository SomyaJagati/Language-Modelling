import streamlit as st
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path

st.set_page_config(layout="centered", initial_sidebar_state="collapsed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Trained_models"
IMAGES_DIR = BASE_DIR / "Images"
TRAINING_DIR = BASE_DIR / "Training"

EMBEDDING_DIM = 10
BLOCK_SIZE = 10
HIDDEN_SIZE = 200
MODEL_METRICS = {
    "MLP": {
        "Train Loss": 1.7504,
        "Train Perplexity": 5.7571,
        "Dev Loss": 1.8506,
        "Dev Perplexity": 6.3634,
        "Test Loss": 1.8411,
        "Test Perplexity": 6.3032,
    },
    "RNN": {
        "Train Loss": 1.7712,
        "Train Perplexity": 5.8781,
        "Dev Loss": 1.8047,
        "Dev Perplexity": 6.0784,
        "Test Loss": 1.7988,
        "Test Perplexity": 6.0424,
    },
    "LSTM": {
        "Train Loss": 1.7042,
        "Train Perplexity": 5.4972,
        "Dev Loss": 1.7559,
        "Dev Perplexity": 5.7887,
        "Test Loss": 1.7464,
        "Test Perplexity": 5.7339,
    },
    "GRU": {
        "Train Loss": 1.7040,
        "Train Perplexity": 5.4957,
        "Dev Loss": 1.7581,
        "Dev Perplexity": 5.8012,
        "Test Loss": 1.7496,
        "Test Perplexity": 5.7524,
    },
}


class MLP(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(block_size * embedding_dim, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb_flat = emb.reshape(emb.shape[0], -1)
        h = self.relu(self.linear1(emb_flat))
        logits = self.linear2(h)
        return logits


class RNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths, hidden=None):
        emb = self.embedding(x)
        packed_emb = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden_out = self.rnn(packed_emb, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        logits = self.linear(out)
        return logits, hidden_out


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths, hidden=None):
        emb = self.embedding(x)
        packed_emb = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden_out, cell_out) = self.lstm(packed_emb, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        logits = self.linear(out)
        return logits, (hidden_out, cell_out)


class GRU(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.gru = torch.nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths, hidden=None):
        emb = self.embedding(x)
        packed_emb = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden_out = self.gru(packed_emb, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        logits = self.linear(out)
        return logits, hidden_out


@st.cache_resource
def build_vocabulary():
    """Build character-to-index and index-to-character mappings."""
    words = open(TRAINING_DIR / "Indian_Names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    if "." in chars:
        chars.remove(".")
    
    str_to_idx = {s: i + 2 for i, s in enumerate(chars)}
    str_to_idx["."] = 1
    str_to_idx["<PAD>"] = 0
    
    idx_to_str = {i: s for s, i in str_to_idx.items()}
    
    return str_to_idx, idx_to_str


@st.cache_resource
def load_models():
    """Load all trained models from locally trusted checkpoint files."""
    str_to_idx, idx_to_str = build_vocabulary()
    
    models = {}
    model_files = {
        "MLP": "mlp_model.pkl",
        "RNN": "rnn_model.pkl",
        "LSTM": "lstm_model.pkl",
        "GRU": "gru_model.pkl"
    }
    
    for model_name, filename in model_files.items():
        model_path = MODELS_DIR / filename
        try:
            model = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model.eval()
            models[model_name] = model
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
    
    return models, str_to_idx, idx_to_str


def sample_names(model, model_name, start_char, str_to_idx, idx_to_str, num_samples=5):
    """Generate names from a model."""
    start_idx = str_to_idx.get(start_char, str_to_idx.get("a", 2))
    end_idx = str_to_idx["."]
    
    min_len = 3
    temperature = 0.8
    max_len = 30
    block_size = BLOCK_SIZE
    
    names = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            if model_name == "MLP":
                context = [0] * block_size
                context[-1] = start_idx
                out = [start_idx]
                
                for _ in range(max_len):
                    x = torch.tensor([context], dtype=torch.long, device=DEVICE)
                    logits = model(x)
                    probs = F.softmax(logits / temperature, dim=1)
                    ix = torch.multinomial(probs, num_samples=1).item()
                    
                    if ix == end_idx and len(out) > min_len:
                        break
                    
                    if ix != end_idx:
                        out.append(ix)
                    
                    context = context[1:] + [ix]
            
            else:
                out = [start_idx]
                ix = start_idx
                hidden = None
                
                for _ in range(max_len):
                    x = torch.tensor([[ix]], dtype=torch.long, device=DEVICE)
                    lengths = torch.tensor([1], device=DEVICE)
                    
                    logits, hidden = model(x, lengths, hidden)
                    probs = F.softmax(logits[:, -1, :] / temperature, dim=1)
                    ix = torch.multinomial(probs, num_samples=1).item()
                    
                    if ix == end_idx and len(out) > min_len:
                        break
                    
                    if ix != end_idx:
                        out.append(ix)
            
            name_str = "".join([idx_to_str[i] for i in out if i != 0])
            names.append(name_str)
    
    return names


@st.cache_data
def get_metrics_table():
    """Return model metrics captured from the training notebook evaluation output."""
    rows = []
    for model_name, metrics in MODEL_METRICS.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    return rows


def main():
    st.title("Language Modelling: Name Generation")
    st.write(
        "This application compares four deep learning models (MLP, RNN, LSTM, and GRU) "
        "trained on Indian names. Use the controls below to generate names from different models."
    )
    
    models, str_to_idx, idx_to_str = load_models()
    
    if not models:
        st.error("No models could be loaded. Please check the Trained_models directory.")
        return
    
    st.divider()
    
    st.subheader("Model Training Performance")

    st.write("Train, dev, and test loss/perplexity comparison from the notebook evaluation.")
    st.dataframe(get_metrics_table(), use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if (IMAGES_DIR / "loss_vs_epoch_train_plot.png").exists():
            st.image(
                str(IMAGES_DIR / "loss_vs_epoch_train_plot.png"),
                caption="Training Loss"
            )
    
    with col2:
        if (IMAGES_DIR / "loss_vs_epoch_train_Eval_plot.png").exists():
            st.image(
                str(IMAGES_DIR / "loss_vs_epoch_train_Eval_plot.png"),
                caption="Train vs Dev vs Test Loss"
            )
    
    st.divider()
    
    st.subheader("Generate Names")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_char_input = st.text_input("Enter starting character", value="a", max_chars=1)
        start_char = start_char_input if start_char_input else "a"
    
    with col2:
        selected_model = st.selectbox("Select model", list(models.keys()))
    
    st.divider()
    
    if "generated_names" not in st.session_state:
        st.session_state.generated_names = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate", key="generate_btn"):
            with st.spinner("Generating names..."):
                new_names = sample_names(
                    models[selected_model],
                    selected_model,
                    start_char,
                    str_to_idx,
                    idx_to_str,
                    num_samples=5
                )
                st.session_state.generated_names = new_names
    
    with col2:
        if st.button("Generate More", key="generate_more_btn"):
            if st.session_state.generated_names:
                with st.spinner("Generating more names..."):
                    new_names = sample_names(
                        models[selected_model],
                        selected_model,
                        start_char,
                        str_to_idx,
                        idx_to_str,
                        num_samples=5
                    )
                    st.session_state.generated_names.extend(new_names)
    
    if st.session_state.generated_names:
        st.divider()
        st.subheader("Generated Names")
        
        names_text = "\n".join(st.session_state.generated_names)
        st.text(names_text)
        
        if st.button("Clear", key="clear_btn"):
            st.session_state.generated_names = []
            st.rerun()


if __name__ == "__main__":
    main()
