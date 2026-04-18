# Language Modelling: Name Generation

This project compares four deep learning models trained on Indian names:

- `MLP`
- `RNN`
- `LSTM`
- `GRU`

The Streamlit app loads the trained models, shows training/evaluation plots, compares model scores, and generates new names from a selected model.

## Features

- Compare `MLP`, `RNN`, `LSTM`, and `GRU` in one interface
- View training loss and evaluation loss plots
- Inspect train, dev, and test loss/perplexity in tabular form
- Generate sample names by choosing a model and starting character

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── Images/
│   ├── loss_vs_epoch_train_plot.png
│   └── loss_vs_epoch_train_Eval_plot.png
├── Trained_models/
│   ├── mlp_model.pkl
│   ├── rnn_model.pkl
│   ├── lstm_model.pkl
│   └── gru_model.pkl
└── Training/
    ├── Language_Modelling.ipynb
    └── Indian_Names.txt
```

## Run The App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Streamlit:

```bash
streamlit run app.py
```

## Training Plots

### Loss vs Epoch

![Training Loss](Images/loss_vs_epoch_train_plot.png)

### Train vs Dev vs Test Loss

![Train Dev Test Loss](Images/loss_vs_epoch_train_Eval_plot.png)

## Model Scores

The table below matches the scores shown in the Streamlit app and comes from the evaluation output in `Training/Language_Modelling.ipynb`.

| Model | Train Loss | Train Perplexity | Dev Loss | Dev Perplexity | Test Loss | Test Perplexity |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 1.7504 | 5.7571 | 1.8506 | 6.3634 | 1.8411 | 6.3032 |
| RNN | 1.7712 | 5.8781 | 1.8047 | 6.0784 | 1.7988 | 6.0424 |
| LSTM | 1.7042 | 5.4972 | 1.7559 | 5.7887 | 1.7464 | 5.7339 |
| GRU | 1.7040 | 5.4957 | 1.7581 | 5.8012 | 1.7496 | 5.7524 |

## Notes

- The trained checkpoints in `Trained_models/` are loaded by the Streamlit app.
- The plots are stored in the `Images/` folder.
- Training, evaluation, and model-saving steps are available in `Training/Language_Modelling.ipynb`.
