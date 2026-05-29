import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datasets

from pathlib import Path
from numpy.random import normal, seed, uniform
from datasets import Dataset, DatasetDict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# 1. Kaggle setup
# ============================================================

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
else:
    creds = ''  # paste your kaggle.json contents here if needed
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()
    if not cred_path.exists() and creds:
        cred_path.parent.mkdir(exist_ok=True)
        cred_path.write_text(creds)
        cred_path.chmod(0o600)

    path = Path('us-patent-phrase-to-phrase-matching')
    if not path.exists():
        import kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)

# ============================================================
# 2. Load & prepare data
# ============================================================

df = pd.read_csv(path / 'train.csv')
print(df)
print(df.describe(include='object'))

df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
print(df.input.head())

# ============================================================
# 3. Tokenization
# ============================================================

model_nm = 'microsoft/deberta-v3-small'

tokz = AutoTokenizer.from_pretrained(model_nm)

print(tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!"))
print(tokz.tokenize("A platypus is an ornithorhynchus anatinus."))

ds = Dataset.from_pandas(df)

def tok_func(x):
    return tokz(x["input"])

tok_ds = ds.map(tok_func, batched=True)

row = tok_ds[0]
print(row['input'], row['input_ids'])
print("Token ID for '▁of':", tokz.vocab['▁of'])

tok_ds = tok_ds.rename_columns({'score': 'labels'})

# ============================================================
# 4. Eval/test set
# ============================================================

eval_df = pd.read_csv(path / 'test.csv')
print(eval_df.describe())

eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

# ============================================================
# 5. Validation set split
# ============================================================

dds = tok_ds.train_test_split(0.25, seed=42)
print(dds)

# ============================================================
# 6. Overfitting/underfitting demo (optional, no internet needed)
# ============================================================

def f(x): return -3 * x**2 + 2 * x + 20

def plot_function(f, min=-2.1, max=2.1, color='r'):
    x = np.linspace(min, max, 100)[:, None]
    plt.plot(x, f(x), color)

np.random.seed(42)

def noise(x, scale): return normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1 + noise(x, mult)) + noise(x, add)

x_demo = np.linspace(-2, 2, num=20)[:, None]
y_demo = add_noise(f(x_demo), 0.2, 1.3)

def plot_poly(degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_demo, y_demo)
    plt.scatter(x_demo, y_demo)
    plot_function(model.predict)

plot_function(f); plt.title("True function"); plt.show()
plot_poly(1);     plt.title("Underfit (degree 1)"); plt.show()
plot_poly(10);    plt.title("Overfit (degree 10)"); plt.show()
plot_poly(2); plot_function(f, color='b'); plt.title("Just right (degree 2)"); plt.show()

# ============================================================
# 7. Correlation demo (California Housing — skipped if offline)
# ============================================================

def corr(x, y): return np.corrcoef(x, y)[0][1]

def show_corr(df, a, b):
    x, y = df[a], df[b]
    plt.scatter(x, y, alpha=0.5, s=4)
    plt.title(f'{a} vs {b}; r: {corr(x, y):.2f}')
    plt.show()

try:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    housing = housing['data'].join(housing['target']).sample(1000, random_state=52)
    print(housing.head())
    np.set_printoptions(precision=2, suppress=True)
    print(np.corrcoef(housing, rowvar=False))
    show_corr(housing, 'MedInc', 'MedHouseVal')
    show_corr(housing, 'MedInc', 'AveRooms')
    subset = housing[housing.AveRooms < 15]
    show_corr(subset, 'MedInc', 'AveRooms')
    show_corr(subset, 'MedHouseVal', 'AveRooms')
    show_corr(subset, 'HouseAge', 'AveRooms')
except Exception as e:
    print(f"Skipping California Housing demo (download failed): {e}")

# ============================================================
# 8. Metrics
# ============================================================

def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}

# ============================================================
# 9. Training
# ============================================================

bs     = 128
epochs = 4
lr     = 8e-5

args = TrainingArguments(
    'outputs',
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to='none',
)

model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

trainer = Trainer(
    model,
    args,
    train_dataset=dds['train'],
    eval_dataset=dds['test'],
    tokenizer=tokz,
    compute_metrics=corr_d,
)

trainer.train()

# ============================================================
# 10. Predictions & submission
# ============================================================

preds = trainer.predict(eval_ds).predictions.astype(float)
print("Raw predictions:", preds)

preds = np.clip(preds, 0, 1)
print("Clipped predictions:", preds)

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds,
})

submission.to_csv('submission.csv', index=False)
print("submission.csv saved!")