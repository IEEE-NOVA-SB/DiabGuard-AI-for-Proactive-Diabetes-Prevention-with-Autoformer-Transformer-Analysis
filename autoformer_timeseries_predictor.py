#!/usr/bin/env python
# coding: utf-8

# ## Tarefas
# 
# ### pré processar dataset e ver como entra no modelo bem
# 
# ### como usar optuna ou outra forma de hypterparameter optimization para o modelo

# Tokenizing a dataset, especially for tasks beyond traditional natural language processing (NLP), such as time series prediction or processing numerical and categorical data, requires a tailored approach. The "best" way to tokenize such data depends on the specific characteristics of your dataset and the model you plan to use. Here are several strategies you can consider, each with its own advantages:
# 1. For Numerical Data
# 
#     Binning/Quantization: This involves dividing the range of numerical values into bins and assigning each bin a unique token. It's particularly useful for continuous data, helping to reduce the model's complexity by categorizing similar values.
# 
#     Normalization and Discretization: Normalize the data to a specific range (e.g., 0 to 1) and then discretize it into fixed intervals. Each interval can then be represented as a token. This approach maintains relative differences between values.
# 
#     Direct Encoding (with Caution): For numerical values that already take on a limited set of integers, you might consider using these directly as tokens. However, this can be challenging for models to interpret meaningfully and is generally less common.
# 
# 2. For Categorical Data
# 
#     One-hot Encoding: Convert each categorical value to a binary vector with a 1 in the position corresponding to the category. While not directly a "tokenization" method, it's a form of encoding that can be used prior to tokenization for models that require numerical input.
# 
#     Integer Encoding: Assign each unique category a unique integer. This is a straightforward form of tokenization but requires careful handling to avoid implying ordinal relationships where none exist.
# 
# 3. For Mixed Data Types
# 
#     Custom Tokenization: Develop a tokenization scheme that handles different types of data within your dataset, assigning unique tokens to different categories, bins, or normalized values across your features.
# 
#     Embedding Layers: Use embedding layers to map your tokenized numerical and categorical data to dense vectors of fixed size. This approach is particularly useful in deep learning models where embeddings can capture more nuanced relationships between tokens.
# 
# 4. Advanced Techniques
# 
#     Learned Embeddings: Instead of manually tokenizing and then applying a static embedding, some models can learn the most effective representations during training. This can be particularly powerful for complex datasets.
# 
#     Autoencoder-Based: For high-dimensional data, an autoencoder can learn a compressed representation of the data, which could then be discretized and used as tokens.
# 
# Choosing the Right Approach
# 
#     Understand Your Data: Analyze the characteristics of your data (e.g., range of numerical values, number of categories, presence of ordinal relationships) to choose the most appropriate tokenization method.
# 
#     Consider Your Model: Different models may require or benefit from specific types of input. For example, transformer-based models typically work with integer tokens and may benefit from embeddings, while models designed for continuous input might handle normalized numerical values more directly.
# 
#     Experiment and Validate: Often, the best approach is found through experimentation. Try different tokenization methods and evaluate your model's performance to find the most effective strategy.
# 
# 
# 
# --------
# 
# 
# 
# Yes, each value in a column can be represented as a token, but this process requires careful consideration of how the data is encoded and prepared for the model. Tokenization typically refers to converting text data into numerical tokens that a machine learning model can understand. However, when dealing with numerical or non-textual data, such as values in a column of a time series dataset, the concept of "tokenization" can be adapted to suit the data type.
# 
# For non-textual data, tokenization might involve different strategies, such as:
# 
#     Discretization: Numerical values can be discretized into bins, and each bin can be represented as a token. This approach is useful for continuous data, allowing the model to treat each bin as a categorical value.
# 
#     Categorical Encoding: If the values in a column are categorical (including discretized numerical values), each unique category can be assigned a unique token. This is straightforward for inherently categorical data.
# 
#     Embedding Numerical Values: For models that accept numerical input directly (e.g., traditional neural networks for time series prediction), each numerical value could be considered an "implicit token" that is fed into an embedding layer. This is not tokenization in the traditional sense but rather a method of preparing numerical data for model input.
# 
#     Custom Encoding Schemes: Depending on the nature of the data and the model, custom encoding schemes may be developed. For example, in time series data, patterns or sequences of values might be encoded into tokens that represent specific states or transitions.
# 
# For the case of using AutoModelForCausalLM with time series data, if you wish to treat each value as a token, you would need to:
# 
#     Encode the Values: Convert each numerical value into a discrete token. This could involve mapping each unique value to a unique integer or discretizing the range of values and mapping each range to a unique integer.
# 
#     Use an Appropriate Tokenizer: Although the AutoTokenizer is designed for text data, you would need to ensure that it can handle your encoded values appropriately. This might involve custom preprocessing steps to ensure that the tokenizer interprets your numerical tokens correctly.
# 
#     Consider Model Compatibility: Ensure that the model can interpret and learn from the encoded tokens effectively. Since causal language models are designed for text, using them for numerical time series data in this way is unconventional and may require experimentation and adjustment to achieve good results.
# 
# It's important to note that treating each value as a token is more common in text processing. When applying this concept to numerical data for a model originally designed for natural language processing (NLP), it's essential to carefully consider how the data is represented and whether this approach aligns with the model's capabilities and the goals of your analysis.

# In[ ]:


import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ucimlrepo import fetch_ucirepo 
import optuna


# In[ ]:


# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 

print("Number of features:" + str(X.shape[1]))

print(X.head(10))

print("Number of targets:" + str(y.shape[1]))

print(y.head(10))


# In[ ]:


for column in X.columns:
    unique_values = X[column].unique()
    print(f"Unique values in column '{column}': {unique_values} \n")


# In[ ]:


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# In[ ]:


class AutoformerTimeSeriesPredictor(pl.LightningModule):
    def __init__(self, model_name='facebook/autoformer', sequence_length=128):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_length = sequence_length

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# In[ ]:


dataset = TimeSeriesDataset(sequences, targets)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AutoformerTimeSeriesPredictor()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader)

