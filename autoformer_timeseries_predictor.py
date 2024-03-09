#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import torch
import torch.nn as nn


import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import TrainingArguments, Trainer
from transformers import AutoformerConfig, AutoformerModel
import optuna

# Fetch dataset
from ucimlrepo import fetch_ucirepo


# In[ ]:


dataset = fetch_ucirepo(id=296)
X, y = dataset.data.features, dataset.data.targets


# In[ ]:


print(X)


# In[ ]:


print(y)


# In[ ]:


# Replace NaN in numerical columns with the median and in categorical columns with 'Unknown'
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col].fillna(X[col].median(), inplace=True)
    elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
        X[col].fillna('Unknown', inplace=True)  # Or use mode: X[col].fillna(X[col].mode()[0], inplace=True)

# Encode categorical features using LabelEncoder
label_encoders = {col: LabelEncoder().fit(X[col]) for col in X.select_dtypes(include=['object', 'category']).columns}
X = X.apply(lambda col: label_encoders[col.name].transform(col) if col.name in label_encoders else col)

print(X)


# In[ ]:


# Assuming 'y' is a DataFrame with a single target column
column_name = y.columns[0]  # Dynamically get the name of the column

# Encode the target column if it's categorical
if y[column_name].dtype == 'object' or y[column_name].dtype.name == 'category':
    le_y = LabelEncoder()
    y[column_name] = le_y.fit_transform(y[column_name].astype(str))

print(y)


# In[ ]:


# Splitting the dataset and scaling features
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Codigo com linhas de codigo mais pequenas

# In[ ]:


# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32) if isinstance(X_train_scaled, pd.DataFrame) else torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded.values, dtype=torch.long) if isinstance(y_train_encoded, pd.DataFrame) else torch.tensor(y_train_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32) if isinstance(X_test_scaled, pd.DataFrame) else torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded.values, dtype=torch.long) if isinstance(y_test_encoded, pd.DataFrame) else torch.tensor(y_test_encoded, dtype=torch.long)


# In[ ]:


# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False) # Typically, you don't need to shuffle the test set


# In[ ]:


# Initializing a default Autoformer configuration
configuration = AutoformerConfig(
    # Add your model-specific configurations here
    prediction_length=10,  # Example: predict 10 time steps into the future
    context_length=30,  # Example: look at 30 time steps in the past
    # You might need to adjust other parameters depending on your dataset and requirements
    feature_size=47,  # Assuming you have 47 features
)
# Initializing a model from the configuration
model = AutoformerModel(configuration)

# Accessing the model configuration if needed
configuration = model.config


# In[ ]:


class AutoformerClassifier(nn.Module):
    def __init__(self, autoformer_model, num_features, num_classes):
        super(AutoformerClassifier, self).__init__()
        self.autoformer = autoformer_model  # The loaded Autoformer model
        self.classifier = nn.Linear(num_features, num_classes)  # Classification layer

    def forward(self, past_time_features, past_observed_mask):
        # Adjust the forward method to accept past_time_features and past_observed_mask
        outputs = self.autoformer(past_time_features=past_time_features, past_observed_mask=past_observed_mask)
        
        # Assuming we are interested in the last timestep's output for classification
        # You might need to adapt this depending on your specific use case and what the model outputs
        # For instance, outputs.last_hidden_state might not be directly applicable depending on how Autoformer's output is structured
        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        logits = self.classifier(last_hidden_state[:, -1, :])
        return logits


# In[ ]:


def train_and_evaluate(model, learning_rate, X_train, mask_train, y_train, X_val, epochs=3):
    model.train()  # Set the model to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        # Training loop
        for X_batch, mask_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(past_time_features=X_batch, past_observed_mask=mask_batch)
            loss = nn.CrossEntropyLoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        total, correct = 0, 0
        with torch.no_grad():  # No gradients needed for validation
            for X_batch, mask_batch, y_batch in val_loader:
                outputs = model(past_time_features=X_batch, past_observed_mask=mask_batch)
                _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        validation_accuracy = correct / total
        print(f'Epoch {epoch+1}, Validation Accuracy: {validation_accuracy:.4f}')
    
    return validation_accuracy


# In[ ]:


def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    # Assuming the datasets are tensors already, otherwise, convert them
    validation_accuracy = train_and_evaluate(model, learning_rate, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    
    return validation_accuracy


# In[ ]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Adjust n_trials as needed based on computational resources and needs


# In[ ]:


print(f'Best trial: {study.best_trial.params}')

