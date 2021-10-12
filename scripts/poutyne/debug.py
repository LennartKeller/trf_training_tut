#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from torch.optim import AdamW
from poutyne import Model
from poutyne_transformers import ModelWrapper, MetricWrapper, TransformerCollator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from poutyne_modules import (
    so_data_collator,
    make_tokenization_func,
    make_rename_func,
    PoutyneSequenceOrderingLoss,
    make_compute_metrics_func,
)


# In[2]:


MODEL_NAME_OR_PATH = "bert-base-cased"
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
DEVICE = "cuda:0"
N_EPOCHS = 3


# In[3]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
transformer = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME_OR_PATH, return_dict=True, num_labels=1
)


# In[4]:


dataset = load_from_disk("../data/rocstories/")
# Downsampling for debugging...
dataset = dataset.filter(lambda _, index: index < 300, with_indices=True)


# In[5]:


tokenization_func = make_tokenization_func(
    tokenizer=tokenizer,
    text_column="text",
    add_special_tokens=False,
    padding="max_length",
    truncation=True,
)
dataset = dataset.map(tokenization_func, batched=True)


# In[6]:


rename_target_column = make_rename_func({"so_targets": "labels"}, remove_src=True)
dataset = dataset.map(rename_target_column, batched=True)


# In[7]:


dataset


# In[8]:


dataset = dataset.remove_columns(
    ["text", "storyid", "storytitle"] + [f"sentence{i}" for i in range(1, 6)]
)
dataset.set_format("torch")


# In[9]:


collate_fn = TransformerCollator(
    y_keys=["labels", "input_ids"], custom_collator=so_data_collator, remove_labels=True
)

train_dataloader = DataLoader(
    dataset["train"], batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    dataset["val"], batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn
)
test_dataloader = DataLoader(
    dataset["test"], batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn
)


# In[10]:


wrapped_transformer = ModelWrapper(transformer)

optimizer = AdamW(wrapped_transformer.parameters(), lr=LEARNING_RATE)
loss_fn = PoutyneSequenceOrderingLoss(target_token_id=tokenizer.cls_token_id)

metric_func = MetricWrapper(make_compute_metrics_func(tokenizer.cls_token_id))

model = Model(
    wrapped_transformer, optimizer, loss_fn, batch_metrics=[metric_func], device=DEVICE
)


# In[ ]:


model.fit_generator(train_dataloader, val_dataloader, epochs=N_EPOCHS)


# In[ ]:


test_loss = model.evaluate_generator(test_dataloader)


# In[ ]:


test_loss


# In[ ]:
