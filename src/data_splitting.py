#import pandas as pd
import numpy as np

def train_val_split(data):
    """
    divide dataset en 80% train y 20% validation

    """
    shuffled= data.sample(frac=1,  random_state= 42).reset_index(drop=True)

    cut_idx= int(0.8* len(data)) #split

    train= shuffled.iloc[:cut_idx].copy() #80%
    validation= shuffled.iloc[cut_idx:].copy()#20%
    print(f'Train set shape: {train.shape}, \nValidation set shape: {validation.shape}')

    return train, validation