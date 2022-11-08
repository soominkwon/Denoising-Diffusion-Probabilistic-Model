#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:40:42 2022

@author: soominkwon
"""

from ddpm import DDPM
from torchvision import transforms
from datasets import load_dataset
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

def main():
    # loading mnist dataset
    dataset = load_dataset("mnist")
    
    # define image transformations (e.g. using torchvision)
    transform = Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    
    # define function
    def transformations(examples):
       examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
       del examples["image"]
    
       return examples
    
    transformed_dataset = dataset.with_transform(transformations).remove_columns("label")
    
    final_dataset = transformed_dataset["train"]
    
    # initializing variables
    scheduler = "linear"
    timesteps = 500
    image_size = 28
    image_channels = 1
    
    batch_size = 128
    learning_rate = 1e-3
    epochs = 10
    
    num_images = 50
    
    ddpm = DDPM(scheduler, timesteps, image_size, image_channels)
    trained_unet = ddpm.train(dataset=final_dataset, batch_size=batch_size,
                              learning_rate=learning_rate, epochs=epochs)
    sampled_images = ddpm.sample(model=trained_unet, num_images=num_images,
                                 image_size=image_size, channels=image_channels)
    
    plt.imshow(sampled_images[-1][15].reshape(image_size, image_size), cmap="gray")
    plt.show
    

if __name__ == "__main__":
    main()

    