#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 00:30:52 2022

@author: soominkwon
"""

import torch
from unet import Unet
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

class DDPM:
    def __init__(self, scheduler, timesteps, image_size, image_channels):
        # initializing parameters
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.image_size = image_size
        self.image_channels = image_channels
        
        # computing parameters for forward process
        self.betas = self.beta_schedule()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # computing parameters for reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        
    def beta_schedule(self, beta_start=0.0001, beta_end=0.02):
        """
        Function to compute the beta schedule. Default is linear, but quadratic and sigmoid are available.
        """
            
        print("Note: Using " + self.scheduler + " scheduling for beta." + "\n")
        
        if self.scheduler == "quadratic":
            return torch.linspace(beta_start**0.5, beta_end**0.5, self.timesteps) ** 2
        elif self.scheduler == "sigmoid":
            betas = torch.linspace(-6, 6, self.timesteps)
            return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            return torch.linspace(beta_start, beta_end, self.timesteps)
            

    def extract(self, a, t, x_shape):
        """
        Helper function to retrieve certain value of noise (alpha) from indexed t.
        
        Arguments:
            a:         PyTorch tensor of alpha values.
            t:         PyTorch tensor of the time index.
            x_shape:   Shape of the image 'x'
            
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    
    def q_sample(self, x_start, t, noise=None):
        """
        Function to obtain q(x_t | x_0) following the setup above.
        
        Arguments:
            x_start:         Starting image x_0
            t:               Time sample 't'
            noise:           Noise (default is Gaussian)
        """
    
        # base case
        if noise is None:
            noise = torch.randn_like(x_start) # z (it does not depend on t!)
    
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
    
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Function to sample p(x_{t-1} | x_t)
        """
    
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
    
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # x_{t-1}
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Function to sample x_0 from x_T, iteratively.
        """
        
        device = next(model.parameters()).device
    
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
    
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((shape[0],), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs


    def unet_loss(self, denoise_model, x_start, t):
        """
        Defining the loss function for the U-Net model.
        
        Arguments:
            denoise_model:      U-Net model to predict noise given noisy image
            x_start:            Starting clean image x_0
            t:                  Timestep t
        
        """
        
        # random sample z    
        noise = torch.randn_like(x_start)
        
        # compute x_t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # recover z from x_t
        predicted_noise = denoise_model(x_noisy, t)

        return F.smooth_l1_loss(noise, predicted_noise)      
     
        
    def train(self, dataset, batch_size, learning_rate, epochs):
        """
        Function to train U-Net model. The U-Net model must be trained before sampling.
        
        Arguments:
            dataset:        Training dataset
            batch_size:     Batch size for training 
            learning_rate:  Learning rate for model
            epochs:         Number of training iterations
        
        """
        
        # using GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # making data into a PyTorch dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # instantiating model
        model = Unet(dim=self.image_size,
                     channels=self.image_channels,
                     dim_mults=(1, 2, 4,))
        model.to(device)
            
        # choosing optimizer
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # training loop
        for epoch in range(epochs):
            print("Training Epoch:", epoch)
            
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(device)
            
                # sample t from U(0,T)
                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
            
                loss = self.unet_loss(model, batch, t)
            
                if step % 100 == 0:
                    print("Loss Value:", loss.item())
            
                loss.backward()
                optimizer.step()
                
        return model
    
    
    @torch.no_grad()
    def sample(self, model, num_images, image_size, channels):
        """
        Function to sample images given a trained U-Net model.
        
        Arguments:
            model:          Trained U-Net model to estimate means
            num_images:     Number of images to sample
            image_size:     Dimensions of image (should be square)
            channels:       Number of channels for images (e.g. 3 for RGB)
        """
        return self.p_sample_loop(model, shape=(num_images, channels, image_size, image_size))
