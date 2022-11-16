#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 02:11:11 2022

@author: soominkwon

This script samples images from pre-trained models and computes the FID scores of the sampled images,
given that the dataset of the real images is available. The following command is useful if you are 
trying to sample images only:
    
pip install diffusers torch accelerate
"""

from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import DDIMPipeline, DDIMScheduler
from torchvision import transforms
import numpy as np
import time
import torchvision
import torch
import os


def load_data(data_dir, num_images):
    """
    Function to load in real data.
    """
    
    filenames = [name for name in os.listdir(data_dir)]
    filenames = filenames[:num_images]
    
    transform = transforms.Compose([transforms.CenterCrop(256)])
    
    batch = torch.zeros(num_images, 3, 256, 256, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        batch[i] = transform(torchvision.io.read_image(os.path.join(data_dir, filename)))
        
    return batch


def sample_ddpm(repo_id, num_images, timesteps):
    """
    Function to sample from images using built-in pipeline using DDPM.
    
    Arguments:
        repo_id:      Name of repo for model
        timesteps:    Number of timesteps T
        num_images:   Number of images to sample
    """
    
    sampled_imgs = torch.zeros(num_images, 3, 256, 256)
    image_pipe = DDIMPipeline.from_pretrained(repo_id)
    scheduler = DDIMScheduler.from_config(repo_id)
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("Scheduler:", scheduler.beta_schedule)
    
    all_times = []
    
    for i in range(num_images):
        start_time = time.time()
        image = image_pipe(num_inference_steps=timesteps).images[0]
        end_time = time.time()
        all_times.append(end_time - start_time)
        sampled_imgs[i] = transform(image)
        
    print("Total Sampling Time:", np.sum(all_times))
    print("Average Sampling Time:", np.mean(all_times))
    times = (np.sum(all_times), np.mean(all_times))
    return sampled_imgs, times


def compute_FID(real_data, generated_data):
    """
    Function to compute the FID score between generated data and real data.
    """
    # instantiating
    fid = FrechetInceptionDistance(feature=64)

    fid.update(real_data, real=True)
    fid.update(generated_data, real=False)
    
    return fid.compute()


def convert_sample(sample):
    image_processed = (sample + 1.0) * 127.5
    return image_processed.to(torch.uint8)


def main():
    # initializing variables
    timesteps = np.linspace(100, 2000, 20)
    fid_scores = np.zeros((20, 2))
    avg_times = np.zeros((20, 2))
    total_times = np.zeros((20, 2))
    
    repo_id_celeba_hq = "google/ddpm-celebahq-256"
    data_dir_celeba_hq = "celeba_hq_256"
    
    repo_id_lsun_bedroom = "google/ddpm-bedroom-256"
    data_dir_lsun_bedroom = "lsun_bedroom_256"
    
    num_images = 16
    it = 0
    
    # load in real data to compute fid scores
    real_data_celeba_hq = load_data(data_dir=data_dir_celeba_hq, num_images=num_images)
    real_data_bedroom = load_data(data_dir=data_dir_lsun_bedroom, num_images=num_images)
    
    for t in timesteps:
        t = int(t)
        
        # generate samples
        generated_data_celeb, times_celeb = sample_ddpm(repo_id=repo_id_celeba_hq, num_images=num_images, timesteps=t)
        generated_data_bed, times_bed = sample_ddpm(repo_id=repo_id_lsun_bedroom, num_images=num_images, timesteps=t)
        
        # compute fid scores
        fid_score_celeb = compute_FID(real_data=real_data_celeba_hq, generated_data=convert_sample(generated_data_celeb))
        fid_score_bedroom = compute_FID(real_data=real_data_bedroom, generated_data=convert_sample(generated_data_bed))

        fid_scores[it, 0] = fid_score_celeb
        fid_scores[it, 1] = fid_score_bedroom 
        
        total_times[it, 0] = times_celeb[0]
        total_times[it, 1] = times_bed[0]

        avg_times[it, 0] = times_celeb[1]
        avg_times[it, 1] = times_bed[1]
        
        it += 1
        
    # saving results
    np.savez('ddpm_timesteps.npz', timesteps)
    np.savez('ddpm_total_times.npz', total_times)
    np.savez('ddpm_avg_times.npz', avg_times)
    np.savez('ddpm_fid_scores.npz', fid_scores)  
        

if __name__ == "__main__":
    main()