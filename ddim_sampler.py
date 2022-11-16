# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import DDIMPipeline, DDIMScheduler
from diffusers import UNet2DModel
import torch
import tqdm
import os
import time
import torchvision
from torchvision import transforms


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


def sample_ddim(repo_id, num_images, timesteps):
    """
    Function to sample from images using built-in pipeline using DDIM.
    
    Arguments:
        repo_id:      Name of repo for model
        timesteps:    Number of timesteps T
        num_images:   Number of images to sample
    """
    
    # instantiate model
    model = UNet2DModel.from_pretrained(repo_id)
    scheduler = DDIMScheduler.from_config(repo_id)
    scheduler.set_timesteps(num_inference_steps=timesteps)
    # transform = transforms.Compose([transforms.ToTensor()])

    sampled_imgs = torch.zeros(num_images, 3, 256, 256)
    all_times = []
    
    for j in range(num_images):
        start_time = time.time()

        # sample noisy image
        noisy_sample = torch.randn(
            1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
        sample = noisy_sample

        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
            # 1. predict noise residual
            with torch.no_grad():
                residual = model(sample, t).sample

              # compute previous image and set x_t -> x_t-1
            sample = scheduler.step(residual, t, sample).prev_sample
        end_time = time.time()
        all_times.append(end_time - start_time)
        
        # add sample
        sampled_imgs[j] = sample

    # record times
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
    #timesteps = np.linspace(10, 200, 20)
    timesteps = [10, 20, 30, 40, 50]
    fid_scores = np.zeros((20, 2))
    avg_times = np.zeros((20, 2))
    total_times = np.zeros((20, 2))
    
    repo_id_celeba_hq = "google/ddpm-celebahq-256"
    data_dir_celeba_hq = "celeba_hq_256"
    
    repo_id_lsun_bedroom = "google/ddpm-bedroom-256"
    data_dir_lsun_bedroom = "lsun_bedroom_256"
    
    num_images = 8
    it = 0
    
    # load in real data to compute fid scores
    real_data_celeba_hq = load_data(data_dir=data_dir_celeba_hq, num_images=num_images)
    real_data_bedroom = load_data(data_dir=data_dir_lsun_bedroom, num_images=num_images)
    
    for t in timesteps:
        t = int(t)
        
        # generate samples
        generated_data_celeb, times_celeb = sample_ddim(repo_id=repo_id_celeba_hq, num_images=num_images, timesteps=t)
        generated_data_bed, times_bed = sample_ddim(repo_id=repo_id_lsun_bedroom, num_images=num_images, timesteps=t)
        
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
    np.savez('ddim_timesteps.npz', timesteps)
    np.savez('ddim_total_times.npz', total_times)
    np.savez('ddim_avg_times.npz', avg_times)
    np.savez('ddim_fid_scores.npz', fid_scores)    


if __name__ == "__main__":
    main()


