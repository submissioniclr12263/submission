# Note: this code is strongly influenced by https://colab.research.google.com/github/st-howard/blog-notebooks/blob/main/MNIST-Diffusion/Diffusion%20Digits%20-%20Generating%20MNIST%20Digits%20from%20noise%20with%20HuggingFace%20Diffusers.ipynb

import torch, torchvision, datasets, diffusers, accelerate
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import PIL
import numpy as np
import random
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 32
    train_batch_size = 48
    eval_batch_size = 48
    num_epochs = 9
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmpup_steps = 500
    mixed_precision = 'fp16'
    seed = 0
    
config = TrainingConfig()

mnist_dataset = datasets.load_dataset('mnist', split='train')
mnist_dataset.reset_format()

def transform(dataset):
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (config.image_size, config.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 2*(x-0.5)),
        ]
    )
    images = [preprocess(image) for image in dataset["image"]]
    return {"images": images}

indices_zero = [i for i, datum in enumerate(mnist_dataset) if datum['label'] == 0]
indices_one  = [i for i, datum in enumerate(mnist_dataset) if datum['label'] == 1]
indices_one  = random.sample(indices_one, int(len(indices_zero) * 0.25))

indices = indices_zero + indices_one
mnist_dataset.set_transform(transform)
subset = Subset(mnist_dataset, indices)

train_dataloader = torch.utils.data.DataLoader(
    subset,
    batch_size = config.train_batch_size,
    shuffle = True,
)

model = diffusers.UNet2DModel(
    sample_size=config.image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128,128,256,512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)

lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmpup_steps,
    num_training_steps=(len(train_dataloader)*config.num_epochs),
)
device = torch.device('cuda:0')
model.to(device)
# If loading a model:
#model = diffusers.UNet2DModel.from_pretrained('data/model1').to(device)

M, m  = 20, .1
tau   = lambda t: np.exp(-(t * m + t**2 * (M-m)/2)/2)
a     = lambda t: (1-tau(t)**2)**(1/2)
ap    = lambda t: -tau(t)/a(t) * bp(t)
b     = lambda t: tau(t)
bp    = lambda t: -(m+(M-m)*t)/2 * tau(t)

def train_loop(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler):

    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            batch_size = clean_images.shape[0]
            if batch_size == config.train_batch_size:
                ts   = torch.rand((batch_size,), device=clean_images.device)
                # Change here for training with different grids. For instance, to
                # have half of the points in (.3, .5) you should define ts as follows
                #ts1  = torch.rand((3*batch_size//16,), device=clean_images.device) * .3
                #ts2  = torch.rand((batch_size//2,), device=clean_images.device) * .2 + .3
                #ts3  = torch.rand((5*batch_size//16,), device=clean_images.device) * .5 + .5
                #ts   = torch.cat((ts1, ts2, ts3))
                
                a_ts = torch.tensor(a(ts.cpu().numpy()), device=clean_images.device)
                b_ts = torch.tensor(b(ts.cpu().numpy()), device=clean_images.device)
                noisy_images = a_ts[:,None,None,None] * noise + b_ts[:,None,None,None] * clean_images
                
                with accelerator.accumulate(model):
                    noise_pred = model(noisy_images, ts*1000)["sample"]
                    loss = torch.nn.functional.mse_loss(noise_pred,noise)
                    accelerator.backward(loss)
                    
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                if step % 100 == 99:
                    progress_bar.update(100)
                    logs = {
                        "loss" : loss.detach().item(),
                        "lr" : lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
    
    accelerator.unwrap_model(model)


# Code to train
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
accelerate.notebook_launcher(train_loop, args, num_processes=1)

# Code to generate
with torch.no_grad():
    ts = np.linspace(1, 0, n+1)
    x  = torch.randn((20,1,32,32)).to(model.device)
    
    for i, t in enumerate(ts[:-1]):
        noiser = model(x, t*n)['sample']
        bf = lambda x, t: bp(t)/b(t)*x + 2*(ap(t) - bp(t)/b(t)*a(t))*noiser
        z  = torch.randn_like(x)
        dt = -(ts[i+1] - ts[i])
        x = x - bf(x, t) * dt + (-2*bp(t)/b(t) * dt) ** (1/2) * z

        # Visualization code
        '''
        if i % (n//10) == 0:
            print(t)
            fig, axes = plt.subplots(nrows=1, ncols=x.shape[0], figsize=(x.shape[0], 1))
            for j in range(x.shape[0]):
                axes[j].imshow(torchvision.transforms.ToPILImage()(x.clamp(-1, 1)[j].squeeze(0)), cmap='gray')
                axes[j].axis('off')
            plt.show()
        '''