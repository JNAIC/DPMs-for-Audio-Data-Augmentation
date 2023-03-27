#Training
from diffusers import DPMSolverMultistepScheduler,get_cosine_schedule_with_warmup,DPMSolverSinglestepScheduler
import torch.functional as F
import random
import torch.nn as nn
from tqdm import tqdm_notebook
from tqdm import tqdm
from utils import Unet_Conditional,US8K
from torchvision.transforms import RandAugment
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils import US8K
dataset1=US8K(transform_size=128,train=True,root="Preprocessing_us8k")
dataset2=US8K(transform_size=128,train=True,root="Preprocessing_us8k_augmentation")
urbansound8k=["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"]

noise_scheduler=DPMSolverMultistepScheduler(num_train_timesteps=1000)
noise_scheduler.set_timesteps(num_inference_steps=20)
augmen=RandAugment()
loss=nn.MSELoss()
device='cuda'
model = Unet_Conditional(labels_dim=10,dim=64).to(device)
model=torch.load('./Diffusion_us8k_dim64/Diffusion_us8k_dim64_90augmentation_175.pt')
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def forward(model,scheduler,config,batch_size=16,sample_class=1,device='cuda'):
    sample=torch.randn(batch_size,3,config.image_size,config.image_size).to(device)
    for i,t in enumerate(tqdm(scheduler.timesteps)):
        #print(t.shape)
        with torch.no_grad():
            residual=model(sample,t=t*torch.ones(batch_size).long().to(device),label=sample_class*torch.ones(batch_size).long().to(device))
        sample=scheduler.step(residual,t,sample).prev_sample
    return sample


epoches=3500

for epoch in tqdm_notebook(range(0,epoches)):
    if(random.randint(0,20)==0):
        dataloader=DataLoader(dataset1, batch_size=30, shuffle=True)
    else:
        dataloader=DataLoader(dataset2, batch_size=30, shuffle=True)
    
    for data,label in tqdm_notebook(dataloader):
        #print(label.shape)
        
        data=data.to(device)
        # data=255*data
        # data=torch.tensor(data,dtype=torch.uint8)
        # data=augmen(data)
        # data=data.float()
        # data=data/255

        label=torch.argmax(label,dim=1).to(device).long()
        optimizer.zero_grad()


        noise=torch.randn_like(data)
        timesteps=torch.randint(0,noise_scheduler.num_train_timesteps,(data.shape[0],)).to(device)
        noisy_image=noise_scheduler.add_noise(data,noise,timesteps)


        noise_pred=model(noisy_image,time=timesteps,label=label.long())
        loss_val=loss(noise_pred,noise)

        loss_val.backward()
        optimizer.step()
       
    if(epoch%5==0):
        print("Epoch: ",epoch,"Loss: ",loss_val.item())
        torch.save(model,f'./Diffusion_us8k_dim64/Diffusion_us8k_dim64_90augmentation_{epoch}.pt')    
