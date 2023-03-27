import torch
from tqdm import tqdm_gui,tqdm
from diffusers import DPMSolverMultistepScheduler,get_cosine_schedule_with_warmup,DPMSolverSinglestepScheduler,DDIMScheduler
from utils import ESC
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet_Conditional
from diffusers.utils import floats_tensor
import matplotlib.pyplot as plt
import datetime
import numpy as np
from timm.models.xception import xception
choose=1
def forward(model,scheduler,image_size=224,batch_size=16,sample_class=1,device='cuda',channels=3):
    global choose
    sample=torch.randn(batch_size,channels,image_size,image_size).to(device)
    for i,t in enumerate(tqdm(scheduler.timesteps)):
        #print(t.shape)
        with torch.no_grad():
            #print(sample.shape,t*torch.ones(batch_size).long(),sample_class*torch.ones(batch_size))
            if(choose==0):
                residual=model(sample,t=t*torch.ones(batch_size).long().to(device),y=sample_class*torch.ones(batch_size).long().to(device))
            else:
                residual=model(sample,time=t*torch.ones(batch_size).long().to(device),label=sample_class*torch.ones(batch_size).long().to(device))
        sample=scheduler.step(residual,t,sample).prev_sample
    return sample

device='cuda'
judge=True
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import os
k_ranges=[2,10]#the hyperparameter k

for k in k_ranges:
    os.makedirs("Synthetic_us8k_"+str(k),exist_ok=True)
    urbansound=["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"]
    #model=Unet_Conditional
    import random
    name="./Diffusion_us8k_dim64/Diffusion_us8k_dim64_90augmentation_130.pt"#720'
    #name="./Diffusion_us8k_dim64/Diffusion_us8k_dim64_1275.pt"
    model=torch.load(name).to(device=device)
    model.eval()
    noise_scheduler=DPMSolverSinglestepScheduler()
    print("Name: ",name)
    #noise_scheduler.load_config("noise_scheduler.pt/scheduler_config.json")
    noise_scheduler.set_timesteps(num_inference_steps=20)
    random.seed(random.seed(random.seed(1)))
    judge_net=None
    if(judge):
        judge_net=torch.load("judge.pt").to(device=device)
        judge_net.eval()
    print("judge:",judge)
    cnt=0
    from tqdm import tqdm
    for epoch in tqdm(range(1510)):
        if(cnt>8750):break
        temp=random.randint(0,9)
        print("Class:",temp)
        image=forward(model,noise_scheduler,128,batch_size=10,sample_class=temp,device=device,channels=3)
    # plt.figure(figsize=(10,10))
        print(image.max(),image.min())
        if(judge==False):
            image=image.permute(0,2,3,1).clip(0,1).cpu().detach().numpy()
        else:
            output=judge_net(image)

        for i in range(10):
            #print(image.shape)
            #plt.subplot(2,5,i+1)
            #image[i]=(image[i]-image[i].min())/(image[i].max()-image[i].min())
            #plt.imshow(image[i])
        # print(image.max(),image.min())
        #plt.show()
            # plt.subplot(4,5,10+(i+1))
            #plt.imshow(image[i].permute(1,2,0).clip(0,1).cpu().detach().numpy()[::-1])
            if(judge):
                
                
                a, idx1 = torch.sort(output[i], descending=True)#descending为alse，升序，为True，降序
                idx = idx1[:k]
                print("Cnt: ",cnt,name,idx[:k])
                #print(output.argmax().cpu().detach().numpy(),temp)
                if(temp in idx.cpu().detach().numpy()):
                    cnt+=1
                    image_to_save=image.permute(0,2,3,1).clip(0,1).cpu().detach().numpy()[i]
                    plt.imsave("Synthetic_us8k_"+str(k)+"/"+"0-"+str(temp)+"-"+str(i)+"-"+str(epoch)+".jpg",image_to_save)
            else:
                plt.imsave("Synthetic_us8k/"+"0-"+str(temp)+"-"+str(i)+"-"+str(epoch)+".jpg",image[i])
