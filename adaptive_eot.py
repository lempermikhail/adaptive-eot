import sys, os, random
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from core import Smooth  # official implementation
import torchattacks



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
cifar_loader = torch.utils.data.DataLoader(cifar_test, batch_size=1, shuffle=True)
# For this porject, I got CIFAR-10 ResNet20 model from https://github.com/chenyaofo/pytorch-cifar-models
model = torch.hub.load('chenyaofo/pytorch-cifar-models','cifar10_resnet20',pretrained=True,trust_repo=True).to(device).eval()



num_classes = 10
sigma = 0.25
smoothed_clf = Smooth(model, num_classes, sigma)

#transformations
def transform_rotate(image):
    angle = random.uniform(-15, 15)
    return TF.rotate(image, angle)

def transform_translate(image):
    dx, dy = random.uniform(-10, 10), random.uniform(-10, 10)
    return TF.affine(image, angle=0, translate=(dx, dy), scale=1.0, shear=0)

def transform_brightness(image):
    factor = random.uniform(0.7, 1.3)
    return TF.adjust_brightness(image, factor)

def transform_contrast(image):
    factor = random.uniform(0.7, 1.3)
    return TF.adjust_contrast(image, factor)

transforms_list = [("rotate", transform_rotate),("translate", transform_translate),("brightness", transform_brightness),("contrast", transform_contrast),]


def adaptive_eot_attack(x_orig, y_target,base_model,num_steps=100, epsilon=0.2, lr=0.05,sigma=0.25, num_samples=100):
    x_adv = x_orig.clone().detach().to(device).requires_grad_(True) #copy of the input image
    optimizer = torch.optim.Adam([x_adv], lr=lr)

    for step in range(num_steps):
        grads = torch.zeros(len(transforms_list),device=device)  #stores how strong the gradients are for each transformation
        for i, (_, tf) in enumerate(transforms_list):
            acc = 0.0
            for _ in range(num_samples):
                img_t = tf(x_adv.squeeze(0).cpu()).unsqueeze(0).to(device) #applying current transformation to the image
                noise = torch.randn_like(img_t) * sigma #Gaussian noise to simulate randomized smoothing
                out = base_model(torch.clamp(img_t + noise, 0, 1))
                loss = F.cross_entropy(out, y_target)
                loss.backward(retain_graph=True)
                acc += x_adv.grad.abs().mean().item()
                x_adv.grad.zero_()
            grads[i] = acc/num_samples#store gradients
        idx = int(torch.argmax(grads))#best transformation
        _,chosen_tf = transforms_list[idx]

        optimizer.zero_grad() #update using transformation
        img_t = chosen_tf(x_adv.squeeze(0).cpu()).unsqueeze(0).to(device)# applying the chosen transformation,noise
        noise = torch.randn_like(img_t) * sigma # Gaussian noise to simulate randomized smoothing
        out = base_model(torch.clamp(img_t + noise, 0, 1))
        loss = F.cross_entropy(out, y_target)
        loss.backward()
        optimizer.step()

        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon) #perturbation has to be within epsilon range
        x_adv.data = torch.clamp(x_orig + delta, 0, 1)

        if step%10==0:
            print("Step " + str(step) + ", Loss: "+ str(round(loss.item(), 4)) + ", Transform: " + transforms_list[idx][0])

    return x_adv.detach()


def eval_base(loader, attack_fn, name, num_images=25):
    success=0
    for i, (x,y) in enumerate(loader):
        if i >= num_images:
            break
        x, y = x.to(device),y.to(device)
        y_t = (y + 1)%10
        x_adv = attack_fn(x,torch.tensor([y_t]).to(device), model)
        with torch.no_grad():
            pred =int(model(x_adv).argmax(dim=1).item())
        if pred == int(y_t):
            success += 1
        print(i)
    print(name + " on Smoothed: " + str(success) +" out of " + str(num_images) + " successful (" + str(round(100 * success / num_images, 2)) + "%)")



def eval_smoothed(loader, attack_fn, name, num_images=25):
    success=0
    for i, (x,y) in enumerate(loader):
        if i >= num_images:
            break
        x, y = x.to(device), y.to(device)
        y_t = (y + 1)%10
        x_adv = attack_fn(x, torch.tensor([y_t]).to(device), model)
        pred = smoothed_clf.predict(x_adv, 100, 0.001, 32)
        if pred == int(y_t):
            success +=1

    print(name + " on Smoothed: " + str(success) +" out of " + str(num_images) + " successful (" +str(round(100 *success/num_images, 2)) + "%)")


def eval_both(loader, attack, name, num_images=100):
    base_success = 0
    smooth_success = 0
    for i, (x, y) in enumerate(loader):
        if i >= num_images:
            break
        x, y = x.to(device), y.to(device)
        y_t = (y + 1) % 10
        x_adv = attack(x, y_t)

        with torch.no_grad():
            pred_base = model(x_adv).argmax(dim=1).item()
        if pred_base == y_t.item():
            base_success += 1

        pred_smooth = smoothed_clf.predict(x_adv, n=100, alpha=0.001, batch_size=32)
        if pred_smooth == y_t.item(): #target; != y.item() - untargeted
            smooth_success += 1

    print(name + " - base: " + str(base_success)+ "/" + str(num_images)+" (" + str(round(100 * base_success / num_images, 2)) + "%), "+"smoothed: " + str(smooth_success) + "/" + str(num_images) + " (" + str(round(100 * smooth_success / num_images, 2)) + "%)")


print("Vanilla PGD")
pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=100,random_start=True)
pgd_attack.set_device(device)
pgd_attack.set_mode_targeted_random()

eval_both(cifar_loader, pgd_attack, "PGD")

print("PGD-EOT")
eot_attack = torchattacks.EOTPGD(model,eps=8/255,alpha=2/255,steps=100,eot_iter=100,random_start=True)
eot_attack.set_device(device)
eot_attack.set_mode_targeted_random()
eval_both(cifar_loader, eot_attack, "PGD-EOT")


print("Adaptive EOT")
eval_base(cifar_loader,adaptive_eot_attack, "Adaptive EOT")
eval_smoothed(cifar_loader, adaptive_eot_attack,"Adaptive EOT")