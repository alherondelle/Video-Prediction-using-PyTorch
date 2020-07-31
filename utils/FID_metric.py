
###################FRÉCHET INCEPTION DISTANCE ####################
'''
This script implements the Fréchet Inception Distance inspired by the Tensorflow official implementation. It is adapted to the computation of 3 channels (non-RGB) images from the MSG satellite. The bucket links directly to the images generated by a Retrospective Cycle WGAN and their ground truth. 
''''
import torch
import torch.nn as nn
import torchvision.models as models
import scipy as sp
import numpy as np
from numpy import mean
from numpy import cov 
import linalg
import os

###### Importation of inception V3 for the FID computation. Withdral of the last Fully Connected layer #########

Inception = models.inception_v3(pretrained=True, progress=True, transform_input=False, aux_logits=False)
removed = list(Inception.children())[:-1]
InceptionV3 = nn.Sequential(*removed).cuda()

###### Definition of the Fréchet Distance 

'''
It is computed from the activation functions of Inception V3 on ground truth images and corresponding generated images. 
Covariance is computed on the mean of the output columuns for each feature.
'''

def calculate_fid(act1,act2):
  mu1, sigma1 = np.mean(act1, axis=(1,2)), cov(np.mean(act1, axis=(1)), rowvar=True)
  mu2, sigma2 = np.mean(act2, axis=(1,2)), cov(np.mean(act2, axis=(1)), rowvar=True)
  ssdiff = np.sum((mu1-mu2)**2.0)
  covmean = sp.linalg.sqrtm(sigma1.dot(sigma2))
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  fid = ssdiff + np.trace(sigma1 + sigma2 -2.0*covmean)
  return fid

###### Computation of the activation function from true and generated images
'''
act_t : activation function associated with the real image
act_f : activation function associated with the generated image
'''
def compute_act_inceptionv3(img_true, img_false):
    resized_A1 = nn.functional.interpolate(A1, size=(299,299), mode='bilinear', align_corners=False).cuda()
    resized_real_A = nn.functional.interpolate(real_A, size=(299,299), mode='bilinear', align_corners=False).cuda()
    act_t = InceptionV3(resized_real_A).squeeze(0)
    act_f = InceptionV3(resized_A1).squeeze(0)
    return act_t, act_f

if __name__ == '__main__':
    img_gen = torch.rand(1, 3, 299, 299).cuda()
    img_gt = torch.rand(1, 3, 299, 299).cuda()
    act_true, act_false = compute_act_inceptionv3(img_gt, img_gen)
    act_true = np.array(act_true.detach().cpu())
    act_false = np.array(act_false.detach().cpu())
    FID = calculate_fid(act_true, act_false)
    print ('Test on random images : ', FID)



