import datetime
import six
import json
import os
import torch
from google.oauth2 import service_account
from google.cloud import storage
import matplotlib.pyplot as plt
from google.colab import files
import glob
import random
import os
import torchvision.transforms as transforms
import torch
import random
import pickle
from torch.utils.data import Dataset
from google.cloud import storage
from PIL import Image

#Importation à partir d'un Google Bucket

def import_bucket():
  os.system('export GOOGLE_APPLICATION_CREDENTIALS="cle.json"')
      

    # Explicitly use service account credentials by specifying the private key
    # file.
  storage_client = storage.Client.from_service_account_json('cle.json')

      # Make an authenticated API request
  buckets = list(storage_client.list_buckets())
  blobs = list(storage_client.list_blobs(buckets[2]))
  for blob in blobs :
    blob.download_to_filename('data/METEOSAT/train/'+blob.name)

def unzip():
  os.system('mv data/METEOSAT/train/2_1.zip data/METEOSAT/test')
  os.system('mv data/METEOSAT/train/2_2.zip data/METEOSAT/test')
  for folder in os.listdir('data/METEOSAT'):
    for fichier in os.listdir(os.path.join('data/METEOSAT', folder)):
      cmd1 = 'unzip data/METEOSAT/'+folder+'/'+fichier+' -d data/METEOSAT/'+folder+'/'
      cmd2 = 'rm -rf data/METEOSAT/'+folder+'/'+fichier
      cmd3 = 'rm -rf data/METEOSAT/'+folder+'/__MACOSX'
      os.system(cmd1)
      os.system(cmd2)
      os.system(cmd3)
  os.system('rm -rf data/METEOSAT/train/3_6/3_7')




  




