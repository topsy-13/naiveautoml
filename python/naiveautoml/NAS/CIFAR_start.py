def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

import os 

# Load all the paths of the pickle files
files_path = os.listdir("CIFAR-10")
cifar_dict = {}
for file in files_path: 
  if file.startswith("data_batch"):
    cifar_dict.update(unpickle(file))

# Turn into numpy array 
