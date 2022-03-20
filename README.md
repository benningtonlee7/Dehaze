# Dehaze
### Dependencies and Installation:

- Python 3.7
- PyTorch>=1.0
- NVIDIA GPU and CUDA

### Directory file structure
```
 -Dehaze 
   - train/ (directory copy the unzip files downloaded from the competition website
     - clean_images 
     - clean_images_labels 
     ....
   - Res2Net.py
   - eval.py
   - dataloader.py
   - train.py
   - model.py
   - KTDN.pth (Download here: https://drive.google.com/file/d/1fJeNJq14ij2TFPuTc8S0esfHdzwYYvYe/view?usp=sharing)
   
```

### Evaluation

```
 # If GPU is available 
 CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --cuda --model-dir KTDN.pth 
 # Else
 python eval.py --model-dir KTDN.pth 
```

After running the above command, there should be a folder named "results", and inside the folder there should be the dehazed output images.