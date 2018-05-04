# S-Net
S-Net: A Scalable Convolutional Neural Network for JPEG Compression Artifact Reduction

## Network Architecture
![Error](https://github.com/zhenngbolun/S-Net/blob/master/network.jpg)

## Results

Model | Parameters| CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ 
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
DenseNet (L=40, k=12) |1.0M |7.00 |5.24 | 27.55|24.42
DenseNet (L=100, k=12)|7.0M |5.77 |4.10 | 23.79|20.20
DenseNet (L=100, k=24)|27.2M |5.83 |3.74 | 23.42|19.25
DenseNet-BC (L=100, k=12)|0.8M |5.92 |4.51 | 24.15|22.27
DenseNet-BC (L=250, k=24)|15.3M |**5.19** |3.62 | **19.64**|17.60
DenseNet-BC (L=190, k=40)|25.6M |- |**3.46** | -|**17.18**

LIVE1 Result

QF | L1 | L2 | L8 |            
-------|-------|--------|--------|
 40    | 34.41/0.9402  | 34.48/0.9410  | 34.61/0.9422  
 20    | 32.05/0.9034  | 32.13/0.9046  | 32.26/0.9067  
 10    | 29.67/0.8415  | 29.75/0.8435  | 29.87/0.8467  
<br>
BSDS500 Result <br>
QF     | L1            | L2            | L8            
:----- |:-------------:|:-------------:|:-------------:
 40    | 34.27/0.9394  | 34.33/0.9401  | 34.45/0.9413  
 20    | 31.97/0.9017  | 32.04/0.9028  | 32.15/0.9047  
 10    | 29.64/0.8391  | 29.71/0.8410  | 29.82/0.8440  
<br>
WIN143 Result <br>
QF     | L1            | L2            | L8            
:----- |:-------------:|:-------------:|:-------------:
 20    | 34.38/0.9220  | 34.47/0.9232  | 34.61/0.9250  
<br>
![Error](https://github.com/zhenngbolun/S-Net/blob/master/result1.jpg)
![Error](https://github.com/zhenngbolun/S-Net/blob/master/result2.jpg)

