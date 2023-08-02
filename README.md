# ShrinkNet
Without any criteria to determine the importance of weights, we choose to generate random masks to block a propotion of channels at each layers of pretrained networks.

Original accuracy of ResNet56 on Cifar10 is 93.29%. We set different shrink ratio for three stages of ResNet56, namely, 0.75, 0.75, 0.5. Accuracy of pruned ResNet56 is 93.42%.

```python
python main.py -m shrink56
```
