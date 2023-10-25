# ShrinkNet
Without any criteria to determine the importance scores of weights or activation values, we choose to generate random masks to block a propotion of channels at each layer of pretrained networks.

Original accuracy of ResNet56 on Cifar10 is 93.29%. We set different shrink ratio for three stages of ResNet56, namely, 0.75, 0.75, 0.5. Accuracy of pruned ResNet56 is 93.42%,93.22% and 93.30% with different random seeds.

# Running
```python
python main.py -m shrink56
```

# To do
- [ ] cut weights to accelerate model inference.
- [ ] Add function of FLOPs and Params calculation.

