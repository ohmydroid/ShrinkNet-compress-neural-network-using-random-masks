import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, idx, stride=1,expansion=1):
        super(BasicBlock, self).__init__()

        self.block_idx = idx
        self.shrink_state = False 

        self.planes = planes
        self.expansion = expansion

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(in_planes, expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(expansion*planes),
                                    )

       
        self.conv2 = nn.Sequential(
                                    nn.Conv2d(expansion*planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(planes),
                                    )
        if self.shrink_state:

           self.inner_mask = torch.bernoulli(torch.empty(expansion*planes).fill_(self.shrink_ratio)).view(1,-1,1,1)
           self.out_mask   = torch.bernoulli(torch.empty(planes).fill_(self.shrink_ratio)).view(1,-1,1,1)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
                                        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(planes)
                                        )

    def shrinkblock(self, shrink_ratio=0.75, mask_mode='fixed', freeze=False):
        #print('shrink_ratio of {}th block is {}'.format(block_idx, self.shrink_ratio))

        if freeze:
           for module in self.modules():
               if isinstance(module, nn.Conv2d):
                  for param in module.parameters():
                      param.requires_grad = False

        if mask_mode=='bernoulli': 
           self.inner_mask = torch.bernoulli(torch.empty(self.expansion*self.planes).fill_(shrink_ratio)).view(1,-1,1,1)
           self.out_mask   = torch.bernoulli(torch.empty(self.planes).fill_(shrink_ratio)).view(1,-1,1,1)
           #self.res_mask = torch.bernoulli(torch.tensor(0.5))
           print((self.inner_mask==1.0).sum().item())
           print((self.out_mask==1.0).sum().item())
           #print(self.res_mask)
        
        # generate fixed masks
        else:
           self.inner_mask = torch.zeros(1,self.expansion*self.planes,1,1)
           self.inner_mask[0,:int(shrink_ratio*self.expansion*self.planes), 0, 0]=1.0 

           #self.res_mask = torch.bernoulli(torch.tensor(0.5))
           
           self.out_mask = torch.zeros(1,self.planes,1,1)
           self.out_mask[0,:int(shrink_ratio*self.planes), 0, 0]=1.0 

        self.shrink_state=True

    def forward(self, x):
        out = self.conv1(x) 

        if self.shrink_state:
           out = out*self.inner_mask.to(out.device)
           #print('after inner_mask_{}:{}'.format(self.block_idx,self.inner_mask[0,-1,0,0]))
           
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        if self.shrink_state:
           out = out*self.out_mask.to(out.device)

        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out



 
class ShrinkNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_scaler=1, expansion=1):
        super(ShrinkNet, self).__init__()
        self.idx = 0 
        self.in_planes = 16*width_scaler

        self.conv1 = nn.Sequential(nn.Conv2d(3,self.in_planes, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(self.in_planes),
                                   nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(block, 16*width_scaler, num_blocks[0], stride=1, expansion=expansion)
        self.layer2 = self._make_layer(block, 32*width_scaler, num_blocks[1], stride=2, expansion=expansion)
        self.layer3 = self._make_layer(block, 64*width_scaler, num_blocks[2], stride=2, expansion=expansion)

        self.linear = nn.Linear(64*width_scaler, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride, expansion):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.idx+=1
            layers.append(block(self.in_planes, planes, self.idx, stride, expansion))
            self.in_planes = planes 

        return nn.Sequential(*layers)


    def shrinknet(self, shrink_ratio, freeze=False, mask_mode='bernoulli'):
        
        index=0 
        for name, module in self.named_modules():
            if hasattr(module, 'shrinkblock'):
                module.shrinkblock(shrink_ratio=shrink_ratio[index], freeze=freeze, mask_mode=mask_mode)
                index += 1     


       
    def forward(self, x):
        
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def shrinknet20(num_classes,expansion,width_scaler=1):
    return ShrinkNet(BasicBlock, [3, 3, 3],num_classes, expansion=expansion, width_scaler=width_scaler)


def shrinknet56(num_classes,expansion,width_scaler=1):
    return ShrinkNet(BasicBlock, [9, 9, 9],num_classes, expansion=expansion, width_scaler=width_scaler)


