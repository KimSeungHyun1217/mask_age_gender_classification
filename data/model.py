import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import reduce
import torch

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class PretrainedModel:
    def __init__(self, model_name,num_class):
        self.model_name = model_name
        self.num_class = num_class
        model = getattr(models,model_name)(pretrained=True)
        
        for name, mod in reversed(list(model.named_modules())):
            if isinstance(mod, nn.Linear):
                mod_path = name.split('.')
                classifier_parent = reduce(nn.Module.get_submodule, mod_path[:-1], model)
                setattr(classifier_parent, mod_path[-1], nn.Sequential(
                    nn.Linear(mod.in_features, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.7),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.num_class)
                ))
                break

        self.model = model
        
    def __call__(self):
        return self.model
        

class Ensemble(nn.Module):
    def __init__(self,modelA,modelB,modelC):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        
        self.multi_fc = nn.Linear(3,3)
        
    def forward(self,x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)
        
        # anslist = []
        
        # for i,y,z in zip(out1,out2,out3):
        #     if int(torch.argmax(i)) == 2:
        #         anslist.append(i.tolist())
        #     elif int(torch.argmax(y)) == 2:
        #         anslist.append(y.tolist())
        #     elif int(torch.argmax(z)) == 2:
        #         anslist.append(z.tolist())
        #     else:
        #         anslist.append(i.tolist())
        
        # final = torch.tensor(anslist,requires_grad=True)
        #final = final.to(device)
        out= out1+out2+out3
        final = self.multi_fc(out)
        
        return final
        