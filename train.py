import time
import copy
import torch

def train_or_eval(mode, model, data_loader, criterion, optimizer, scheduler, n_epoch=10, device='cpu'):
    if mode == 'train':
        model.train()  # 모델을 학습 모드로 설정
    else:
        model.eval()   # 모델을 평가 모드로 설정

    since = time.time()
    for epoch in range(1, n_epoch + 1):
        print('Epoch {}/{}'.format(epoch, n_epoch))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        total = 0

        for imgs, labels in data_loader:
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(mode == 'train'):
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        if(mode == 'train'):
            scheduler.step()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = running_corrects.double() / total

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model




    
