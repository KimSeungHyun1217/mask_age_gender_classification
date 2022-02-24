def eval(model, data_loader, criterion, optimizer, scheduler, n_epoch=10):
    model.eval()
    since = time.time()
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        total = 0

        for imgs, labels in data_loader:
            imgs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizaer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            total += label.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = running_corrects.double() / total

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
    return model