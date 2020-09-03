
import os
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import torch


def model_train(model, scheduler, optimizer, val_interval, start_epoch, num_epochs,  \
                train_dataloader, val_dataloader, criteria, out_dir, device):
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    best_val_loss = np.inf
    best_val_acc = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        epoch_loss = 0
        for img, label in train_dataloader:
            img, label = Variable(img).to(device), Variable(label).to(device)
            output = model(img)
            #print (output, tuple(output))
            if not output.shape:
                output = output[0]
            loss = criteria(output, label)
            #print (loss.data.cpu().numpy())
            epoch_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        torch.save(model.state_dict(),os.path.join(out_dir,'epoch_{}.pth'.format(epoch)))

        print ('Epoch - {0}, Training Loss - {1}'.format(epoch, epoch_loss/len(train_dataloader)))

        if epoch%val_interval == 0 :
            val_loss, val_acc = model_validation(model, val_dataloader, criteria, device)
            val_loss, val_acc = val_loss/len(val_dataloader), val_acc/len(val_dataloader)*100
            print ('Validation Loss - {}'.format(val_loss))
            print ('Validation Accuracy - {}'.format(val_acc))
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                torch.save(model.state_dict(),os.path.join(out_dir,'best_loss_chkpt.pth'))
                print ('Best validation loss - ',val_loss)
            if val_acc > best_val_acc :
                best_val_acc = val_acc
                torch.save(model.state_dict(),os.path.join(out_dir,'best_acc_chkpt.pth'))
                print ('Best validation accuracy - ',val_acc)                
        print ('#'*50)


def model_validation(model, val_dataloader, criteria, device):

    model.eval()

    epoch_loss = 0
    epoch_accuracy = 0

    class_correct = list(0. for i in range(11))
    class_total = list(0. for i in range(11))

    for img, target in val_dataloader:
        img, target = Variable(img).to(device), Variable(target).to(device)
        BS = img.shape[0]
        #print (bs)
        output = model(img)
        loss = criteria(output, target)
        _, pred = output.max(dim = -1)
        acc = torch.eq(pred,target).sum().data.cpu().numpy()

        epoch_accuracy += acc/BS
        epoch_loss += loss.data.cpu().numpy()

        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # # calculate test accuracy for each object class
        for i in range(BS):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    for i in range(10):
        print('Test Accuracy of class %1s: %2d%% (%2d/%2d)' % ( i, 100 * class_correct[i] / class_total[i],  \
            np.sum(class_correct[i]), np.sum(class_total[i])))


    return epoch_loss, epoch_accuracy