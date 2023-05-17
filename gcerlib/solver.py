import torch
from torch import optim
import segmentation_models_pytorch as smp
import os
import tqdm
from torch.utils.data import DataLoader, Dataset
import csv

# determine the appropriate trainer to use
def get_trainer(config):
    schema = config.training.schema.lower()
    
    if schema in ['fully supervised', 'fs', 'fully', 'supervised', 'f']:
        return FullySupervisedTrainer(config)

# for loading dicts of numpy images
class ImageDictLoader(Dataset):
    def __init__(self, dataset, config, mode='train'):
        super.__init__(self)
        
        self.dataset = dataset
    
    def __getitem__(self, index):
        
        # NO PREPROCESSING FOR NOW
        # TODO add preprocessing
        input = self.dataset['input'][index]
        target = self.dataset['target'][index]
        
        return 

class BaseTrainer(object):
    def __init__(self, config):
        pass         
        # will flesh out later
    
class FullySupervisedTrainer(BaseTrainer):
    def __init__(self, config, dataset, model):
        super.__init__(config, dataset, model)
        
        self.device = config.device
        self.dataset = dataset
        self.model = model
        
        self.epochs = config.training.epochs
        self.batch_size = config.training.batch_size
        self.accumulate_grads = config.training.accumulate_grads
        self.save_best_model = config.training.save_best_model
        self.save_every_model = config.training.save_every_model
        self.num_classes = config.model.classes
        
        self.path = os.path.join(config.out_path, 'models')
        
        # configure optimizer
        optimizer_name = config.training.optimizer.lower()
        if optimizer_name in ['adam', 'a'] or config.training.optimizer is None:
            self.optimizer = optim.Adam(
                lr=config.training.lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
        else:
            raise NotImplementedError(f'[TRAIN] Optimizer {config.training.optimizer} not implemented.')
        
        # configure loss
        loss_name = config.training.loss.lower()
        if loss_name in ['crossentropy', 'ce', 'cross', 'entropy', 'c']:
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss_name in ['focal', 'focalloss', 'focal loss']:
            self.loss = smp.utils.losses.FocalLoss()
        else:
            raise NotImplementedError(f'[TRAIN] Loss {config.training.loss} not implemented.')
        
        self.train_loader = DataLoader(dataset=dataset['train'], batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=dataset['val'], batch_size=self.batch_size, shuffle=True)
            

    def train_step(self):
        
        self.model.eval(False)
        self.model.train(True)
        
        epoch_loss = 0.
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity
        PC = 0. 	# Precision
        SP = 0.     # Specificity
        F1 = 0.		# F1 Score
        F2 = 0.		# F2 Score
        JC = 0.		# Jaccard Score/IoU
        length = 0

        # simple progress bar implementation
        # https://adamoudad.github.io/posts/progress_bar_with_tqdm/
        with tqdm(self.train_loader, unit='batch') as tepoch:
            for i, (images, GT) in enumerate(tepoch):
                tepoch.set_description(f'Epoch {self.epoch+1} Training')

                # GT : Ground Truth
                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.model(images)

                loss = self.loss(SR,GT)
                epoch_loss += loss.item()

                # Backprop + optimize
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # calculate metrics
                tp, fp, fn, tn = smp.metrics.get_stats(
                    output=SR,
                    target=GT,
                    mode='multiclass' if self.num_classes > 1 else 'binary'
                )
                # temp_acc, temp_dice, temp_prec, temp_reca = self.segmentation_metrics(GT, SR)
                acc += smp.metrics.accuracy(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                SE += smp.metrics.sensitivity(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                PC += smp.metrics.precision(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                JC += smp.metrics.iou_score(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                SP += smp.metrics.specificity(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                F1 += smp.metrics.f1_score(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                F2 += smp.metrics.fbeta_score(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                length += 1

                tepoch.set_postfix(loss=loss.item())

        acc = acc/length
        SE = SE/length
        PC = PC/length
        JC = JC/length
        SP = SP/length
        F1 = F1/length
        F2 = F2/length
        # DC = DC/length

        # Print the log info
        print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
            self.epoch+1, self.num_epochs, \
            epoch_loss,\
            acc,SE,PC,F1,JC))

        # log everything
        self.history['Epoch'].append(self.epoch+1)
        logger(self.history, epoch_loss, acc, SE, PC, F1, JC, phase='Training')

        return epoch_loss, JC

    @torch.no_grad()
    def val_step(self):
        self.model.train(False)
        self.model.eval(True)
        
        epoch_val_loss = 0.
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity
        PC = 0. 	# Precision
        SP = 0.     # Specificity
        F1 = 0.		# F1 Score
        F2 = 0.		# F2 Score
        JC = 0.		# Jaccard Score/IoU
        length = 0

        # simple progress bar implementation
        # https://adamoudad.github.io/posts/progress_bar_with_tqdm/
        with tqdm(self.val_loader, unit='batch') as tepoch:
            for i, (images, GT) in enumerate(tepoch):
                tepoch.set_description(f'Epoch {self.epoch+1} Validation')

                # GT : Ground Truth
                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.model(images)

                loss = self.loss(SR,GT)
                epoch_val_loss += loss.item()

                # Backprop + optimize
                # self.model.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                # calculate metrics
                tp, fp, fn, tn = smp.metrics.get_stats(
                    output=SR,
                    target=GT,
                    mode='multiclass' if self.num_classes > 1 else 'binary'
                )
                # temp_acc, temp_dice, temp_prec, temp_reca = self.segmentation_metrics(GT, SR)
                acc += smp.metrics.accuracy(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                SE += smp.metrics.sensitivity(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                PC += smp.metrics.precision(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                JC += smp.metrics.iou_score(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                SP += smp.metrics.specificity(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                F1 += smp.metrics.f1_score(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                F2 += smp.metrics.fbeta_score(tp, fp, fn, tn, mode='multiclass' if self.num_classes > 1 else 'binary')
                length += 1

                tepoch.set_postfix(loss=loss.item())

        acc = acc/length
        SE = SE/length
        PC = PC/length
        JC = JC/length
        SP = SP/length
        F1 = F1/length
        F2 = F2/length
        # DC = DC/length

        # Print the log info
        print('Epoch [%d/%d], Loss: %.4f, \n[Validation] Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f, DC: %.4f' % (
            self.epoch+1, self.num_epochs, \
            epoch_val_loss,\
            acc,SE,PC,F1,JC))

        # log everything
        self.history['Epoch'].append(self.epoch+1)
        logger(self.history, epoch_val_loss, acc, SE, PC, F1, JC, phase='Validation')

        return epoch_val_loss, JC

    # TODO : implement test_step
    @torch.no_grad()
    def test_step(self):
        self.unet.train(False)
        self.unet.eval()

        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        DC = 0.		# Dice Coefficient
        length=0
        for i, (images, GT) in enumerate(self.valid_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))
            temp_acc, temp_dice, temp_prec, temp_reca = self.segmentation_metrics(GT, SR)

            acc += temp_acc
            SE += temp_reca
            PC += temp_prec
            DC += temp_dice
            length += 1
                
        acc = acc/length
        SE = SE/length
        PC = PC/length
        F1 = 2 * (PC * SE) / (PC + SE)
        DC = DC/length


        f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([
            "Model Type",
            "Accuracy",
            "Sensitivity",
            "Precision",
            "F1",
            "Dice",
            "Learning Rate",
            "Best Epoch",
            "Num epochs",
            "Num epochs decay",
            "Augmentation prob"
        ])
        wr.writerow([self.model_type,acc,SE,PC,F1,DC,self.lr,self.best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
        f.close()

def logger(history_dict, loss, acc, SE, PC, F1, DC, phase='Training'):
    history_dict[phase + ' Loss'].append(loss)
    history_dict[phase + ' Accuracy'].append(acc)
    history_dict[phase + ' Sensitivity'].append(SE)
    history_dict[phase + ' Precision'].append(PC)
    history_dict[phase + ' F1'].append(F1)
    history_dict[phase + ' IoU'].append(DC)


