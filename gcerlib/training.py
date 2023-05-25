import sys
import torch
from torch import optim
import segmentation_models_pytorch as smp
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.metrics import *
from segmentation_models_pytorch.losses import *
from statistics import fmean
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import re
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from gcerlib.data.load_data import ImageDictLoader

# determine the appropriate trainer to use
def get_trainer(config, dataset, model):
    schema = config.training.schema.lower()
    
    if schema in ['fully supervised', 'fs', 'fully', 'supervised', 'f']:
        return FullySupervisedTrainer(config, dataset, model)

class BaseTrainer(object):
    def __init__(self):
        super().__init__()
        pass         
        # will flesh out later
    
class FullySupervisedTrainer(BaseTrainer):
    def __init__(self, config, dataset, model):
        super().__init__()
        
        self.config = config
        self.device = config.device
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        print(f'[TRAIN] Using device: {self.device}')
        self.dataset = dataset
        self.model = model.double()
        self.model_name = f'{config.experiment_name}_{config.model.model_arch}_{config.model.model_encoder}'
        
        self.num_epochs = config.training.epochs
        self.batch_size = config.training.batch_size
        self.accumulate_grads = config.training.accumulate_grads
        self.save_best_model = config.training.save_best_model
        self.save_every_model = config.training.save_every_model
        self.num_classes = config.model.classes
        self.is_binary = self.num_classes == 1
        
        self.out_path = config.out_path
        self.model_path = os.path.join(config.out_path, 'models')
        if not os.path.exists(self.model_path): os.mkdir(self.model_path)
        self.test_images_path = os.path.join(config.out_path, 'test_images')
        if not os.path.exists(self.test_images_path): os.mkdir(self.test_images_path)
        self.plot_paths = os.path.join(config.out_path, 'plots')
        if not os.path.exists(self.plot_paths): os.mkdir(self.plot_paths)
        self.error_viz_path = os.path.join(config.out_path, 'error_viz')
        if not os.path.exists(self.error_viz_path): os.mkdir(self.error_viz_path)
        self.save_all_test_data = config.evaluation.save_all_test_data
        
        # configure optimizer
        optimizer_name = config.training.optimizer.lower()
        if optimizer_name in ['adam', 'a'] or config.training.optimizer is None:
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                lr=config.training.lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
        else:
            raise NotImplementedError(f'[TRAIN] Optimizer {config.training.optimizer} not implemented.')
        
        # configure loss
        loss_name = config.training.loss.lower()
        if loss_name in ['crossentropy', 'ce', 'cross', 'entropy', 'c', 'bce']:
            self.loss = torch.nn.CrossEntropyLoss() if config.model.classes > 1 else torch.nn.BCELoss()
        elif loss_name in ['focal', 'focalloss', 'focal loss']:
            self.loss = FocalLoss(
                mode='multiclass' if self.num_classes > 1 else 'binary',
            )
        elif loss_name in ['jaccard', 'j', 'jac', 'jaccard loss']:
            self.loss = JaccardLoss(
                mode='multiclass' if self.num_classes > 1 else 'binary',
            )
        else:
            raise NotImplementedError(f'[TRAIN] Loss {config.training.loss} not implemented.')
        
        self.metric_fns = [f1_score, iou_score, accuracy, precision, recall]
        self.history = self.init_history()
        self.epoch = 1
        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_delta = config.training.early_stopping_min_delta
        self.metrics_reduction = config.evaluation.metrics_reduction
        self.grad_accumulation = config.training.accumulate_grads
        
        self.config.training.return_best_model = config.training.return_best_model
        self.config.evaluation.use_best_model = config.evaluation.use_best_model
        self.config.evaluation.use_model_path = config.evaluation.use_model_path
        
        self.train_data = ImageDictLoader(dataset['train'], config)
        self.val_data = ImageDictLoader(dataset['val'], config)
        self.test_data = ImageDictLoader(dataset['test'], config)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    def train_step(self):
        self.model.train()
                
        epoch_loss = 0.
        self.init_batch_metrics()
        
        # simple progress bar implementation
        # https://adamoudad.github.io/posts/progress_bar_with_tqdm/
        with tqdm(self.train_loader, unit='batch') as tepoch:
            for i, (X, y_true) in enumerate(tepoch):
                tepoch.set_description(f'Epoch {self.epoch}/{self.num_epochs} Training')

                # clear grad
                self.model.zero_grad(set_to_none=True)
                
                
                # y_true : Ground Truth
                X = X.to(self.device).double()
                y_true = y_true.to(self.device)

                # y_pred : Segmentation Result
                y_pred = self.model(X)
                if self.config.model.classes == 1:
                    y_pred = F.sigmoid(y_pred.squeeze()).double()

                if self.loss.__class__.__name__ == 'BCELoss':
                    loss = self.loss(y_pred.double(), y_true.double())
                elif self.loss.__class__.__name__ == 'CrossEntropyLoss':
                    loss = self.loss(y_pred, y_true.long())
                elif self.loss.__class__.__name__ == 'JaccardLoss':
                    loss = self.loss(y_pred, y_true.long())
                else:
                    loss = self.loss(y_pred, y_true)
                epoch_loss += loss.item()

                # Backprop + optimize
                loss.backward()
                self.optimizer.step()
                
                self.update_batch_metrics(y_pred, y_true)
                tepoch.set_postfix(loss=loss.item())
                
                # if i > 2: break
        
        epoch_loss /= len(self.train_loader)
        self.update_history(epoch_loss, phase='train')
        self.print_batch_metrics()

        return epoch_loss

    @torch.no_grad()
    def val_step(self):
        self.model.eval()
        
        self.init_batch_metrics()
        epoch_val_loss = 0.

        # simple progress bar implementation
        # https://adamoudad.github.io/posts/progress_bar_with_tqdm/
        with tqdm(self.val_loader, unit='batch') as tepoch:
            for i, (X, y_true) in enumerate(tepoch):
                tepoch.set_description(f'Epoch {self.epoch}/{self.num_epochs} Validation')

                # y_true : Ground Truth
                X = X.to(self.device).double()
                y_true = y_true.to(self.device)

                # y_pred : Segmentation Result
                y_pred = self.model(X)
                if self.config.model.classes == 1:
                    y_pred = F.sigmoid(y_pred.squeeze())

                if self.loss.__class__.__name__ == 'BCELoss':
                    loss = self.loss(y_pred.double(), y_true.double())
                elif self.loss.__class__.__name__ == 'CrossEntropyLoss':
                    loss = self.loss(y_pred, y_true.long())        
                elif self.loss.__class__.__name__ == 'JaccardLoss':
                    loss = self.loss(y_pred, y_true.long())        
                else:
                    loss = self.loss(y_pred, y_true)
                epoch_val_loss += loss.item()
                
                self.update_batch_metrics(y_pred, y_true)

                tepoch.set_postfix(val_loss=loss.item())
                
                # if i > 2: break
        epoch_val_loss /= len(self.val_loader)
        self.update_history(epoch_val_loss, phase='val')
        self.print_batch_metrics()

        return epoch_val_loss

    # TODO : implement test_step
    @torch.no_grad()
    def test_step(self):
        
        print('[TRAIN] Running inference on test images')
        
        error_cmap = ListedColormap(['black', 'red'])
        # TODO REMOVE THIS
        # https://gist.github.com/jgomezdans/402500?permalink_comment_id=2264839#gistcomment-2264839
        # seg_cmap = ListedColormap(['darkgreen', 'lawngreen', 'saddlebrown', 'dodgerblue', 'yellow', 'lightgray', 'dimgray', 'purple'])
        vals = np.linspace(0,1,256)
        np.random.shuffle(vals)
        seg_cmap = ListedColormap(plt.cm.jet(vals))
        
        self.model.eval()

        self.init_batch_metrics()
        
        with tqdm(self.test_loader, unit='batch') as tepoch:
            for i, (X, y_true) in enumerate(tepoch):
            
                X = X.to(self.device)
                y_true = y_true.to(self.device)
                y_pred = F.sigmoid(self.model(X))
                
                if self.is_binary:
                    self.update_batch_metrics(y_pred.squeeze(), y_true.squeeze())
                else:
                    self.update_batch_metrics(y_pred, y_true)
                
                y_pred = y_pred.argmax(dim=1)
                
                raw_image = self.test_data.inverse_norm(X.cpu())
                
                if self.save_all_test_data:
                    # visualize errors
                    for j in range(len(X)):                        
                        if len(X) == 1:
                            image_id = f'{i:04d}'
                        else:
                            image_id = f'{i:04d}_{j:04d}'
                        
                        # save images, masks, and predictions
                        image = X[j].cpu().numpy().transpose(2, 1, 0)
                        image = (image * 255).astype(np.uint8)
                        
                        raw_image = raw_image[j].cpu().numpy().transpose(2, 1, 0)
                        raw_image = (raw_image * 255).astype(np.uint8)
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        # print(image.shape, image.min(), image.max())
                        cv2.imwrite(os.path.join(self.test_images_path, f'{image_id}_input.png'), image)
                        
                        
                        target = y_true[j].cpu().numpy().astype(np.uint8)
                        # print(target.shape, target.min(), target.max())
                        cv2.imwrite(os.path.join(self.test_images_path, f'{image_id}_target.png'), target)
                        
                        pred = y_pred[j].cpu().numpy().astype(np.uint8)
                        # print(pred.shape, pred.min(), pred.max())
                        cv2.imwrite(os.path.join(self.test_images_path, f'{image_id}_pred.png'), target)
                        
                        error = target != pred
                        cv2.imwrite(os.path.join(self.test_images_path, f'{image_id}_error.png'), target)
                        
                        # save error visualizations
                        fig, ax = plt.subplots(2, 3)
                        ax = ax.ravel()
                        ax[0].imshow(raw_image, interpolation='none')
                        ax[0].set_xticks([])
                        ax[0].set_yticks([])
                        ax[0].set(xlabel='Optical Data')
                        ax[1].imshow(image, interpolation='none')
                        ax[1].set_xticks([])
                        ax[1].set_yticks([])
                        ax[1].set(xlabel='Normalized Data')
                        ax[2].imshow(target, cmap=seg_cmap, interpolation='none')
                        ax[2].set(xlabel='Ground Truth')
                        ax[2].set_xticks([])
                        ax[2].set_yticks([])
                        ax[3].imshow(pred, cmap=seg_cmap, interpolation='none')
                        ax[3].set(xlabel='Predicted Mask')
                        ax[3].set_xticks([])
                        ax[3].set_yticks([])
                        ax[4].imshow(error, cmap=error_cmap, interpolation='none')
                        ax[4].set(xlabel='Error')
                        ax[4].set_xticks([])
                        ax[4].set_yticks([])
                        ax[5].imshow(image, interpolation='none')
                        ax[5].imshow(error, cmap=error_cmap, alpha=error.astype(float), interpolation='none')
                        ax[5].set(xlabel='Error Overlay')
                        ax[5].set_xticks([])
                        ax[5].set_yticks([])
                        
                        fig.tight_layout()
                        plt.style.use('ggplot')
                        plt.suptitle(f'{self.model_name}\n{image_id}')
                        plt.savefig(os.path.join(self.error_viz_path, f'{image_id}.png'), dpi=300)
                        plt.close()

                tepoch.set_postfix(mIoU = fmean(self.batch_metrics['iou_score']))
        
        metrics = {}
        self.print_batch_metrics()
        for metric, value in self.batch_metrics.items():
            metrics[metric] = fmean(value)
        metrics_series = pd.Series(metrics)
        metrics_series.to_csv(os.path.join(self.out_path,'metrics.csv'))
        
        return metrics
        
    def init_history(self, reduction='micro'):
        phases = ['train', 'val']
        history = {}
        history['epoch'] = []
        for phase in phases:
            history[f'{phase}_loss'] = []
            for metric_fn in self.metric_fns:
                history[f'{phase}_{metric_fn.__name__}'] = []
        return history
            
    def update_history(self, loss, phase):
        if phase == 'train':
            self.history['epoch'].append(self.epoch) # dirty hack
        for metric_fn in self.metric_fns:
            self.history[f'{phase}_{metric_fn.__name__}'].append(fmean(self.batch_metrics[metric_fn.__name__]))
        self.history[f'{phase}_loss'].append(loss)
    
    def init_batch_metrics(self):
        batch_metrics = {}
        for metric_fn in self.metric_fns:
            batch_metrics[metric_fn.__name__] = []
        self.batch_metrics = batch_metrics
    
    def update_batch_metrics(self, y_pred, y_true):
        if self.num_classes == 1:
            if y_pred.max() > 1 or y_pred.min() < 0:
                y_pred = F.sigmoid(y_pred).squeeze()
        tp, fp, fn, tn = get_stats(
            output=y_pred.argmax(dim=1) if self.num_classes > 1 else y_pred, # might not work with binary segmentation
            target=y_true,
            mode='multiclass' if self.num_classes > 1 else 'binary',
            num_classes=self.num_classes,
            threshold=None if self.num_classes > 1 else 0.5,
        )
        for metric_fn in self.metric_fns:
            value = metric_fn(tp, fp, fn, tn, reduction=self.metrics_reduction)
            self.batch_metrics[metric_fn.__name__].append(float(value))
    
    def print_batch_metrics(self):
        for metric, value in self.batch_metrics.items():
            print(f'{metric}: {fmean(value):.4f}', end=' | ')
        print()
        
    def save_history(self):
        # print(self.history)
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.out_path,'history.csv'), index=False)
    
    def load_history(self):
        history_path = os.path.join(self.out_path,'history.csv')
        if os.path.exists(history_path):
            history = pd.read_csv(os.path.join(self.out_path,'history.csv'))
            self.history = history.to_dict(orient='list')
            return True
        else:
            print('[TRAIN] No previous history found. Training from scratch...')
            return False
    
    def plot_curves(self):
        pass
    
    def train(self):
        try:
            use_tb = self.config.training.tensorboard
            
            if use_tb: writer = SummaryWriter(log_dir=f'runs/{self.config.experiment_name}')
            
            self.epoch = 0
            if self.config.training.load_from_checkpoint and self.load_history():
                # quick way to find best epoch model
                models = list(Path(self.model_path).glob('best_model*.pth'))
                last_epoch = 0
                for model in models:
                    epoch = int(re.findall(r'(?<=best_model_epoch_)\d+', str(model))[0])
                    if epoch > last_epoch: 
                        last_model = model
                        last_epoch = epoch
                if last_epoch == 0:
                    print('[TRAIN] No previous models found. Please check config file (training.load_from_checkpoint).')
                    print('[TRAIN] Training from scratch...')
                else:
                    self.model.load_state_dict(torch.load(last_model))
                    self.epoch = last_epoch
                    print(f'[TRAIN] Loaded last model at epoch {last_epoch}')
            
            self.model.to(self.device)
            
            best_loss, best_epoch = np.inf, 0
            early_stopper = EarlyStopper(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)
            
            while self.epoch < self.num_epochs:
                self.epoch += 1
                _ = self.train_step()
                val_loss = self.val_step()
                if self.save_every_model:
                    # save model
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, f'epoch_{self.epoch}_val_loss_{val_loss}.pth'))
                if self.save_best_model and val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = self.epoch
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, f'best_model_epoch_{best_epoch}_val_loss_{val_loss}.pth'))
                if early_stopper.early_stop(val_loss):
                    print(f'[TRAIN] Stopping training at epoch {self.epoch} with val_loss {val_loss:.4f} and best_epoch {best_epoch} with best_val_loss {best_loss:.4f}')
                    break
                if use_tb:
                    writer.add_scalar('Loss/train', self.history['train_loss'][-1], self.epoch)
                    writer.add_scalar('Loss/val', self.history['val_loss'][-1], self.epoch)
                    for metric_fn in self.metric_fns:
                        writer.add_scalar(f'{metric_fn.__name__}/train', self.history[f'train_{metric_fn.__name__}'][-1], self.epoch)
                        writer.add_scalar(f'{metric_fn.__name__}/val', self.history[f'val_{metric_fn.__name__}'][-1], self.epoch)
            
                self.save_history()
            self.plot_history()
            if self.config.training.return_best_model:
                self.model.load_state_dict(torch.load(os.path.join(self.model_path, f'best_model_epoch_{best_epoch}_val_loss_{best_loss}.pth')))
            return self.model

        except KeyboardInterrupt:
            print(f'[TRAIN] Stopping training at epoch {self.epoch} with val_loss {val_loss:.4f} and best_epoch {best_epoch} with best_val_loss {best_loss:.4f}')
            self.save_history()
            exit(0)
        
        except UnboundLocalError as e:
            print(f'[TRAIN] Error: {e}', file=sys.stderr)
            print(f'You are most likely trying to load from a previous model that does not exist. Please check your config file.', file=sys.stderr)
            exit(-1)
            
        except Exception as e:
            print(f'[TRAIN] Error: {e}', file=sys.stderr)
            exit(-1)

    def plot_history(self):
        for metric_fn in self.metric_fns:
            for phase in ['train', 'val']:
                plt.plot(self.history['epoch'], self.history[f'{phase}_{metric_fn.__name__}'], label=phase)
            val_max = max(self.history[f'val_{metric_fn.__name__}'])
            arg_val_max = self.history[f'val_{metric_fn.__name__}'].index(val_max)+1
            plt.plot(arg_val_max, val_max, 'ro', label=f'{val_max:.4f} at epoch {arg_val_max}')
            plt.axvline(x=arg_val_max, color='r', linestyle='--')
            plt.legend()
            plt.suptitle(f'{self.model_name}\n{metric_fn.__name__}')
            plt.savefig(os.path.join(self.plot_paths, f'{metric_fn.__name__}.png'))
            plt.close()
            
        plt.plot(self.history['epoch'], self.history['train_loss'], label='train')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='val')
        val_min = min(self.history[f'val_loss'])
        arg_val_min = self.history[f'val_loss'].index(val_min)+1
        plt.plot(arg_val_min, val_min, 'ro', label=f'{val_min:.4f} at epoch {arg_val_min}')
        plt.axvline(x=arg_val_min, color='r', linestyle='--')
        plt.legend()
        plt.suptitle(f'{self.model_name}\nloss')
        plt.savefig(os.path.join(self.plot_paths, 'loss.png'))
        plt.close()
        
    def evaluate(self):
        
        if self.config.evaluation.use_best_model:
            # quconfig.evaluation.useway to find best epoch model
            best_models = list(Path(self.model_path).glob('best_model*.pth'))
            best_epoch = 0
            for model in best_models:
                epoch = int(re.findall(r'(?<=best_model_epoch_)\d+', str(model))[0])
                if epoch > best_epoch: 
                    best_model = model
                    best_epoch = epoch
            self.model.load_state_dict(torch.load(best_model))
            print('[EVALUATE] Loaded best model:', best_model)

        elif self.eval_model_path:
            self.model.load_state_dict(torch.load(self.eval_model_path))
            print('[EVALUATE] Loaded model:', self.eval_model_path)

        if self.config.evaluation.load_previous_log:
            self.load_history()
            print('[EVALUATE] Loaded previous history:', os.path.join(self.out_path,'history.csv'))
        
        
        print('[EVALUATE] Plotting history...')
        self.plot_history()
        self.model.to(self.device)
            
        print('[EVALUATE] Evaluating...')
        self.test_step()
        
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



