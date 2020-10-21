import torchvision.models as v_models
import torch.nn as nn
import torch
from datasets.entropy import EntropyLoader
from utils.basic_utils import print_cuda_statistics
from torch.utils.tensorboard import SummaryWriter
import logging
import os

class NaiveResNetAgent:
    def __init__(self, config):
        self.logger = logging.getLogger("Agent")
        self.config = config
        
        self.renset18 = v_models.resnet18(pretrained=True)
        num_inftrs = self.renset18.fc.in_features
        self.renset18.fc = nn.Linear(num_inftrs, self.config.num_classes)

        self.data_loader = EntropyLoader(self.config)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.renset18.parameters(),
                                          lr=self.config.learning_rate,
                                          betas=self.config.betas,
                                          weight_decay=self.config.weight_decay)
        

        self.current_epoch = 1
        self.best_valid_acc = 0
        self.lowest_val_loss = 99999
        self.best_model = self.renset18.state_dict()
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.manual_seed(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            #print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")

        self.renset18 = self.renset18.to(self.device)
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.modality)
        
    def run(self):
        try:
            self.train()
            self.test()

            self.summary_writer.flush()
            self.summary_writer.close()
            self.logger.info('Model with best validation of {:.3f} saved...'.format(self.best_valid_acc))

            torch.save(self.best_model, os.path.join(self.config.exp_root, self.config.exp_name + '.pth'))
        except KeyboardInterrupt:
            self.logger.info("You have exited waiting for clean up...")

    def train(self):
        for epoch in range(self.config.max_epoch):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
                    
    def train_one_epoch(self):
         
        self.renset18.train()
        epoch_loss = 0.0
        corrects = 0
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.renset18(data)
            loss = self.loss(output, target)
            _, preds = torch.max(output, 1)  # get the index of the max log-probability
            corrects += torch.sum(preds == target)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.3f}'.format(
                    self.current_epoch, 
                    batch_idx * len(data), 
                    len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader), 
                    loss.item()
                    ))
            
        acc = corrects.item() / len(self.data_loader.train_loader.dataset)
        epoch_loss /= len(self.data_loader.train_loader.dataset)
        self.summary_writer.add_scalar('Loss/train', epoch_loss, self.current_epoch)
        self.summary_writer.add_scalar('Acc/train', acc, self.current_epoch)

    def validate(self):

        self.renset18.eval()
        val_loss = 0.0
        corrects = 0

        with torch.no_grad():
            for data, target in self.data_loader.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.renset18(data)
                val_loss += self.loss(output, target).item()  # sum up batch loss
                _, preds = torch.max(output, 1)  # get the index of the max log-probability
                corrects += torch.sum(preds == target)

        val_acc = corrects.item() / len(self.data_loader.val_loader.dataset)
        val_loss /= len(self.data_loader.val_loader.dataset)

        if val_loss < self.lowest_val_loss:
            self.lowest_val_loss = val_loss
            self.best_model = self.renset18.state_dict()

        
        self.logger.info('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, 
            corrects, 
            len(self.data_loader.val_loader.dataset),
            100. * val_acc
            ))  

        self.summary_writer.add_scalar('Loss/val', val_loss, self.current_epoch)
        self.summary_writer.add_scalar('Acc/val', val_acc, self.current_epoch)
        
    def test(self):
        
        self.renset18.load_state_dict(self.best_model)
        self.renset18.eval()

        test_loss = 0.0
        corrects = 0

        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.renset18(data)
                test_loss += self.loss(output, target).item()  # sum up batch loss
                _, preds = torch.max(output, 1)  # get the index of the max log-probability
                corrects += torch.sum(preds == target)

        test_acc = corrects.item() / len(self.data_loader.test_loader.dataset)
        test_loss /= len(self.data_loader.test_loader.dataset)

        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, 
            corrects, 
            len(self.data_loader.test_loader.dataset),
            100. * test_acc
            ))

        self.summary_writer.add_scalar('Loss/test', test_loss, self.current_epoch)
        self.summary_writer.add_scalar('Acc/test', test_acc, self.current_epoch)  