#Load the best two resnets for each modality
#Apply call back functions to the specific layers I want to extract
#Collect the feature maps into containers for each modality
#Apply the gated fusion
#Pass it as input to a fully connected classifier
import torchvision.models as v_models
import torch.nn.functional as F
import torch.nn as nn
import torch
from datasets.entropy import EntropyLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

class MgafAgent:
    def __init__(self, config):
        self.logger = logging.getLogger("Agent")
        self.config = config

        
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

        self.mri_interm_feats = {}
        self.pet_interm_feats = {}

        self.attach_points = ['layer2.1.conv2','layer3.1.conv2','layer4.1.conv2']
        #self.attach_points = ['layer4.1.conv2'] 
        
        
        self.mri_model = self._init_resnet(self.config.mri_weights, mode='MRI').to(self.device)
        self.pet_model = self._init_resnet(self.config.pet_weights, mode='PET').to(self.device)

        self._add_hooks(self.mri_model, self.mri_interm_feats)
        self._add_hooks(self.pet_model, self.pet_interm_feats)

        self.classifier = nn.Sequential(
            nn.Linear(175616, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 4, bias=True)
        ).to(self.device)

        self.data_loader = EntropyLoader(self.config, split=True)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(),
                                          lr=self.config.learning_rate,
                                          betas=self.config.betas,
                                          weight_decay=self.config.weight_decay)
        self.current_epoch = 1

        self.clf = SGDClassifier()
        self.kmeans = MiniBatchKMeans(n_clusters=self.config.num_classes, random_state=42, batch_size=self.config.batch_size)
        self.ipca = IncrementalPCA(n_components=self.config.num_classes, batch_size=self.config.batch_size)

        self.best_val_acc = 0.0
        self.best_model = {}
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.modality)

    def _init_resnet(self, weights_path, mode=None):
        assert type(mode) == str

        model = v_models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        model.load_state_dict(torch.load(weights_path))
        self.logger.info(f'Initialized the {mode} model with best weights')
        
        model.mode = mode
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        return model

    def get_kernel(self, channel_size):
        kernel = torch.tensor([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]], dtype=torch.float32, device=self.device).detach()
        return kernel.view(1,1,3,3).repeat(1,channel_size,1,1)

    def _add_hooks(self, model, container):
        for name, module in model.named_modules():
            if name in self.attach_points:
                self.logger.info(f'{model.mode}: Added hook {name} in {module}')
                module.register_forward_hook(self._hook_accum_wrapper(name, container))

    def _hook_accum_wrapper(self, name, container):
        def hook_fn(module, input, output):
            container[name] = output.data.detach()
        return hook_fn

    def gated_fusion(self, features):
        kernel = self.get_kernel(features.shape[1])
        filtered = F.conv2d(features, kernel, padding=1)
        filtered = torch.sigmoid(filtered)
        return features * filtered #This is element wise mul of two tensors
    
    def multimodal_gated_fusion(self):
        #print('fusion')
        temp = [] 
        for mri_fts, pet_fts in zip(self.mri_interm_feats.values(), self.pet_interm_feats.values()):
            output = self.gated_fusion(mri_fts) + self.gated_fusion(pet_fts) 
            output = torch.flatten(output, start_dim=1)
            print(f'MRI:{mri_fts.shape} fused with PET:{pet_fts.shape} yield --> {output.shape}')
            temp.append(output)
        temp = torch.cat(temp, dim=1)
        #print(f'Final output {temp.shape}')
        return temp

    def singlemode_gate_fusion(self, fts_vals):
        temp = []
        for fts in fts_vals:
            output = self.gated_fusion(fts)
            output = torch.flatten(output, start_dim=1)
            print(f'{fts.shape} -> {output.shape}')
            temp.append(output)
        temp = torch.cat(temp, dim=1)
        return temp

    def simple_concat(self):
        temp = [] 
        for mri_fts, pet_fts in zip(self.mri_interm_feats.values(), self.pet_interm_feats.values()):
            mri_fts = torch.flatten(mri_fts, start_dim=1)
            pet_fts = torch.flatten(pet_fts, start_dim=1)
            output = torch.cat((mri_fts, pet_fts), dim=1)
            print(f'MRI:{mri_fts.shape} fused with PET:{pet_fts.shape} yield --> {output.shape}')
            temp.append(output)
        temp = torch.cat(temp, dim=1)
        print(f'Final output {temp.shape}')
        return temp

    def run(self):
        try:
            self.train()
            #self.test()
            
            #self.svm_train_one_epoch_mri()
            self.test_svm()         
            
            #print(x_r[y == 1, 0])
            self.summary_writer.flush()
            self.summary_writer.close()
        except KeyboardInterrupt:
            print('done')

    def train(self):
        for epoch in range(self.config.max_epoch):
            #self.train_one_epoch()
            #self.validate()
            self.svm_train_one_epoch()
            self.current_epoch += 1

    def svm_train_one_epoch(self):
        
        for batch_idx, (mri_data, pet_data, target) in enumerate(self.data_loader.train_loader, start=1):
            #Only the mri and pet data needs to go to gpu since we train svm on cpu later
            mri_data, pet_data = mri_data.to(self.device), pet_data.to(self.device)
            target = target.numpy()
            _ = self.mri_model(mri_data)
            _ = self.pet_model(pet_data)
            fused_output = self.simple_concat().cpu().numpy()
            #fused_output = self.multimodal_gated_fusion().cpu().numpy()
            l = np.unique(target)
            #print(l)
            self.clf.partial_fit(fused_output, target, classes=l)
            #self.ipca.partial_fit(fused_output)

    def test_svm(self):

        #dim_reducer = LinearDiscriminantAnalysis(n_components=2)
        
        preds = []
        ground_truths = []
        outputs = []
        
        for batch_idx, (mri_data, pet_data, target) in enumerate(self.data_loader.test_loader, start=1):
            #Only the mri and pet data needs to go to gpu since we train svm on cpu later
            mri_data, pet_data = mri_data.to(self.device), pet_data.to(self.device)
            target = target.numpy()
            _ = self.mri_model(mri_data)
            _ = self.pet_model(pet_data)
            fused_output = self.simple_concat().cpu().numpy()
            #fused_output = self.multimodal_gated_fusion().cpu().numpy()
            
            
            preds.append(self.clf.predict(fused_output)) 
            ground_truths.append(target)
            

        ground_truths = np.concatenate(ground_truths)
        preds = np.concatenate(preds)
        print(ground_truths.shape, preds.shape)

            #outputs.append(fused_output)
        #plot_lda(outputs, ground_truths)
        #ground_truths = np.concatenate(ground_truths)
        #outputs = np.concatenate(outputs)
        #print(classification_report(ground_truths, preds, target_names=target_names))
        #print(accuracy_score(ground_truths, preds))
        #plot_confusion_matrix(confusion_matrix(ground_truths, preds))
        
        #print(preds[:100])
        #print(ground_truths[:100])
        print(np.sum(preds == ground_truths) / preds.shape[0])

        #print(self.kmeans.score(preds, ground_truths))

    def test_svm_mri(self):

        #dim_reducer = LinearDiscriminantAnalysis(n_components=2)
        
        preds = []
        ground_truths = []
        outputs = []
        
        for batch_idx, (_, pet_data, target) in enumerate(self.data_loader.test_loader, start=1):
            #Only the mri and pet data needs to go to gpu since we train svm on cpu later
            pet_data = pet_data.to(self.device)
            target = target.numpy()
            _ = self.pet_model(pet_data)
            #fused_output = self.simple_concat().cpu().numpy()
            fused_output = self.singlemode_gate_fusion(self.pet_interm_feats.values()).cpu().numpy()
            
            preds.append(self.clf.predict(fused_output)) 
            ground_truths.append(target)
            
        preds = np.concatenate(preds)
        ground_truths = np.concatenate(ground_truths)

        print( np.sum(preds == ground_truths) * 100 / preds.shape[0] )

        
    def svm_train_one_epoch_mri(self):
        
        for batch_idx, (_, pet_data, target) in enumerate(self.data_loader.train_loader, start=1):
            #Only the mri and pet data needs to go to gpu since we train svm on cpu later
            pet_data = pet_data.to(self.device)
            target = target.numpy()
            _ = self.pet_model(pet_data)
            #fused_output = self.simple_concat().cpu().numpy()
            fused_output = self.singlemode_gate_fusion(self.pet_interm_feats.values()).cpu().numpy()
            l = np.unique(target)
            
            if batch_idx == 1:
                print(l)
                self.clf.partial_fit(fused_output, target, classes=l)
            else:
                self.clf.partial_fit(fused_output, target)
            #self.ipca.partial_fit(fused_output)

    def train_one_epoch(self):
        self.classifier.train()
        epoch_loss = 0.0
        corrects = 0
        
        for batch_idx, (mri_data, pet_data, target) in enumerate(self.data_loader.train_loader, start=1):
            mri_data, pet_data, target = mri_data.to(self.device), pet_data.to(self.device), target.to(self.device)

            _ = self.mri_model(mri_data)
            _ = self.pet_model(pet_data)
                #print(list(zip(*self.mri_interm_feats.values(), *self.pet_interm_feats.values())))
            fused_output = self.multimodal_gated_fusion()

            self.optimizer.zero_grad()
            output = self.classifier(fused_output)
            loss = self.loss(output, target)
            _, preds = torch.max(output, 1)
            corrects += torch.sum(preds == target).item()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.3f}'.format(
                    self.current_epoch, 
                    batch_idx * len(target), 
                    len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader), 
                    loss.item()
                    ))
            
        epoch_acc   = corrects / len(self.data_loader.train_loader.dataset)
        epoch_loss /= len(self.data_loader.train_loader)
        self.logger.info('Epoch Trainig Acc: {:.2f}%'.format(100. * epoch_acc))

        self.summary_writer.add_scalar('Loss/train', epoch_loss, self.current_epoch)
        self.summary_writer.add_scalar('Acc/train', epoch_acc, self.current_epoch)
        #self.summary_writer.add_scalar('Loss/train', epoch_loss, self.current_epoch)
        #self.summary_writer.add_scalar('Acc/train', acc, self.current_epoch)

    def validate(self):
        self.classifier.eval()
        val_loss = 0.0
        corrects = 0
        
        with torch.no_grad():
            for mri_data, pet_data, target in self.data_loader.val_loader:
                mri_data, pet_data, target = mri_data.to(self.device), pet_data.to(self.device), target.to(self.device)

                _ = self.mri_model(mri_data)
                _ = self.pet_model(pet_data)
                    #print(list(zip(*self.mri_interm_feats.values(), *self.pet_interm_feats.values())))
                fused_output = self.multimodal_gated_fusion()

                output = self.classifier(fused_output)
                val_loss += self.loss(output, target).item()
                _, preds = torch.max(output, 1)
                corrects += torch.sum(preds == target).item()
        


        val_acc   = corrects / len(self.data_loader.val_loader.dataset)
        val_loss /= len(self.data_loader.val_loader)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model = self.classifier.state_dict()

        
        self.logger.info('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, 
            corrects, 
            len(self.data_loader.val_loader.dataset),
            100. * val_acc
            ))
        
        self.summary_writer.add_scalar('Loss/val', val_loss, self.current_epoch)
        self.summary_writer.add_scalar('Acc/val', val_acc, self.current_epoch)
    
    def test(self):
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        self.classifier.load_state_dict(self.best_model)
        self.classifier.eval()
        loss = 0.0
        corrects = 0
        
        predictions = []
        ground_truths = []

        with torch.no_grad():
            for mri_data, pet_data, target in self.data_loader.test_loader:
                mri_data, pet_data, target = mri_data.to(self.device), pet_data.to(self.device), target.to(self.device)

                _ = self.mri_model(mri_data)
                _ = self.pet_model(pet_data)
                    #print(list(zip(*self.mri_interm_feats.values(), *self.pet_interm_feats.values())))
                fused_output = self.multimodal_gated_fusion()

                output = self.classifier(fused_output)
                loss += self.loss(output, target).item()
                _, preds = torch.max(output, 1)
                predictions.append(preds)
                corrects += torch.sum(preds == target).item()
                ground_truths.append(target)
        
        acc   = corrects / len(self.data_loader.test_loader.dataset)
        loss /= len(self.data_loader.test_loader)
  
        predictions = torch.cat(predictions).cpu().numpy()
        ground_truths = torch.cat(ground_truths).cpu().numpy()

        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, 
            corrects, 
            len(self.data_loader.test_loader.dataset),
            100. * acc
            ))

        target_names = ['CN', 'nMCI', 'cMCI', 'AD']
        print(confusion_matrix(ground_truths, predictions))
        print(classification_report(ground_truths, predictions, target_names=target_names))
        #self.summary_writer.add_scalar('Loss/train', loss, self.current_epoch)
        #self.summary_writer.add_scalar('Acc/train', acc, self.current_epoch)

def plot_confusion_matrix(matrix):
    
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = matrix / row_sums
    np.fill_diagonal(norm_matrix, 0)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].matshow(matrix, cmap='gray')
    axes[1].matshow(norm_matrix, cmap='gray')
    fig.suptitle('CN:0 | nMCI:1 | cMCI:2 | AD:3')
    plt.show()

def plot_lda(data, labels):
    colors       = ['navy', 'turquoise', 'darkorange', 'red']
    target_names = ['CN',   'nMCI',      'cMCI',       'AD'] 
    for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=color, alpha=.8, lw=2, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()
