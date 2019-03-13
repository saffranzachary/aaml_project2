
# coding: utf-8

# # DCGAN PYTORCH IMPLEMENTATION
# Forked from here: https://github.com/pytorch/examples


from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

import time
import pdb
from tqdm import tqdm, tqdm_notebook

# from models import *

def plotline(data, xlabel, ylabel, title, path):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path)
    plt.clf()

def show(img):
    npimg = img.numpy()
    npimg = npimg-np.amin(npimg)
    npimg = npimg/np.amax(npimg)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis("off")
        
    plt.show()


#manualSeed = random.randint(1, 10000)
manualSeed = 4532
print("Random Seed: ", manualSeed)
random.seed(manualSeed)

torch.manual_seed(manualSeed)
cudnn.benchmark = True

use_cuda=True
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

if torch.cuda.is_available() and not use_cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


imageSize = 64 # square images for now!

ngpu = int(1) # use one GPU
nz = int(100) # code dimension (This is the() random noise) input dimension of the generator network)
ngf = int(64) # output dimension of the generator network
ndf = int(64) # input dim (image size) for the discriminator net
nc = 3 # number of input channels (e.g. 3 for RGB channels)


##########################################################################
### Provided network architectures - used for reference
##########################################################################

# Let us create the Generator network
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise):
        if isinstance(noise.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, noise, range(self.ngpu))
        else:
            output = self.main(noise)
        return output


# Let us create the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)




##########################################################################
### My modification of provided network architectures
##########################################################################

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class EmbeddingNorm(nn.Module):
    def __init__(self, scale = 1):
        super().__init__()
        self.scale = scale 

    def forward(self, x):
        # divide by embedding-wise norm
        norm = x.norm(dim = 1).unsqueeze(dim = 1)
        return self.scale * x / norm

class MyGenerator(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class MyDiscriminator(nn.Module):
    def block(self, in_filters, out_filters):
            return nn.Sequential(
                nn.Conv2d(
                    in_filters, 
                    out_filters, 
                    kernel_size = 3, 
                    stride = 2, 
                    padding = 1,
                    bias = False
                ),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
                # nn.Dropout2d(p = 0.2)
            )

    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.image_embedding = nn.Sequential(
            # input is 3 x 64 x 64
            self.block(3, 64),
            # state size. 64 x 32 x 32
            self.block(64, 128),
            # state size. 128 x 16 x 16
            self.block(128, 256),
            # state size. 256 x 8 x 8
            self.block(256, 512),
            # state size. 512 x 4 x 4
            self.block(512, 1024),
            # state size, 1024 x 2 x 2
            nn.Conv2d(
                1024,
                2048,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                bias = False
            ),
            # state size, 2048 x 1 x 1
            Flatten(),
            EmbeddingNorm()
        )
        self.validity_score = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            embedding = nn.parallel.data_parallel(self.image_embedding, input, range(self.ngpu))
        else:
            embedding = self.image_embedding(input)
        validity = self.validity_score(embedding) 
        return validity.view(-1), embedding



##############################################################################
### V2s are implementations more closely adapted from provided architectures
##############################################################################


# Let us create the Generator network
class MyGeneratorV2(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.label_embedding = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz + NUM_CLASSES, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, label):
        input = torch.cat((noise, self.label_embedding(label)), dim=1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# Let us create the Discriminator network
class MyDiscriminatorV2(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*8) x 4 x 4
        )
        self.embedding = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False), # try just flattening as well as projecting down
            # state size. ndf x 1 x 1
            Flatten(),
            EmbeddingNorm(scale = 16)
        )
        self.validity_score = nn.Sequential(
            # state size. ndf
            nn.Linear(ndf * 8 * 16, 1),
            # state size. 1
            nn.Sigmoid()
        )
        self.classification = nn.Linear(ndf * 8 * 16, NUM_CLASSES)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        embedding = self.embedding(output)
        validity = self.validity_score(embedding)
        classification = self.classification(embedding)
        # return output.view(-1, 1).squeeze(1)
        return validity.view(-1), embedding, classification




###############################################################
### TRAIN 
###############################################################


# custom weights initialization called on netG and netD
def weights_init(m):
    t = type(m)
    if t in {nn.Linear, nn.Conv1d, nn.Conv2d}:
        nn.init.xavier_uniform(m.weight)
    elif t in {nn.BatchNorm2d}:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
    return



USE_GPU = torch.cuda.is_available()
NUM_WORKERS = 4

DATA_PATH = "./data/"
OUT_PATH = "./results/"

NUM_CLASSES = 8

SAVE_MODEL_INTERVAL = 4
BATCH_PRINT_INTERVAL = 50

NUM_EPOCHS = 200
BATCH_SIZE = 128

REAL_G_LABEL = 1
REAL_D_LABEL = 1
FAKE_LABEL = 0

LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002

NUM_D_UPDATES = 1



class Trainer():
    def __init__(self, data, netD, netG, critD, critG, optimD, optimG):
        self.data = data
        self.netD = netD
        self.netG = netG
        self.critD = critD
        self.critG = critG
        self.optimD = optimD
        self.optimG = optimG
        self.fixed_noise = torch.FloatTensor(NUM_CLASSES**2, nz, 1, 1).normal_(0, 1) # to monitor progress
        self.fixed_labels = torch.Tensor([[c] * NUM_CLASSES for c in range(NUM_CLASSES)]).view(-1).long()

    def run(self):
        training_losses_D = []
        training_losses_G = []
        training_confusion = []
        training_start = time.time()
        for epoch in range(61, NUM_EPOCHS):
            self.netD.train()
            self.netG.train()
            start = time.time()
            lossD, lossG, confusion = self.run_epoch()
            end = time.time()
            print(
                "Epoch", epoch, "/", NUM_EPOCHS,
                "Training",
                "| Loss D", lossD,
                "| Loss G", lossG, 
                "| Confusion", confusion,
                "| Time", end - start
            )
            training_losses_D.append(lossD)
            training_losses_G.append(lossG)
            training_confusion.append(confusion)
            self.save_fixed_noise_images(epoch)
            self.plot_losses(training_losses_G, training_losses_D, training_confusion)
            if SAVE_MODEL_INTERVAL > 0 and epoch % SAVE_MODEL_INTERVAL == 0:
                print("saving model from epoch", epoch)
                torch.save(self.netD, OUT_PATH + "trained_D_" + str(epoch) + ".pt")
                torch.save(self.netG, OUT_PATH + "trained_G_" + str(epoch) + ".pt")
        training_end = time.time()
        print("Done training! Time:", training_end - training_start)
        return training_losses_D, training_losses_G
    
    def run_epoch(self):
        total_loss_D = 0
        total_loss_G = 0
        confusion = 0
        num_batches = len(self.data)
        num_samples = len(self.data.dataset)
        print("running epoch with", num_samples, "samples in", num_batches, "batches")
        batch_start_time = time.time()
        for batch_ind, (real_imgs, labels) in tqdm(enumerate(self.data)):

            # Discriminator update 
            real_imgs = Variable(real_imgs)
            labels = Variable(labels)
            for i in range(NUM_D_UPDATES):
                self.netG.zero_grad()
                self.netD.zero_grad()
                noise = torch.FloatTensor(len(real_imgs), nz, 1, 1).normal_(0, 1)
                noise = Variable(noise)
                if USE_GPU:
                    real_imgs = real_imgs.cuda()
                    labels = labels.cuda()
                    noise = noise.cuda()
                gen_imgs = self.netG(noise, labels)
                d_out_real,_,real_classification = self.netD(real_imgs)
                d_out_fake,_,fake_classification = self.netD(gen_imgs)
                d_loss = self.critD(d_out_real, d_out_fake, real_classification, fake_classification, labels, batch_ind)
                d_loss.backward()
                self.optimD.step()
            total_loss_D += d_loss.data[0]

            # confusion
            confusion += d_out_fake.round().sum().data[0]

            # generator update
            _, real_embeddings, _ = self.netD(real_imgs) # for embedding loss
            self.netG.zero_grad()
            self.netD.zero_grad()
            noise = torch.FloatTensor(len(real_imgs), nz, 1, 1).normal_(0, 1)
            noise = Variable(noise)
            if USE_GPU:
                noise = noise.cuda()
            gen_imgs = self.netG(noise, labels)
            d_out, fake_embeddings, fake_classification = self.netD(gen_imgs)
            g_loss = self.critG(d_out, fake_embeddings, fake_classification, real_embeddings, labels, batch_ind)
            g_loss.backward()
            self.optimG.step()
            total_loss_G += g_loss.data[0]

            if BATCH_PRINT_INTERVAL > 0 and batch_ind % BATCH_PRINT_INTERVAL == 0:
                batch_end_time = time.time()
                print(
                    "batch", batch_ind,
                    "d-loss", d_loss.data[0],
                    "g-loss", g_loss.data[0],
                    "time", batch_end_time - batch_start_time
                )
                batch_start_time = batch_end_time
        total_loss_D /= num_batches
        total_loss_G /= num_batches
        confusion /= num_samples
        return total_loss_D, total_loss_G, confusion

    def plot_losses(self, g_loss, d_loss, confusion):
        plotline(g_loss, "Epoch", "G Loss", "G Loss", OUT_PATH + "g_loss.png")
        plotline(d_loss, "Epoch", "D Loss", "D Loss", OUT_PATH + "d_loss.png")
        plotline(confusion, "Epoch", "Confusion", "Confusion", OUT_PATH + "confusion.png")

    def save_fixed_noise_images(self, epoch):
        self.netG.eval()
        if USE_GPU:
            self.fixed_noise = self.fixed_noise.cuda()
            self.fixed_labels = self.fixed_labels.cuda()
        fake = self.netG(Variable(self.fixed_noise), Variable(self.fixed_labels))
        print("saving imges from epoch", epoch)
        vutils.save_image(
            fake.data,
            '%s/gen_results_epoch_%03d.png' % (OUT_PATH, epoch),
            normalize=True
        )


dataset = dset.ImageFolder(
    root=DATA_PATH,
    transform=transforms.Compose([
        transforms.Scale(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers= NUM_WORKERS if USE_GPU else 0,
    pin_memory = USE_GPU
)

netD = MyDiscriminatorV2(1)
netD.apply(weights_init)
netG = MyGeneratorV2(1)
netG.apply(weights_init)


netD = torch.load(OUT_PATH + "trained_D_60.pt")
netG = torch.load(OUT_PATH + "trained_G_60.pt")


if USE_GPU:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), LEARNING_RATE_D, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), LEARNING_RATE_G, betas=(0.5, 0.999))


bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()

def critDBCE(real_scores, fake_scores):
    loss = bce_loss
    real_labels = torch.FloatTensor(len(real_scores)).fill_(REAL_D_LABEL)
    fake_labels = torch.FloatTensor(len(fake_scores)).fill_(FAKE_LABEL)
    if USE_GPU:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
    return loss(real_scores, Variable(real_labels)) + loss(fake_scores, Variable(fake_labels))

def critDMSE(real_scores, fake_scores):
    loss = mse_loss
    real_labels = torch.FloatTensor(len(real_scores)).fill_(REAL_D_LABEL)
    fake_labels = torch.FloatTensor(len(fake_scores)).fill_(FAKE_LABEL)
    if USE_GPU:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
    return loss(real_scores, Variable(real_labels)) + loss(fake_scores, Variable(fake_labels))

def critDWasserstein(real_scores, fake_scores):
    return -torch.mean(real_scores) + torch.mean(fake_scores)

def critGBCE(fake_scores):
    labels = torch.FloatTensor(len(fake_scores)).fill_(REAL_G_LABEL)
    if USE_GPU:
        labels = labels.cuda()
    return bce_loss(fake_scores, Variable(labels))

def critGMSE(fake_scores):
    labels = torch.FloatTensor(len(fake_scores)).fill_(REAL_G_LABEL)
    if USE_GPU:
        labels = labels.cuda()
    return mse_loss(fake_scores, Variable(labels))

def critGWasserstein(fake_scores):
    return -torch.mean(fake_scores)

def critGFeatureMatching(fake_embeddings, real_embeddings):
    return torch.mean((fake_embeddings - real_embeddings) ** 2, dim = 1).mean()

def critGFMBCE(fake_embeddings, real_embeddings, fake_scores):
    return critGBCE(fake_scores) + critGFeatureMatching(fake_embeddings, real_embeddings)



PDB_LOSS = False

def critDConditional(real_validity, fake_validity, real_classification, fake_classification, labels, batch):
    real_classification_loss = cross_entropy(real_classification, labels) 
    fake_classification_loss = cross_entropy(fake_classification, labels)
    bce = critDBCE(real_validity, fake_validity)
    if batch % BATCH_PRINT_INTERVAL == 0:
        if PDB_LOSS:
            pdb.set_trace()
        print(
            "d-loss:",
            "real_ce", real_classification_loss.data[0], 
            "fake_ce", fake_classification_loss.data[0], 
            "bce", bce.data[0]
        )
    return real_classification_loss + fake_classification_loss + bce

def critGConditional(fake_validity, fake_embeddings, fake_classification, real_embeddings, labels, batch):
    bce = critGBCE(fake_validity)
    classification = cross_entropy(fake_classification, labels)
    feature_matching = critGFeatureMatching(fake_embeddings, real_embeddings)
    if batch % BATCH_PRINT_INTERVAL == 0:
        if PDB_LOSS:
            pdb.set_trace()
        print(
            "g-loss:",
            "fm", feature_matching.data[0], 
            "ce", classification.data[0], 
            "bce", bce.data[0]
        )
    return feature_matching + classification + bce





trainer = Trainer(
    dataloader, 
    netD, 
    netG, 
    critDConditional, 
    critGConditional, 
    optimizerD, 
    optimizerG
)
trainer.run()

torch.save(netD, OUT_PATH + "final_trained_D.pt")
torch.save(netG, OUT_PATH + "final_trained_G.pt")

