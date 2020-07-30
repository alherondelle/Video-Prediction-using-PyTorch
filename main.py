# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from utils.start_tensorboard import run_tensorboard
from models.seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from data.MovingMNIST import MovingMNIST
from data.MeteosatLoader import import_bucket, unzip 
from data.processing_data import split_dataset4KITTI, process_data
import data.ImageTestDataset
import data.ImageTrainDataset
from data.ImageTestDataset import ImageTestDataset
from data.ImageTrainDataset import ImageTrainDataset
import argparse
import torchvision.transforms as transforms 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')

opt = parser.parse_args()


##########################
######### MODEL ##########
##########################

class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size
        self.n_steps_past = 10
        self.n_steps_ahead = 10  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        # Permutation pour mettre la dim des channels à la bonne place 
        y_hat = y_hat.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)
        preds = torch.cat([x.cpu(), y_hat.cpu()], dim=1)[0]
        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        '''difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu(), difference.cpu()], dim=1)[
            0].unsqueeze(1)
        print('preds shape : ', preds.shape, 'y_plot shape : ', y_plot.shape, 'difference_plot shape : ', difference_plot.shape)
        '''
        # concat all images
        final_image = torch.cat([preds, y_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()

        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?
        #Permutation ajoutée pour avoir correspondance entre Y et Y_hat dans la loss function : peut-être pas une solution
        y_hat = y_hat.permute(0,2,3,4,1)
        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x, y_hat, y)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))

    @pl.data_loader
    def train_dataloader(self):
        transforms_ = [transforms.Normalize((0.500,0.500,0.500),(0.500,0.500,0.500))]
        train_data = ImageTestDataset(
            'data/METEOSAT/pkl/train_data.pkl',
            transforms_=transforms_,
            nt=20
            )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)

        return train_loader

    @pl.data_loader
    def test_dataloader(self):
        test_data = ImageTestDataset(
            'data/METEOSAT/pkl/test_data.pkl',
            transforms_=transforms_,
            nt=20
            )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True)

        return test_loader



def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=3)

    model = MovingMNISTLightning(model=conv_lstm_model)

    trainer = Trainer(max_epochs=opt.epochs,
                      gpus=opt.n_gpus,
                      distributed_backend='dp',
                      early_stop_callback=False,
                      use_amp=opt.use_amp
                      )

    trainer.fit(model)


if __name__ == '__main__':
    import_bucket()
    unzip()
    split_dataset4KITTI('data/METEOSAT')
    process_data()
    torch.multiprocessing.set_start_method('spawn')
    p1 = Process(target=run_trainer)                    # start trainer
    p1.start()
    p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
    p2.start()
    p1.join()
    p2.join()