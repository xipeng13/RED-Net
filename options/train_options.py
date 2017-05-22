# Xi Peng, May 2017
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.name = 'res50_lr0.01_res3x'
        self.parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
        self.parser.add_argument('--bs', type=int, default=10,
                    help='mini-batch size')
        self.parser.add_argument('--load_pretrained', type=bool, default=True,
                    help='use pretrained model')
        self.parser.add_argument('--load_checkpoint', type=bool, default=False,
                    help='use checkpoint model')
        self.parser.add_argument('--resume_epoch', type=int, default=1,
                    help='manual epoch number to restart')
        self.parser.add_argument('--nEpochs', type=int, default=30,
                    help='number of total training epochs to run')
        self.parser.add_argument('--best_rmse', type=float, default=1.,
                    help='best result until now')

        self.parser.add_argument('--train_list', type=str, default='train_list.txt',
                    help='train image list')
        self.parser.add_argument('--val_list', type=str, default='val_list.txt',
                    help='validation image list')

        self.parser.add_argument('--print_freq', type=int, default=20,
                    help='print log every n iterations')
        self.parser.add_argument('--display_freq', type=int, default=20,
                    help='display figures every n iterations')

        self.parser.add_argument('--momentum', type=float, default=0.90,
                    help='momentum term of sgd')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay term of sgd')
        self.parser.add_argument('--beta1', type=float, default=0.5,
                    help='momentum term of adam') 
        self.isTrain = True
