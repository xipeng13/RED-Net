# Xi Peng, May 2017
import argparse
import os
from utils import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_dir', type=str, default='./dataset',
                    help='training data or listfile path')
        self.parser.add_argument('--checkpoint_dir',type=str, default='./checkpoint',
                    help='checkpoints are saved here')
        self.parser.add_argument('--nThreads', type=int, default=4,
                    help='number of data loading threads')
        self.parser.add_argument('--ifValidate', type=bool, default=True,
                    help='evaluate model on validation set')
        self.parser.add_argument('--use_visdom', type=bool, default=True,
                    help='use visdom to display')
        self.parser.add_argument('--use_html', type=bool, default=False,
                    help='use html to store images')
        self.parser.add_argument('--display_winsize', type=int, default=256,
                    help='display window size') ##TO DO

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        self.opt.name = self.name # experiment name

        #str_ids = self.opt.gpu_ids.split(',')
        #self.opt.gpu_ids = []
        #for str_id in str_ids:
        #    id = int(str_id)
        #    if id >= 0:
        #        self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoint_dir, self.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
