import sys
sys.path.append("./pix2pix")
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

class Wrapper():
    """docstring for Wrapper"""
    def __init__(self, edge_name="edge_map", ske_name="skeleton_map"):
        sys.argv.extend(['--dataroot', ""])
        opt = TrainOptions().parse()
        opt.model = "pix2pix"
        opt.netD = "n_layers"
        opt.netG = "resnet_9blocks"
        opt.n_layers_D = 5
        opt.input_nc = 2
        opt.output_nc = 1
        opt.continue_train = True
        opt.lambda_L1 = 100
        opt.gan_mode = "vanilla"
        opt.no_dropout = False
        opt.norm = "batch"
        opt.pool_size = 0
        opt.name = edge_name
        self.edge_mapping_model = create_model(opt)
        self.edge_mapping_model.setup(opt)
        opt.name = ske_name
        opt.n_layers_D = 6
        opt.input_nc = 1
        self.skeleton_mapping_model = create_model(opt)
        self.skeleton_mapping_model.setup(opt)
        print("Pix2pix mapping models loaded successfully!")

    def edge_to_skeleton(self, input):
        self.skeleton_mapping_model.set_input(input)
        self.skeleton_mapping_model.optimize_parameters()
        return self.skeleton_mapping_model.fake_B

    def skeleton_to_edge(self, input):
        self.edge_mapping_model.set_input(input)
        self.edge_mapping_model.optimize_parameters()
        return self.edge_mapping_model.fake_B
        
    def map_and_optimize(self, edge_img, ske_img, real_edge, real_skeleton):
        edge_input = {'A': edge_img, 'B': real_skeleton, 'A_paths': None}
        new_skeleton = self.edge_to_skeleton(edge_input)
        ske_input = {'A': ske_img, 'B': real_edge, 'A_paths': None}
        new_edge = self.skeleton_to_edge(ske_input)

        return new_edge.cuda(), new_skeleton.cuda()

    def save_models(self, outDir, edge_net_D, edge_net_G, skeleton_net_D, skeleton_net_G):
        torch.save(self.edge_mapping_model.netD.state_dict(), outDir + edge_net_D)
        torch.save(self.edge_mapping_model.netG.state_dict(), outDir + edge_net_G)
        torch.save(self.skeleton_mapping_model.netD.state_dict(), outDir + skeleton_net_D)
        torch.save(self.skeleton_mapping_model.netG.state_dict(), outDir + skeleton_net_G)
