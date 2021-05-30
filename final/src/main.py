import argparse

import torch

from datasets import prepare_datasets
from eval import Lfw_evaluation, visualize_embeding
from face_net import Pretrained
from train import train


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    casia_dir = '../datasets/casia_with_masks'       
    lfw_dir = '../datasets/lfw_with_masks'
    lfw_pairs_path = '../datasets/pairs_with_masks.txt'
    log_dir = '../logs'
    model_dir = '../models'

    train_acc_interval = 100
    test_acc_interval = 10000

    #datasets
    train_loader, test_loader, visualization_loader = prepare_datasets(casia_dir, args.batch_size)

    #model
    model = Pretrained(args.arcface)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    model.to(device)
    
    #evaluation/training
    if args.visualize_embedings:
        visualize_embeding(model, visualization_loader, device)
    elif args.evaluate:
        lfw_ver = Lfw_evaluation(lfw_dir, lfw_pairs_path, device)
        acc, _, val, _ = lfw_ver.eval(model, far_target=args.far)
        print('LFW - accurancy: ' + str(acc) + ', val: ' + str(val) + ' (@far = ' + str(args.far) + ')')
    elif args.plot_roc:
        lfw_ver = Lfw_evaluation(lfw_dir, lfw_pairs_path, device)
        lfw_ver.plot_roc(model)
    else:
        train(model, device, args.arcface, train_loader, test_loader, log_dir, model_dir, train_acc_interval, test_acc_interval, args.epochs_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="set batch size", action="store", type=int, default=52)
    parser.add_argument("-e", "--epochs_num", help="set number of epochs for training", action="store", type=int, default=10)
    parser.add_argument("-l", "--load_model", help="set path to model to be loaded", action="store", type=str, default=None)
    parser.add_argument("-a", "--arcface", help="use ArcFace as loss function during training", action='store_true')
    parser.add_argument("-e", "--evaluate", help="evaluate on LFW", action='store_true')
    parser.add_argument("-f", "--far", help="far used while verifying on LFW", type=float, default=0.001)
    parser.add_argument("-r", "--plot_roc", help="plot ROC using LFW", action='store_true')
    parser.add_argument("-v", "--visualize_embedings", help="plot visualized embedings", action='store_true')
    args = parser.parse_args()
    main(args)
