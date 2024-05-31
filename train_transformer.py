import torch
import argparse
from transformer.transformer import Transformer
from transformer.dataset import SyntheticStereoPoseDataset
from transformer.train_utils import train, evaluate, visualize, collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='synthetic_transformer') # To be loaded from the data/ folder
    parser.add_argument('--num_layers', type=int, default=5, help='Number of transformer layers')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout in the transformer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='If non-zero, load model from this epoch')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')

    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')

    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--stereo', action='store_true', help='Use stereo keypoints')
    parser.add_argument('--rot6d', action='store_true', help='Use 6D rotation')
    parser.add_argument('--pos', action='store_true', help='Use positional embeddings for keypoints')
    parser.add_argument('--cls', action='store_true', help='Add object classes')

    args = parser.parse_args()
    num_kps = 12 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = f'transformer_{args.num_layers}_{num_kps}'
    model_name += f'_{args.hidden_dim}_{args.num_heads}_{args.dropout}'
    if args.aug: model_name += '_aug'
    if args.stereo: model_name += '_stereo'
    if args.rot6d: model_name += '_6d'
    if args.pos: model_name += '_pos'
    if args.cls: model_name += '_cls'
    
    mode = 'train'

    train_dataset = SyntheticStereoPoseDataset('train', mask=True, rot6d = args.rot6d, augment=args.aug,
                                               stereo='stereo' in model_name, dataset=args.dataset_name, num_kps=num_kps, 
                                               pos_embed='pos' in model_name, cls_embed='cls' in model_name)
    print('Train Dataset length:', len(train_dataset))
    bs = 64
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # test dataloader
    val_dataset = SyntheticStereoPoseDataset('val', stereo='stereo' in model_name, rot6d = args.rot6d,
                                             dataset=args.dataset_name, num_kps=num_kps, augment=False, 
                                             pos_embed='pos' in model_name, cls_embed='cls' in model_name)
    print('Validation Dataset length:', len(val_dataset))
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Define hyperparameters
    input_dim = 2 + 2 + 14 + 12 # 2D keypoint + one-hot obj class
    output_dim = 3 + 3 + 3 + 1  if not args.rot6d else 6 + 3 + 3 + 1 # 6D rotation + 3 translation + 1 articulation

    # print all variables
    print(f'Input dim: {input_dim}, Output dim: {output_dim}, Hidden dim: {args.hidden_dim}, Num layers: {args.num_layers}')

    # Create a Transformer model
    model = Transformer(input_dim, args.hidden_dim, output_dim, args.num_layers, args.num_heads, args.dropout).to(device)

    start_epoch = 0
    # load pretrained model
    if args.start_epoch > 0:
        model.load_state_dict(torch.load(f'checkpoints/{model_name}/{args.start_epoch}.pth'))

    # print number of parameters of the model
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    # Train the model
    if args.train:
        train(model, trainloader, valloader, start_epoch+1, args.num_epochs, args.lr, device, model_name)
    
    if args.test:
        test_dataset = SyntheticStereoPoseDataset('test', mask=True, rot6d = args.rot6d,
                                               stereo='stereo' in model_name, dataset=args.dataset_name, num_kps=num_kps, 
                                               pos_embed='pos' in model_name, cls_embed='cls' in model_name)
        print('Test Dataset length:', len(test_dataset))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn)
        evaluate(testloader, model, device, model_name, half=False)
    # visualize(model, valloader, device)