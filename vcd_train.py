import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataset import dataset_dict
import time
import datetime
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
# Assuming `ChangeDetectionNet` and your dataset class `ChangeDetectionDataset` are defined elsewhere
from vcd import ChangeDetectionNet
# from dataset import ChangeDetectionDataset

def visualize(first_img_t0, first_img_t1, first_label):

    # Convert tensors to PIL images for visualization
    img_t0_pil = TF.to_pil_image(first_img_t0)
    img_t1_pil = TF.to_pil_image(first_img_t1)

    # Convert the mask to 0 and 255 (False to 0, True to 255)
    mask_converted = first_label * 255  # Converts boolean to 0 (no change) and 255 (change)
    label_pil = TF.to_pil_image(mask_converted.byte())  # Ensure it’s in byte format for display

    # Plot the images and mask
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Image at time T0
    axs[0].imshow(img_t0_pil)
    axs[0].set_title("Image T0")
    axs[0].axis('off')

    # Image at time T1
    axs[1].imshow(img_t1_pil)
    axs[1].set_title("Image T1")
    axs[1].axis('off')

    # Ground truth change mask
    axs[2].imshow(label_pil, cmap='gray')
    axs[2].set_title("Ground Truth Change Mask")
    axs[2].axis('off')

    # Save the figure
    plt.savefig("visualization.png", bbox_inches='tight')
    plt.close(fig)

f_f1=open("image_f1.txt","w")

def CD_evaluate(model, data_loader, device, save_imgs_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Prec', utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Rec', utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('Acc', utils.SmoothedValue(window_size=1, fmt='{value:.3f} ({global_avg:.3f})'))
    metric_logger.add_meter('F1score', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = 'Test:'
    
    with torch.no_grad():
        for images, target in metric_logger.log_every(data_loader, 100, header):
            # Move target mask to the device
            target = target.to(device)

            # Split images into image_t0 and image_t1
            img_t0_batch, img_t1_batch = [], []
            for img in images:
                img_t0, img_t1 = torch.split(img, 3, dim=0)  # Split along the channel dimension (0)
                img_t0_batch.append(img_t0)
                img_t1_batch.append(img_t1)
            
            # Stack and move to device
            img_t0_batch = torch.stack(img_t0_batch).to(device)
            img_t1_batch = torch.stack(img_t1_batch).to(device)

            
            # Model forward pass
            output = model(img_t0_batch, img_t1_batch)
            # print(output.shape)
            # print("Model output logits (min, max):", output.min().item(), output.max().item())
            # # If the model output is an OrderedDict, extract the 'out' key
            # if isinstance(output, OrderedDict):
            #     output = output['out']
            
            # Convert logits to predicted mask by taking the argmax along the class dimension
            mask_pred = torch.argmax(output, dim=1)  # Shape: [batch_size, height, width]
            mask_pred = mask_pred > 0
            # probs = F.softmax(output, dim=1)  # Apply softmax across the class dimension
            # print("Model output probabilities (min, max):", probs.min().item(), probs.max().item())
            # print(mask_pred.shape)
            # Ground truth mask: ensure it's in binary format
            mask_gt = (target > 0).squeeze(1)  # Remove any singleton dimension if present
            # print(mask_gt.shape)
            # print("Unique values in predictions:", mask_pred.unique())
            # print("Unique values in ground truth:", mask_gt.unique())
            # Compute metrics
            precision, recall, accuracy, f1score = utils.CD_metric_torch(mask_pred, mask_gt)
            
            # Update metrics
            metric_logger.Prec.update(precision.mean(), n=len(precision))
            metric_logger.Rec.update(recall.mean(), n=len(precision))
            metric_logger.Acc.update(accuracy.mean(), n=len(precision))
            metric_logger.F1score.update(f1score.mean(), n=len(f1score))
            
            # Save images if specified
            save_file_name = "{}_{}.png".format(utils.get_rank(), metric_logger.F1score.count)
            if save_imgs_dir:
                assert len(precision) == 1, "save_imgs_dir requires batch_size=1"
                output_pil = data_loader.dataset.get_pil(images[0], mask_gt[0], mask_pred[0])
                output_pil.save(os.path.join(save_imgs_dir, save_file_name))
                f_f1.write(save_file_name + " " + str(f1score[0].item()) + "\n")

        # Synchronize metrics across processes if using distributed training
        metric_logger.synchronize_between_processes()

    # Print final metrics
    print("{} {} Total: {} Metric Prec: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
        header,
        data_loader.dataset.name,
        metric_logger.F1score.count,
        metric_logger.Prec.global_avg,
        metric_logger.Rec.global_avg,
        metric_logger.F1score.global_avg
    ))
    
    return metric_logger.F1score.global_avg
    
def train(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    # print(dataloader)

    for batch in tqdm(dataloader):
        # Unpack batch contents
        imgs, labels = batch  # imgs is a list containing [img_t0, img_t1]
        # print(labels.shape)
        
        # print("imgs length",len(imgs))
        img_t0_batch = []
        img_t1_batch = []
        
        # Process each image pair in imgs
        for img in imgs:
            # Split the channels: [3, 512, 512] for each of img_t0 and img_t1
            img_t0, img_t1 = torch.split(img, 3, 0)  # Split along the channel dimension (0)
            # print(img_t0.shape,img_t1.shape)
            # Add to lists
            img_t0_batch.append(img_t0)
            img_t1_batch.append(img_t1)

        # first_img_t0 = img_t0_batch[0]
        # first_img_t1 = img_t1_batch[0]
        # first_label = labels[0, 0] 
        # visualize(first_img_t0,first_img_t1,first_label)
        # exit()
        # Stack lists into a batch with shape [batch_size, channels, height, width]
        img_t0_batch = torch.stack(img_t0_batch).to(device)
        img_t1_batch = torch.stack(img_t1_batch).to(device)
        labels = labels.to(device)  # Move labels to device if needed

        optimizer.zero_grad()
        # Now img_t0_batch and img_t1_batch are ready for the model
        outputs = model(img_t0_batch, img_t1_batch)
        # print("output shape",outputs.shape)  
        # Calculate loss
        loss = criterion(outputs, labels[:,0])
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update running loss for reporting
        running_loss += loss.item() * img_t0.size(0)
        
    
    # Average loss over all batches
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and move it to the device
    model = ChangeDetectionNet(backbone_output_dim=384).to(device)
    
    train_dataset = dataset_dict["VL_CMU_CD"](args, train=True)
    test_dataset = dataset_dict["VL_CMU_CD"](args, train=False)
    
    # Load the dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )
    
    # Define weighted softmax cross-entropy loss function
    weights = torch.tensor([0.025, 0.975], device=device)  # Weight for "no change" and "change"
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Set up the optimizer and cosine learning rate scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max set to match number of epochs
    
    best = -1
    start_time = time.time()
    
    # Number of epochs for training
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training step
        epoch_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Evaluation step
        f1score = CD_evaluate(model, test_loader, device=device)
        print(f"Epoch {epoch + 1}, Test F1 Score: {f1score:.4f}")
        
        # Step the scheduler
        scheduler.step() 
        
        # Save the best model based on F1 score
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }

        if f1score > best:
            best = f1score
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'best.pth')
            )
            print(f"New best model saved with F1 score: {best:.4f}")

        # Save the latest checkpoint
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth')
        )
    
    # Save final model outputs and metrics
    save_imgs_dir = os.path.join(args.output_dir, 'img')
    os.makedirs(save_imgs_dir, exist_ok=True)
    _ = CD_evaluate(model, test_loader, device=device, save_imgs_dir=save_imgs_dir)
    utils.save_on_master(
        checkpoint,
        os.path.join(args.output_dir, 'model_{}.pth'.format(epoch))
    )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch change detection', add_help=add_help)
    parser.add_argument('--train-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset2', default='', help='dataset name')
    parser.add_argument('--input-size', default=448, type=int, metavar='N',
                        help='the input-size of images')
    parser.add_argument('--randomflip', default=0.5, type=float, help='random flip input')
    parser.add_argument('--randomrotate', dest="randomrotate", action="store_true", help='random rotate input')
    parser.add_argument('--randomcrop', dest="randomcrop", action="store_true", help='random crop input')
    parser.add_argument('--data-cv', default=0, type=int, metavar='N',
                        help='the number of cross validation')

    parser.add_argument('--model', default='resnet18_mtf_msf_deeplabv3', help='model')
    parser.add_argument('--mtf', default='iade', help='choose branches to use')
    parser.add_argument('--msf', default=4, type=int, help='the number of MSF layers')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--loss', default='bi', type=str, help='the training loss')
    parser.add_argument('--loss-weight', action="store_true", help='add weight for loss')
    parser.add_argument('--opt', default='adam', type=str, help='the optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='the lr scheduler')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--warmup', dest="warmup", action="store_true", help='warmup the lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--pretrained", default='', help='pretrain checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval-every', default=1, type=int, metavar='N',
                        help='eval the model every n epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--save-imgs", dest="save_imgs", action="store_true",
                        help="save the predicted mask")

    
    parser.add_argument("--save-local", dest="save_local", help="save logs to local", action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    output_dir = 'output'
    save_path = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    args.output_dir = os.path.join(output_dir, save_path)
    print("output_dir",args.output_dir)
    main(args)
