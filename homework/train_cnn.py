from models import CNNClassifier, save_model
from utils import load_data
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.optim as optim


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.n_epochs

    # Paths to data
    ubuntu_train_path = r"/home/william/Desktop/UT_Austin_Computer_Vision/UTAustin_hw3/data/train"
    ubuntu_valid_path = r"/home/william/Desktop/UT_Austin_Computer_Vision/UTAustin_hw3/data/valid"

    train_loader = load_data(dataset_path=ubuntu_train_path, num_workers=4, batch_size=batch_size)
    valid_loader = load_data(dataset_path=ubuntu_valid_path, num_workers=4, batch_size=batch_size)

    model = CNNClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    global_steps = 0
    epoch_loss = [100]
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_total_samples = 0.0
        train_total_correct = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_logger.add_scalar('train', loss, global_steps)
            loss.backward()
            optimizer.step()
            global_steps += 1
            train_loss += loss.item()

            # Get accuracy
            _, predicted = torch.max(outputs, 1)
            train_total_samples += labels.size(0)
            train_total_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")
        print(f"Train Accuracy: {train_total_correct / train_total_samples}")

        # Validation
        model.eval()
        val_loss = 0.0
        valid_total_correct = 0.0
        valid_total_samples = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                # Get accuracy
                _, predicted = torch.max(outputs, 1)
                valid_total_samples += labels.size(0)
                valid_total_correct += (predicted == labels).sum().item()

        valid_logger.add_scalar('valid', val_loss/len(valid_loader), epoch)
        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")
        print(f"Validation Accuracy: {valid_total_correct/valid_total_samples}")

        # Save if better than previous models
        if val_loss/len(valid_loader) < sorted(epoch_loss)[0]:
            save_model(model)
        epoch_loss.append(val_loss/len(valid_loader))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()
    train(args)
