from torch.utils.data import DataLoader
import torch

def create_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, pin_memory=True):
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
        print("cuda is not available, setting pin_memory=False")
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_dataset
