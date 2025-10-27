# imports and environment setup
import pandas as pd
from tensorflow.io import gfile # type: ignore
import torch, timm
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2

_ = torch.set_grad_enabled(False)

# common variables: device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else
    "cpu"
))


# load_model_timm, load_dataset
def load_model_timm(model_type, dataset_name, verbose=False):
    """ 
    model   types: B/16, S/16 or Ti/16
    dataset names: cifar100 or oxford-iiit-pet
    """
    index = pd.read_csv('models/index.csv')
    pretrains = set(
        index.query('ds=="i21k"').groupby('name').apply(
        lambda df: df.sort_values('final_val').iloc[-1], 
        include_groups=False).filename
    )
    finetunes = index.loc[index.filename.apply(lambda name: name in pretrains)]
    checkpoint = (
        finetunes.query(f'name=="{model_type}" and adapt_ds=="{dataset_name}"')
        .sort_values('adapt_final_val').iloc[-1].adapt_filename
    ) # Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--cifar100-steps_10k-lr_0.003-res_224
    if verbose: print(f"Loaded checkpoint: {checkpoint}")
    
    timm_modelnames = {
        'Ti/16-224': 'vit_tiny_patch16_224',
        'Ti/16-384': 'vit_tiny_patch16_384',
        'S/16-224': 'vit_small_patch16_224',
        'S/16-384': 'vit_small_patch16_384',
        'B/16-224': 'vit_base_patch16_224',
        'B/16-384': 'vit_base_patch16_384'
    }
    num_classes = 100 if dataset_name == 'cifar100' else 37
    res = int(checkpoint.split('_')[-1])
    model = timm.create_model(timm_modelnames[f'{model_type}-{res}'], num_classes=num_classes)
    
    # downloading a checkpoint automatically
    # may show an error, but still downloads the checkpoint
    if not gfile.exists(f'models/{checkpoint}.npz'):     
        gfile.copy(f'gs://vit_models/augreg/{checkpoint}.npz', f'models/{checkpoint}.npz')
    timm.models.load_checkpoint(model, f'models/{checkpoint}.npz')

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_dataset(dataset_name, batch_size, model_cfg=None, subset_size = 1., res = (224, 224), train=False, download_dataset=False):
    """
    dataset name: cifar100 or oxford-iiit-pet
    """
    dataset = (
        datasets.CIFAR100('data/', train=train, download=download_dataset) if dataset_name == 'cifar100' 
        else datasets.OxfordIIITPet('data/', split=('trainval' if train else 'test'))
    ) 
    if model_cfg is None:
        m, s = [0.5]*3, [0.5]*3
    else:
        m, s = model_cfg['mean'], model_cfg['std']
        res = model_cfg['input_size'][-2:]
    dataset.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=m, std=s),
            v2.Resize(res),
    ])
    if subset_size < 1.0:
        n = len(dataset)
        n_small = int(subset_size * n)
        dataset, _ = random_split(dataset, [n_small, n - n_small])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataset, dataloader