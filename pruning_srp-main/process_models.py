import torch

_ = torch.set_grad_enabled(False)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else
    "cpu"
))

def load_model_timm(model_type, dataset_name, top10_idx = 1, verbose=False):
    """ 
    model_type: 'B/16', 'S/16', 'Ti/16'
    dataset names: cifar100 or oxford-iiit-pet
    """
    import pandas as pd, timm
    from tensorflow.io import gfile # type:ignore

    index = pd.read_csv('models/index.csv')
    pretrains = set(
        index.query('ds=="i21k"').groupby('name').apply(
        lambda df: df.sort_values('final_val').iloc[-1], 
        include_groups=False).filename
    )
    finetunes = index.loc[index.filename.apply(lambda name: name in pretrains)]
    checkpoint = list(
        finetunes.query(f'name=="{model_type}" and adapt_ds=="{dataset_name}"')
        .sort_values('adapt_final_val')['adapt_filename']
    )[-top10_idx] 
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