from data.data_loader_kvasir_v1 import get_data_loader as kvasir_v1
from data.data_loader_hyper_kvasir import get_data_loader as hyper_kvasir
from data.data_loader_Gastrovision import get_data_loader as gastrovision
from data.data_loader_Kvasir_Capsule import get_data_loader as kvasir_capsule
from data.data_loader_WCEBleedGen import get_data_loader as wcebleed

DATASET_LOADERS = {
    "kvasir_v1": kvasir_v1,
    "kvasir_v2": kvasir_v2,
    "hyper_kvasir": hyper_kvasir,
    "gastrovision": gastrovision,
    "kvasir_capsule": kvasir_capsule,
    "wcebleed": wcebleed,
}

def get_dataloaders(dataset_name, data_dir, batch_size):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available datasets: {list(DATASET_LOADERS.keys())}"
        )

    return DATASET_LOADERS[dataset_name](data_dir, batch_size)