"""
graphgps/loader/dataset/jslibs_loader.py

Registers the JSLibs dataset under the 'PyG-JSLibs' format string
so GraphGym's dataset factory can find it via cfg.dataset.format.

Drop this file in graphgps/loader/dataset/ alongside jslibs.py.
No other changes to the GraphGPS loader infrastructure needed.
"""

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader

from graphgps.loader.dataset.jslibs import JSLibsDataset


@register_loader('PyG-JSLibs')
def load_jslibs_dataset(format, name, dataset_dir):
    """
    Factory called by GraphGym when cfg.dataset.format == 'PyG-JSLibs'.

    Reads task mode from cfg.dataset.task_type:
        'classification'          → multiclass  (default, cross-entropy)
        'classification_multilabel' → multilabel (BCE per class)

    Returns a JSLibsDataset instance.  GraphGym will then call
    dataset.get_idx_split() to obtain train/val/test indices.
    """
    task_type = getattr(cfg.dataset, 'task_type', 'classification')
    task = 'multilabel' if task_type == 'classification_multilabel' else 'multiclass'

    dataset = JSLibsDataset(
        root                   = dataset_dir,
        task                   = task,
        min_nodes              = getattr(cfg.dataset, 'min_nodes', 5),
        max_nodes              = getattr(cfg.dataset, 'max_nodes', 2000),
        max_graphs_per_bundler = getattr(cfg.dataset, 'max_graphs_per_bundler', None),
    )

    return dataset
