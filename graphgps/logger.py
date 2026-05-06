import logging
import time

import numpy as np
import torch
from scipy.stats import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix
from sklearn.metrics import r2_score
from torch_geometric.graphgym import get_current_gpu_usage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import infer_task, Logger
from torch_geometric.graphgym.utils.io import dict_to_json, dict_to_tb
from torchmetrics.functional import auroc

import graphgps.metrics_ogb as metrics_ogb
from graphgps.metric_wrapper import MetricWrapper


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


def accuracy_JS(targets, pred_int):
    """Balanced per-class accuracy for JSLibs dataset with class imbalance.

    For each class r:
        pr[r] = correctly_predicted[r] / total_true[r]
    Final score = mean(pr) over ALL classes (including empty ones counted as 0),
    so a model that ignores small libs (chalk, ms) is penalised.

    Identical contract to accuracy_SBM — same args, same return type.

    Args:
        targets  : torch.Tensor [N] long  — ground-truth class indices
        pred_int : torch.Tensor [N] long  — predicted class indices

    Returns:
        float in [0, 1]
    """
    # Convert to numpy FIRST — before any reassignment — so confusion_matrix
    # always receives arrays, not tensors (fixes the original bug in accuracy_SBM
    # where targets was overwritten after being passed to confusion_matrix).
    targets_np  = targets.cpu().detach().numpy()
    pred_int_np = pred_int.cpu().detach().numpy()

    CM         = confusion_matrix(targets_np, pred_int_np).astype(np.float32)
    nb_classes = CM.shape[0]

    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets_np == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
        # else: pr_classes[r] stays 0.0  (class absent in this split)

    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


def _load_label_map():
    """
    Load {idx: lib_name} from split.json for human-readable per-class logs.

    Optional — add to your yaml to enable:
        dataset:
          label_map_path: datasets/JSLibs/raw/split.json

    Returns dict or None if not configured / file missing.
    """
    try:
        path = cfg.dataset.label_map_path
    except AttributeError:
        return None
    if not path:
        return None
    try:
        import json
        with open(path) as f:
            lib_split = json.load(f)
        return {i: lib for i, lib in enumerate(sorted(lib_split.keys()))}
    except Exception as e:
        logging.warning("Could not load label_map from %s: %s", path, e)
        return None


def _log_per_class_breakdown(targets, pred_int, split_name, epoch,
                              label_map=None):
    """
    Print a per-lib accuracy bar chart to the Python logger.
    Called from classification_multi() when cfg.metric_best == 'accuracy-JS'.
    Output goes to the log file only — NOT stored in stats JSON or wandb.

    Example output:
        ──────────────────────────────────────────────────────
          Per-class accuracy  [val  epoch 12]
        ──────────────────────────────────────────────────────
          axios@1.7.9          ████████████████████░░░░░░░░░░  66.7%  (20/30)
          lodash@4.17.21       ████████░░░░░░░░░░░░░░░░░░░░░░  26.0%  (52/200)
          chalk@5.3.0          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0%  (0/8)
        ──────────────────────────────────────────────────────
    """
    targets_np  = targets.cpu().detach().numpy()
    pred_int_np = pred_int.cpu().detach().numpy()

    CM         = confusion_matrix(targets_np, pred_int_np).astype(np.float32)
    nb_classes = CM.shape[0]

    lines = [
        f"\n{'─' * 58}",
        f"  Per-class accuracy  [{split_name}  epoch {epoch}]",
        f"{'─' * 58}",
    ]

    for r in range(nb_classes):
        cluster = np.where(targets_np == r)[0]
        n_true  = cluster.shape[0]
        if n_true == 0:
            continue
        acc_r  = CM[r, r] / float(n_true)
        n_pred = int(CM[r, r])
        name   = label_map[r] if (label_map and r in label_map) else str(r)
        bar    = '█' * int(acc_r * 30) + '░' * (30 - int(acc_r * 30))
        lines.append(
            f"  {name:<28s}  {bar}  {acc_r:5.1%}  ({n_pred}/{n_true})"
        )

    lines.append(f"{'─' * 58}\n")
    logging.info('\n'.join(lines))


class CustomLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False

    # basic properties
    def basic(self):
        stats = {
            'loss': round(self._loss / self._size_current, max(8, cfg.round)),
            'lr': round(self._lr, max(8, cfg.round)),
            'params': self._params,
            'time_iter': round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats

    # task properties
    def classification_binary(self):
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        if true.shape[0] < 1e7:  # AUROC computation for very large datasets is too slow.
            # TorchMetrics AUROC on GPU if available.
            auroc_score = auroc(pred_score.to(torch.device(cfg.accelerator)),
                                true.to(torch.device(cfg.accelerator)),
                                task='binary')
            if self.test_scores:
                # SK-learn version.
                try:
                    r_a_score = roc_auc_score(true.cpu().numpy(),
                                              pred_score.cpu().numpy())
                except ValueError:
                    r_a_score = 0.0
                assert np.isclose(float(auroc_score), r_a_score)
        else:
            auroc_score = 0.

        reformat = lambda x: round(float(x), cfg.round)
        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'precision': reformat(precision_score(true, pred_int)),
            'recall': reformat(recall_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int)),
            'auc': reformat(auroc_score),
        }
        if cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))
        if cfg.metric_best == 'accuracy-JS':
            res['accuracy-JS'] = reformat(accuracy_JS(true, pred_int))
        return res

    def classification_multi(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        reformat = lambda x: round(float(x), cfg.round)

        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int,
                                    average='macro', zero_division=0)),
        }
        if cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))

        # ---- JSLibs: balanced accuracy + per-class breakdown ----
        if cfg.metric_best == 'accuracy-JS':
            res['accuracy-JS'] = reformat(accuracy_JS(true, pred_int))
            _log_per_class_breakdown(
                targets    = true,
                pred_int   = pred_int,
                split_name = self.name,
                epoch      = getattr(self, '_epoch', -1),
                label_map  = _load_label_map(),
            )

        if true.shape[0] < 1e7:
            # AUROC computation for very large datasets runs out of memory.
            # TorchMetrics AUROC on GPU is much faster than sklearn for large ds
            res['auc'] = reformat(auroc(pred_score.to(torch.device(cfg.accelerator)),
                                        true.to(torch.device(cfg.accelerator)).squeeze(),
                                        task='multiclass',
                                        num_classes=pred_score.shape[1],
                                        average='macro'))

            if self.test_scores:
                # SK-learn version.
                sk_auc = reformat(roc_auc_score(true, pred_score.exp(),
                                                average='macro',
                                                multi_class='ovr'))
                assert np.isclose(sk_auc, res['auc'])

        return res

    def classification_multilabel(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)

        # MetricWrapper will remove NaNs and apply the metric to each target dim
        acc = MetricWrapper(metric='accuracy',
                            target_nan_mask='ignore-mean-label',
                            task='binary',
                            cast_to_int=True)
        auroc = MetricWrapper(metric='auroc',
                              target_nan_mask='ignore-mean-label',
                              task='binary',
                              cast_to_int=True)
        ogb_ap = reformat(metrics_ogb.eval_ap(true.cpu().numpy(),
                                              pred_score.cpu().numpy())['ap'])
        # Send to GPU to speed up TorchMetrics if possible.
        true = true.to(torch.device(cfg.accelerator))
        pred_score = pred_score.to(torch.device(cfg.accelerator))
        results = {
            'accuracy': reformat(acc(torch.sigmoid(pred_score), true)),
            'auc': reformat(auroc(pred_score, true)),
            'ap': ogb_ap,
        }

        if self.test_scores:
            true = true.cpu().numpy()
            pred_score = pred_score.cpu().numpy()
            ogb = {
                'accuracy': reformat(metrics_ogb.eval_acc(
                    true, (pred_score > 0.).astype(int))['acc']),
                'ap': reformat(metrics_ogb.eval_ap(true, pred_score)['ap']),
                'auc': reformat(
                    metrics_ogb.eval_rocauc(true, pred_score)['rocauc']),
            }
            assert np.isclose(ogb['accuracy'], results['accuracy'], atol=1e-05)
            assert np.isclose(ogb['ap'], results['ap'], atol=1e-05)
            assert np.isclose(ogb['auc'], results['auc'], atol=1e-05)

        return results

    def subtoken_prediction(self):
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator('ogbg-code2')

        seq_ref_list = []
        seq_pred_list = []
        for seq_pred, seq_ref in zip(self._pred, self._true):
            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

        input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
        result = evaluator.eval(input_dict)
        result['f1'] = result['F1']
        del result['F1']
        return result

    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)
        return {
            'mae': reformat(mean_absolute_error(true, pred)),
            'r2': reformat(r2_score(true, pred, multioutput='uniform_average')),
            'spearmanr': reformat(eval_spearmanr(true.numpy(),
                                                 pred.numpy())['spearmanr']),
            'mse': reformat(mean_squared_error(true, pred)),
            'rmse': reformat(mean_squared_error(true, pred, squared=False)),
        }

    def update_stats(self, true, pred, loss, lr, time_used, params,
                     dataset_name=None, **kwargs):
        if dataset_name == 'ogbg-code2':
            assert true['y_arr'].shape[1] == len(pred)  # max_seq_len (5)
            assert true['y_arr'].shape[0] == pred[0].shape[0]  # batch size
            batch_size = true['y_arr'].shape[0]

            from graphgps.loader.ogbg_code2_utils import idx2vocab, \
                decode_arr_to_seq
            arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)
            mat = []
            for i in range(len(pred)):
                mat.append(torch.argmax(pred[i].detach(), dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)
            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [true['y'][i] for i in range(len(true['y']))]
            pred = seq_pred
            true = seq_ref
        else:
            assert true.shape[0] == pred.shape[0]
            batch_size = true.shape[0]

        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size

    def write_epoch(self, cur_epoch):
        start_time = time.perf_counter()
        basic_stats = self.basic()

        # store so classification_multi() can include epoch in the breakdown log
        self._epoch = cur_epoch

        if self.task_type == 'regression':
            task_stats = self.regression()
        elif self.task_type == 'classification_binary':
            task_stats = self.classification_binary()
        elif self.task_type == 'classification_multi':
            task_stats = self.classification_multi()
        elif self.task_type == 'classification_multilabel':
            task_stats = self.classification_multilabel()
        elif self.task_type == 'subtoken_prediction':
            task_stats = self.subtoken_prediction()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch,
                       'time_epoch': round(self._time_used, cfg.round)}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round),
                     'eta_hours': round(self.eta(cur_epoch) / 3600, cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()
        if cur_epoch < 3:
            logging.info(f"...computing epoch stats took: "
                         f"{time.perf_counter() - start_time:.2f}s")
        return stats


def create_logger():
    """
    Create logger for the experiment

    Returns: List of logger objects

    """
    loggers = []
    names = ['train', 'val', 'test']
    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(CustomLogger(name=names[i], task_type=infer_task()))
    return loggers


def eval_spearmanr(y_true, y_pred):
    """Compute Spearman Rho averaged across tasks.
    """
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(stats.spearmanr(y_true[is_labeled, i],
                                            y_pred[is_labeled, i])[0])

    return {'spearmanr': sum(res_list) / len(res_list)}
