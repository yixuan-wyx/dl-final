"""
"""
import os
import math
import inspect
import torch
import torch.nn as nn
import datasets

from typing import Optional
import torch.utils
from torch.utils.data import DataLoader, Dataset, RandomSampler
from typing import Tuple, Union, Dict, Mapping, Any, Callable

from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

# helper from transformers
from transformers.trainer_utils import seed_worker, RemoveColumnsCollator
from transformers.trainer_pt_utils import IterableDatasetShard

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(object):
    """
    Base tranier for just trial
    """
    def __init__(
                self, 
                model:nn.Module, 
                train_dataset: Optional[Dataset] = None,
                eval_dataset: Optional[Dataset] = None,
                tokenizer = None,
                optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                data_collector = None, 
                args = None,
                logger = None
        ):
            
        self.model = model
        self.args = args

        # dataset
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collector = data_collector
        self.tokenizer = tokenizer

        # args
        self.gradient_accumulation_steps = 1
        self.batch_size = args.batch_size
        self.dataloader_num_workers = args.num_workers
        self.dataloader_pin_memory = True
        self.dataloader_persistent_workers = True
        self.split_batches = False
        self.dataloader_drop_last = False
        self.num_train_epochs = args.num_train_epochs
        self.remove_unused_columns = True
        self.past_index = -1
        self.logger = logger
        self.is_deepspeed_enabled = False

        # run_dir
        self.run_dir = self.args.run_dir

        # device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # initialize accelerator
        self.initialize_accelerator()

        self._signature_columns = None
        default_label_names = ['labels']
        self.label_names = default_label_names

        # optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        

    def initialize_accelerator(self):
        grad_acc_kwargs = {"num_steps": self.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        self.accelerator = Accelerator(
            dispatch_batches=None,
            split_batches=self.split_batches,
            deepspeed_plugin=None,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

    def initialize_train_sampler(self):
        # TODO: Support different sampling strategy
        return RandomSampler(self.train_dataset)
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            print(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=self.logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collector

        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self.batch_size,
            "collate_fn": data_collator,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.dataloader_pin_memory,
            "persistent_workers": self.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self.initialize_train_sampler()
            dataloader_params["drop_last"] = self.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        
        trainloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return trainloader

    def get_data_size(self, dataloader: DataLoader):
        try:
            dataset = dataloader.dataset
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except:
            return len(dataloader) * self.batch_size

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
    
    def train(self):
        trainloader = self.get_train_dataloader()
        
        if len(trainloader) is not None:
            len_dataloader = len(trainloader)
            num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.get_data_size(trainloader)

            max_steps = math.ceil(self.num_train_epochs * num_update_steps_per_epoch)
            num_train_samples = self.get_data_size(trainloader) * self.num_train_epochs
        else:
            raise ValueError("Train loader cannot be None!")

        best_loss = 1e+10
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_steps, eta_min=1e-8)
        for epoch in range(self.num_train_epochs):
            self.accelerator.free_memory()

            loss_meter = AverageMeter()
            
            epoch_iterator = trainloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            for step, inputs in enumerate(epoch_iterator):
                inputs = self._prepare_inputs(inputs)
                output = self.model(**inputs)

                # fetch loss
                loss = output["loss"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                loss_meter.update(loss.item())

                if step % 200 == 0:
                    self.logger.info(f"Epoch [{epoch}] Step [{step}] / [{len(epoch_iterator)}] | loss = {loss.item():.3e}")
                    print("Save checkpoint...")
                    self.save_checkpoints(best_model=True)            

        print("Done!")

    def save_checkpoints(self, best_model:bool=False):
        if hasattr(self.model, "draft_dec"):
            model = self.model.draft_dec
            sdict = model.state_dict()
        else:
            sdict = self.model.state_dict()

        if best_model:
            modelfile = os.path.join(self.run_dir, "peft_model.pth.tar")
            torch.save(sdict, modelfile)