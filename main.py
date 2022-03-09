import os
os.environ["WAND_DISABLE"] = "true"

from _modules.sed_raw import SEClassifier
from _modules.data import US8k, MAVD, GRN
from _modules.phinet_pl.phinet import PhiNet

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torchinfo import summary
from torch.nn import Module
from pathlib import Path
from yaml import load, dump
import numpy as np
import click

def get_backbone(model_name: str) -> Module:
    model_dict = {
        "phinet": PhiNet,
    }
    
    return model_dict[model_name]

@click.command()
@click.argument('dataset', type=str)
@click.argument('config_dir', type=str)
@click.argument('waveform', type=bool)
@click.option('--augmentation','-a',is_flag=True)
def main(config_dir, dataset, waveform, augmentation):
    print(config_dir, dataset, waveform, augmentation)
    config_dir = Path(config_dir)
    confs = sorted(config_dir.iterdir())

    if dataset == 'UrbanSound8K':
        data_module = US8k(
            waveform=waveform,
            debug=False,
            augmentation=augmentation,
            batch_size=64,
        )
        
    elif dataset == 'MAVD':
        data_module = MAVD(
            waveform = False,
            debug = False,
            augmentation=augmentation,
            batch_size =64,
        )
    else:
        data_module = GRN(
            waveform = False,
            debug = False,
            augmentation=augmentation,
            batch_size =64,
        )
        
    
    
    for c in confs:
        with open(c, "r") as read:
            net_yaml = load(read)
            
        print(str(c), dict(net_yaml))
        
        cfg_perf = {
            "name": str(c),
        }
        fold_perf = []
        if dataset == 'UrbanSound8K' or dataset == 'GRN':

            test_acc = []
            for testfold_index in range(1):
                #testfold_index = 9
                print(f"Using test fold: {testfold_index}")
            
                data_module.shuffle_folds(testfold_index)
                
                mod = SEClassifier(cfgs=net_yaml["network_config"], waveform=waveform)
            
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    monitor='val/loss',
                    dirpath=f'./ckp/{c.name}/',
                    filename='models-{epoch:02d}-{val/loss:.2f}',
                    save_top_k=3,
                    mode='min'
                )
                
                early_stopping_cb = pl.callbacks.early_stopping.EarlyStopping(
                    monitor="val/loss"
                )
            
                # wandb_logger = WandbLogger()
                trainer = pl.Trainer(
                    gpus=1,
                    max_epochs=200,
                    callbacks=[checkpoint_callback, early_stopping_cb],
                    # weights_summary=None,
                    # logger=wandb_logger
                )

                trainer.fit(model=mod, datamodule=data_module)
                valid_acc = trainer.validate(ckpt_path='best',datamodule=data_module,verbose=0)
                test_acc.append(trainer.test(ckpt_path='best',datamodule=data_module,verbose=0)[0]["test/acc"])
                

                fold_perf.append({"test_id": testfold_index, "acc test": test_acc[0], "acc valid": valid_acc[0]})
            print(f"-------------------------\nTest acc = {np.mean(test_acc)}, {np.std(test_acc)}\n-------------------------")
            cfg_perf["perf"] = fold_perf
        else:
            mod = SEClassifier(cfgs=net_yaml["network_config"])
            
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor='val/loss',
                dirpath=f'./ckp/{c.name}/',
                filename='models-{epoch:02d}-{val/loss:.2f}',
                save_top_k=3,
                mode='min'
            )
                
            early_stopping_cb = pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss"
            )
            
            # wandb_logger = WandbLogger()
            trainer = pl.Trainer(
                gpus=1,
                max_epochs=100,
                callbacks=[checkpoint_callback, early_stopping_cb],
                # weights_summary=None,
                # logger=wandb_logger
            )

            trainer.fit(model=mod, datamodule=data_module)
            test_acc = trainer.test(ckpt_path='best',datamodule=data_module,verbose=1)
            valid_acc = trainer.validate(ckpt_path='best',datamodule=data_module,verbose=1)
            fold_perf.append({'test acc': test_acc[0], 'valid acc': valid_acc[0]})
            cfg_perf["perf"]=fold_perf


        if not os.path.isdir("results/"+dataset):
            os.makedirs("results/"+dataset)
        with open(Path("results/"+dataset).joinpath(c.name), 'w') as f:
            data = dump(cfg_perf, f)
    
if __name__ == "__main__":
    main()
