# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import time
import glob

# Things needed to debug the Interaction class
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
from utils.interaction import Interaction


from utils.transforms import (
    AddGuidanceSignalDeepEditd,
    AddRandomGuidanceDeepEditd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    SplitPredsLabeld,
)
from monai.data import partition_dataset
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from utils.dynunet import DynUNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    DivisiblePadd,
    ToNumpyd,
    ToTensord,
)
from monai.utils import set_determinism


def get_network(network, labels, args):
    network = DynUNet(
        spatial_dims=3,
        in_channels=len(labels) + 1,
        out_channels=len(labels),
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
        conv1d=args.conv1d,
        conv1s=args.conv1s,
    )
    return network


# MSD Spleen
def get_pre_transforms(labels, device, args):
    spacing = [2 * 0.79296899, 2 * 0.79296899, 5.        ] # 2-factor because of the spatial size
    t_train = [
        LoadImaged(keys=("image", "label"), reader="ITKReader"),
        EnsureChannelFirstd(keys=("image", "label")),
        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing),
        ScaleIntensityRanged(keys="image", a_min=-224, a_max=212, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the spleen HUs
        DivisiblePadd(keys=["image", "label"], k=64, value=0), # Needed for DynUNet
        ### Random Transforms ###
        RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
        Resized(keys=("image", "label"), spatial_size=[256, 256, -1], mode=("area", "nearest")), # downsampled from 512x512x-1 to fit into memory
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        ToTensord(keys=("image", "guidance"), device=device),
        AddGuidanceSignalDeepEditd(keys="image",
                                    guidance="guidance",
                                    sigma=args.sigma,
                                    disks=args.disks,
                                    edt=args.edt,
                                    gdt=args.gdt,
                                    gdt_th=args.gdt_th,
                                    exp_geos=args.exp_geos,
                                    adaptive_sigma=args.adaptive_sigma,
                                    device=device, spacing=spacing),
        ToTensord(keys=("image", "label"), device=torch.device('cpu')), # TODO: check why we need this on the CPU
    ]
    t_val = [
        LoadImaged(keys=("image", "label"), reader="ITKReader"),
        EnsureChannelFirstd(keys=("image", "label")),
        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing),
        ScaleIntensityRanged(keys="image", a_min=-224, a_max=212, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the spleen HUs
        DivisiblePadd(keys=["image", "label"], k=64, value=0), # Needed for DynUNet
        Resized(keys=("image", "label"), spatial_size=[256, 256, -1], mode=("area", "nearest")), # downsampled from 512x512x-1 to fit into memory
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        ToTensord(keys=("image", "guidance"), device=device),
        AddGuidanceSignalDeepEditd(keys="image",
                                    guidance="guidance",
                                    sigma=args.sigma,
                                    disks=args.disks,
                                    edt=args.edt,
                                    gdt=args.gdt,
                                    gdt_th=args.gdt_th,
                                    exp_geos=args.exp_geos,
                                    adaptive_sigma=args.adaptive_sigma,
                                    device=device, spacing=spacing),
        ToTensord(keys=("image", "label"), device=torch.device('cpu')),
    ]

    return Compose(t_train), Compose(t_val)


def get_click_transforms(device, args):
    spacing = [2 * 0.79296899, 2 * 0.79296899, 5.        ] # 2-factor because of the spatial size

    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        ToNumpyd(keys=("image", "label", "pred")),
        # Transforms for click simulation
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditd(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        ToTensord(keys=("image", "guidance"), device=device),
        AddGuidanceSignalDeepEditd(keys="image",
                                    guidance="guidance",
                                    sigma=args.sigma,
                                    disks=args.disks,
                                    edt=args.edt,
                                    gdt=args.gdt,
                                    gdt_th=args.gdt_th,
                                    exp_geos=args.exp_geos,
                                    adaptive_sigma=args.adaptive_sigma,
                                    device=device, spacing=spacing),        #
        ToTensord(keys=("image", "label"), device=torch.device('cpu')),
    ]

    return Compose(t)


def get_post_transforms(labels):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)


def get_loaders(args, pre_transforms_train, pre_transforms_val):
    all_images = sorted(glob.glob(os.path.join(args.input, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(args.input, "labelsTr", "*.nii.gz")))

    all_images = all_images
    all_labels = all_labels
    datalist = [{"image": image_name, "label": label_name} for image_name, label_name in
                zip(all_images, all_labels)]

    datalist = datalist[0: args.limit] if args.limit else datalist
    total_l = len(datalist)

    train_datalist, val_datalist = partition_dataset(
        datalist,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )

    train_ds = PersistentDataset(
        train_datalist, pre_transforms_train, cache_dir=args.cache_dir
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, num_workers=2, batch_size=1, multiprocessing_context='spawn',
    )
    logging.info(
        "{}:: Total Records used for Training is: {}/{}".format(
            args.gpu, len(train_ds), total_l
        )
    )

    val_ds = PersistentDataset(val_datalist, pre_transforms_val, cache_dir=args.cache_dir)

    val_loader = DataLoader(val_ds, num_workers=2, batch_size=1, multiprocessing_context='spawn')
    logging.info(
        "{}:: Total Records used for Validation is: {}/{}".format(
            args.gpu, len(val_ds), total_l
        )
    )

    return train_loader, val_loader


def create_trainer(args):

    set_determinism(seed=args.seed)

    device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_train, pre_transforms_val = get_pre_transforms(args.labels, device, args)
    click_transforms = get_click_transforms(device, args)
    post_transform = get_post_transforms(args.labels)

    train_loader, val_loader = get_loaders(args, pre_transforms_train, pre_transforms_val)

    # define training components
    network = get_network(args.network, args.labels, args).to(device)

    if args.resume:
        logging.info("{}:: Loading Network...".format(args.gpu))
        map_location = {f"cuda:{args.gpu}": f"cuda:{args.gpu}"}
        network.load_state_dict(
            torch.load(args.model_filepath, map_location=map_location)['net']
        )

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network},
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_" + args.network + ".pt",
        ),
    ]

    all_val_metrics = dict()
    all_val_metrics["val_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    for key_label in args.labels:
        if key_label != "background":
            all_val_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            click_probability_key="probability",
            train=False,
            label_names=args.labels,
            max_interactions=args.max_val_interactions,
            args=args,
        ),
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator, interval=args.val_freq, epoch_level=(not args.eval_only)
        ),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
        TensorBoardStatsHandler(
            log_dir=args.output,
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
            save_interval=args.save_interval * 2,
            save_final=True,
            final_filename="checkpoint.pt",
        ),
    ]

    all_train_metrics = dict()
    all_train_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]),
                                               include_background=False)
    for key_label in args.labels:
        if key_label != "background":
            all_train_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            click_probability_key="probability",
            train=True,
            label_names=args.labels,
            max_interactions=args.max_train_interactions,
            args=args,
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )
    return trainer


def run(args):
    for arg in vars(args):
        logging.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")

    if args.export:
        logging.info(
            "{}:: Loading PT Model from: {}".format(args.gpu, args.input)
        )
        device = torch.device(f"cuda:{args.gpu}")
        network = get_network(args.network, args.labels).to(device)

        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        network.load_state_dict(torch.load(args.input, map_location=map_location))

        logging.info("{}:: Saving TorchScript Model".format(args.gpu))
        model_ts = torch.jit.script(network)
        torch.jit.save(model_ts, os.path.join(args.output))
        return

    if not os.path.exists(args.output):
        logging.info(
            "output path [{}] does not exist. creating it now.".format(args.output)
        )
        os.makedirs(args.output, exist_ok=True)

    trainer = create_trainer(args)

    start_time = time.time()
    trainer.run()
    end_time = time.time()

    logging.info("Total Training Time {}".format(end_time - start_time))
    logging.info("{}:: Saving Final PT Model".format(args.gpu))
    torch.save(
        trainer.network.state_dict(), os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.pt")
    )

    logging.info("{}:: Saving TorchScript Model".format(args.gpu))
    model_ts = torch.jit.script(trainer.network)
    torch.jit.save(model_ts, os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.ts"))


def main():
    print(f"CPU Count: {os.cpu_count()}")
    torch.set_num_threads(int(os.cpu_count() / 3)) # Limit number of threads to 1/3 of resources
    print(f"Num threads: {torch.get_num_threads()}")
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input", default="../../tutorials/deepedit_zrrr/ignite/Task09_Spleen/")
    parser.add_argument("-o", "--output", default="output/test")
    parser.add_argument("--cache_dir", type=str, default='cache/test')
    parser.add_argument("-x", "--split", type=float, default=0.8)
    parser.add_argument("-t", "--limit", type=int, default=0, help='Limit the amount of training/validation samples')


    # Configuration
    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0)

    # Model
    parser.add_argument("-n", "--network", default="dynunet", choices=["dynunet", "unetr"])
    parser.add_argument("-r", "--resume", default=False, action='store_true')

    # Training
    parser.add_argument("-a", "--amp", default=False, action='store_true')
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--model_weights", type=str, default='None')
    parser.add_argument("--best_val_weights", default=False, action='store_true')

    # Logging
    parser.add_argument("-f", "--val_freq", type=int, default=1) # Epoch Level
    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--export", default=False, action='store_true')
    parser.add_argument("--eval_only", default=False, action='store_true')
    parser.add_argument("--save_nifti", default=False, action='store_true')

    # Interactions
    parser.add_argument("-it", "--max_train_interactions", type=int, default=10)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=10)
    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=1.0)
    parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)

    # Guidance Signal Hyperparameters
    parser.add_argument("--sigma", type=int, default=3)
    parser.add_argument("--disks", default=False, action='store_true')
    parser.add_argument("--edt", default=False, action='store_true')
    parser.add_argument("--gdt", default=False, action='store_true')
    parser.add_argument("--gdt_th", type=float, default=10)
    parser.add_argument("--exp_geos", default=False, action='store_true')
    parser.add_argument("--conv1d", default=False, action='store_true')
    parser.add_argument("--conv1s", default=False, action='store_true')
    parser.add_argument("--adaptive_sigma", default=False, action='store_true')


    args = parser.parse_args()
    # For single label using one of the Medical Segmentation Decathlon
    args.labels = {'spleen': 1,
                   'background': 0
                   }

    # Restoring previous model if resume flag is True
    args.model_filepath = args.model_weights
    if args.best_val_weights:
        args.model_filepath = os.path.join(args.output, sorted([el for el in os.listdir(args.output) if 'net_key' in el])[-1])
        current_epoch = sorted([int(el.split('.')[0].split('=')[1]) for el in os.listdir(args.output) if 'net_epoch' in el])[-1]
        args.epochs = args.epochs - current_epoch # Reset epochs based on previous model

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)
    run(args)


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
