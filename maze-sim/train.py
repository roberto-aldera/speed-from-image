import pytorch_lightning as pl
from pathlib import Path
import time
import settings
from argparse import ArgumentParser
from evaluate import do_quick_evaluation
from lenet import LeNet
from resnet import ResNet


# start_time = time.time()
#
# Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
# Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
#
# model = settings.MODEL
# trainer = pl.Trainer(default_root_dir=settings.MAZE_MODEL_DIR, max_epochs=10, auto_lr_find=True)
# trainer.fit(model)
# path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, ".ckpt")
# trainer.save_checkpoint(path_to_model)
#
# # trainer.test()
#
# print("Finished Training")
# print("--- Training execution time: %s seconds ---" % (time.time() - start_time))
#
# do_quick_evaluation(model_path=path_to_model)


def main(args):
    # pick model
    if args.model_name == 'lenet':
        model = LeNet(args)
    elif args.model_name == 'resnet':
        model = ResNet(args)

    # print("DATA PATH:", args.data_path)
    # args.default_root_dir = settings.MAZE_MODEL_DIR

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=settings.MAZE_MODEL_DIR)
    trainer.fit(model)

    # save checkpoint that will be used for running evaluation
    path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, ".ckpt")
    trainer.save_checkpoint(path_to_model)
    # trainer.test()


if __name__ == '__main__':
    start_time = time.time()
    parser = ArgumentParser(add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--model_name', type=str, default='lenet', help='lenet or resnet')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == 'lenet':
        parser = LeNet.add_model_specific_args(parser)
    elif temp_args.model_name == 'resnet':
        parser = ResNet.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)
    print("Finished Training")
    print("--- Training execution time: %s seconds ---" % (time.time() - start_time))
