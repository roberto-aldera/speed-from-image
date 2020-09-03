import pytorch_lightning as pl
from pathlib import Path
import time
import settings
from argparse import ArgumentParser
from evaluate import do_quick_evaluation
from lenet import LeNet
from resnet import ResNet

if __name__ == '__main__':
    start_time = time.time()

    Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default="lenet", help="lenet or resnet")
    parser.add_argument("--do_evaluation", type=str, default=False, help="Run evaluation script after training")

    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.model_name == "lenet":
        parser = LeNet.add_model_specific_args(parser)
    elif temp_args.model_name == "resnet":
        parser = ResNet.add_model_specific_args(parser)

    hparams = parser.parse_args()

    # pick model
    model = None
    if hparams.model_name == "lenet":
        model = LeNet(hparams)
    elif hparams.model_name == "resnet":
        model = ResNet(hparams)

    # trainer = pl.Trainer.from_argparse_args(hparams,
    #                                         default_root_dir=settings.MAZE_MODEL_DIR,
    #                                         max_epochs=hparams.max_num_epochs,
    #                                         resume_from_checkpoint='/workspace/data/speed-from-image/maze-models/version_0.1/checkpoints/epoch=28.ckpt')

    trainer = pl.Trainer.from_argparse_args(hparams,
                                            default_root_dir=settings.MAZE_MODEL_DIR,
                                            max_epochs=hparams.max_num_epochs)
    trainer.fit(model)

    # save checkpoint that will be used for running evaluation
    path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, hparams.model_name, ".ckpt")
    trainer.save_checkpoint(path_to_model)

    print("Finished Training")
    print("--- Training execution time: %s seconds ---" % (time.time() - start_time))

    if hparams.do_evaluation:
        do_quick_evaluation(hparams, model, path_to_model)
    # trainer.test()
