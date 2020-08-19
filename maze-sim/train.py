import pytorch_lightning as pl
from pathlib import Path
import time
import settings
from evaluate import do_quick_evaluation

start_time = time.time()

Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)

model = settings.MODEL
trainer = pl.Trainer(default_root_dir=settings.MAZE_MODEL_DIR, max_epochs=10)
trainer.fit(model)
path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, ".ckpt")
trainer.save_checkpoint(path_to_model)

# trainer.test()

print("Finished Training")
print("--- Training execution time: %s seconds ---" % (time.time() - start_time))

do_quick_evaluation(model_path=path_to_model)
