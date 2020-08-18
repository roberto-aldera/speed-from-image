import pytorch_lightning as pl
from pathlib import Path
import time

import settings
from evaluate import do_quick_evaluation

start_time = time.time()

# Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
# Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)

# -> Lightning
pl.seed_everything(0)
model = settings.MODEL
trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=5)
trainer.fit(model)

# trainer.test()
###

print("Finished Training")
print("--- Training execution time: %s seconds ---" % (time.time() - start_time))

# path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, settings.ARCHITECTURE_TYPE, "_checkpoint.pt")
# do_quick_evaluation(model_path=path_to_model)
