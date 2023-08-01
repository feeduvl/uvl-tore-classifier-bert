
# %%
import logging
import shutil
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
import mlflow
from src.experiments.sner import sner
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)
logger = logging.getLogger("training")

# %%
sentence_length = 110

# %%
from tooling.config import Experiment, Transformation

base_experiment_config = Experiment(
    name="Base Config", iterations=1, force=False
)

levels_transformation_config = Transformation(
    description="Levels",
    type="Reduced",
    task="Domain_Level",
    domain_data="Domain_Level",
    activity="Activity",
    stakeholder="Domain_Level",
    system_function="Interaction_Level",
    interaction="Interaction_Level",
    interaction_data="Interaction_Level",
    workspace="Interaction_Level",
    software="System_Level",
    internal_action="System_Level",
    internal_data="System_Level",
)

label_transformation_config = Transformation(
    description="None",
    type="Full",
    task="Task",
    domain_data="Domain_Data",
    activity="Activity",
    stakeholder="Stakeholder",
    system_function="System_Function",
    interaction="Interaction",
    interaction_data="Interaction_Data",
    workspace="Workspace",
    software="System_Level",
    internal_action="System_Level",
    internal_data="System_Level",
)

# %%
from classifiers.staged_bert.classifier import get_hint_transformation
import pickle

hint_transformation = get_hint_transformation(
    transformation_cfg=OmegaConf.structured(levels_transformation_config)
)
hint_label2id = hint_transformation["label2id"]
pickle.dump(
    hint_label2id, open("./src/service/models/hint_label2id.pickle", "wb")
)
hint_id2label = {y: x for x, y in hint_label2id.items()}
pickle.dump(
    hint_id2label, open("./src/service/models/hint_id2label.pickle", "wb")
)

transformation = get_hint_transformation(
    transformation_cfg=OmegaConf.structured(label_transformation_config)
)
label2id = transformation["label2id"]
pickle.dump(label2id, open("./src/service/models/label2id.pickle", "wb"))
id2label = {y: x for x, y in label2id.items()}
pickle.dump(id2label, open("./src/service/models/id2label.pickle", "wb"))

# %% [markdown]
# ## Train BiLSTM First Stage Model

# %%
from tooling.observability import get_run_id
from tooling.config import BiLSTMConfig, BiLSTM
from experiments.bilstm import bilstm
from copy import deepcopy

bilstm_experiment_config = deepcopy(base_experiment_config)
bilstm_experiment_config.name = "BiLSTM Train for Staged Application"

bilstm_config = BiLSTM(sentence_length=sentence_length)

bilstm_cfg = OmegaConf.structured(
    BiLSTMConfig(
        bilstm=bilstm_config,
        experiment=bilstm_experiment_config,
        transformation=levels_transformation_config,
    )
)

bilstm(bilstm_cfg)

bilstm_run_id = get_run_id(bilstm_cfg)

print(bilstm_run_id)

bilstm_run = mlflow.get_run(bilstm_run_id)
mlflow.artifacts.download_artifacts(
    f"{bilstm_run.info.artifact_uri}/0_model",
    dst_path=Path("./src/service/models/"),
)
try:
    shutil.rmtree(Path("./src/service/models/bilstm"))
except FileNotFoundError:
    pass
Path("./src/service/models/0_model").rename("./src/service/models/bilstm")

# %% [markdown]
# ## Train SNER First Stage Model

# %%
from tooling.observability import get_run_id
from tooling.config import SNERConfig, SNER
from experiments.sner import sner
from copy import deepcopy

sner_experiment_config = deepcopy(base_experiment_config)
sner_experiment_config.name = "SNER Train for Staged Application"

sner_config = SNER()

sner_cfg = OmegaConf.structured(
    SNERConfig(
        sner=sner_config,
        experiment=sner_experiment_config,
        transformation=levels_transformation_config,
    )
)

sner(OmegaConf.create(sner_cfg))

sner_run_id = get_run_id(sner_cfg)

print(sner_run_id)

sner_run = mlflow.get_run(sner_run_id)
mlflow.artifacts.download_artifacts(
    f"{sner_run.info.artifact_uri}/0_model.ser.gz",
    dst_path=Path("./src/service/models/"),
)
try:
    Path("./src/service/models/sner.ser.gz").unlink()
except FileNotFoundError:
    pass
Path("./src/service/models/0_model.ser.gz").rename(
    "./src/service/models/sner.ser.gz"
)

# %% [markdown]
# ## Train BERT First Stage Model

# %%
from tooling.observability import get_run_id
from tooling.config import BERTConfig, BERT
from experiments.bert import bert
from copy import deepcopy

bert_1_experiment_config = deepcopy(base_experiment_config)
bert_1_experiment_config.name = "BERT_1 Train for Staged Application"

bert_1_config = BERT(max_len=sentence_length)

bert_1_cfg = OmegaConf.structured(
    BERTConfig(
        bert=bert_1_config,
        experiment=bert_1_experiment_config,
        transformation=levels_transformation_config,
    )
)

bert(OmegaConf.create(bert_1_cfg))

bert_1_run_id = get_run_id(bert_1_cfg)

print(bert_1_run_id)

bert_1_run = mlflow.get_run(bert_1_run_id)
mlflow.artifacts.download_artifacts(
    f"{bert_1_run.info.artifact_uri}/0_model",
    dst_path=Path("./src/service/models/"),
)
try:
    shutil.rmtree(Path("./src/service/models/bert_1"))
except FileNotFoundError:
    pass
Path("./src/service/models/0_model").rename("./src/service/models/bert_1")

# %% [markdown]
# ## Train BERT Second Stage Model for BERT First Stage

# %%
from tooling.observability import get_run_id
from tooling.config import DualModelStagedBERTConfig, StagedBERT
from experiments.dual_model_staged_bert import dual_stage_bert
from copy import deepcopy

bert_2_bert_experiment_config = deepcopy(base_experiment_config)
bert_2_bert_experiment_config.name = "BERT_2 Train for Staged Application"

bert_2_bert_config = StagedBERT(max_len=sentence_length)

bert_2_bert_cfg = OmegaConf.structured(
    DualModelStagedBERTConfig(
        bert=bert_2_bert_config,
        experiment=bert_2_bert_experiment_config,
        transformation=label_transformation_config,
        first_model_bert=bert_1_cfg,
    )
)

dual_stage_bert(OmegaConf.create(bert_2_bert_cfg))

bert_2_bert_run_id = get_run_id(bert_2_bert_cfg)

print(bert_2_bert_run_id)

bert_2_bert_run = mlflow.get_run(bert_2_bert_run_id)
mlflow.artifacts.download_artifacts(
    f"{bert_2_bert_run.info.artifact_uri}/0_model",
    dst_path=Path("./src/service/models/"),
)
try:
    shutil.rmtree(Path("./src/service/models/bert_2_bert"))
except FileNotFoundError:
    pass
Path("./src/service/models/0_model").rename("./src/service/models/bert_2_bert")

# %% [markdown]
# ## Train BERT Second Stage Model for SNER First Stage

# %%
from tooling.observability import get_run_id
from tooling.config import DualModelStagedBERTConfig, StagedBERT
from experiments.dual_model_staged_bert import dual_stage_bert
from copy import deepcopy

bert_2_sner_experiment_config = deepcopy(base_experiment_config)
bert_2_sner_experiment_config.name = "BERT_2 Train for Staged Application"

bert_2_sner_config = StagedBERT(max_len=sentence_length)

bert_2_sner_cfg = OmegaConf.structured(
    DualModelStagedBERTConfig(
        bert=bert_2_sner_config,
        experiment=bert_2_sner_experiment_config,
        transformation=label_transformation_config,
        first_model_sner=sner_cfg,
    )
)

dual_stage_bert(OmegaConf.create(bert_2_sner_cfg))

bert_2_sner_run_id = get_run_id(bert_2_sner_cfg)

print(bert_2_sner_run_id)

bert_2_sner_run = mlflow.get_run(bert_2_sner_run_id)
mlflow.artifacts.download_artifacts(
    f"{bert_2_sner_run.info.artifact_uri}/0_model",
    dst_path=Path("./src/service/models/"),
)
try:
    shutil.rmtree(Path("./src/service/models/bert_2_sner"))
except FileNotFoundError:
    pass

Path("./src/service/models/0_model").rename("./src/service/models/bert_2_sner")

# %% [markdown]
# ## Train BERT Second Stage Model for BILSTM First Stage

# %%
from tooling.observability import get_run_id
from tooling.config import DualModelStagedBERTConfig, StagedBERT
from experiments.dual_model_staged_bert import dual_stage_bert
from copy import deepcopy

bert_2_bilstm_experiment_config = deepcopy(base_experiment_config)
bert_2_bilstm_experiment_config.name = "BERT_2 Train for Staged Application"

bert_2_bilstm_config = StagedBERT(max_len=sentence_length)

bert_2_bilstm_cfg = OmegaConf.structured(
    DualModelStagedBERTConfig(
        bert=bert_2_bilstm_config,
        experiment=bert_2_bilstm_experiment_config,
        transformation=label_transformation_config,
        first_model_bilstm=bilstm_cfg,
    )
)

dual_stage_bert(OmegaConf.create(bert_2_bilstm_cfg))

bert_2_bilstm_run_id = get_run_id(bert_2_bilstm_cfg)

print(bert_2_bilstm_run_id)

bert_2_bilstm_run = mlflow.get_run(bert_2_bilstm_run_id)
mlflow.artifacts.download_artifacts(
    f"{bert_2_bilstm_run.info.artifact_uri}/0_model",
    dst_path=Path("./src/service/models/"),
)
try:
    shutil.rmtree(Path("./src/service/models/bert_2_bilstm"))
except FileNotFoundError:
    pass
Path("./src/service/models/0_model").rename(
    "./src/service/models/bert_2_bilstm"
)

# %% [markdown]
# ## Train BERT E2E Model

# %%
from tooling.observability import get_run_id
from tooling.config import BERTConfig, BERT
from experiments.bert import bert
from copy import deepcopy

bert_experiment_config = deepcopy(base_experiment_config)
bert_experiment_config.name = "BERT Train for E2E Application"

bert_config = BERT(max_len=sentence_length)

bert_cfg = OmegaConf.structured(
    BERTConfig(
        bert=bert_config,
        experiment=bert_experiment_config,
        transformation=label_transformation_config,
    )
)

bert(OmegaConf.create(bert_cfg))

bert_run_id = get_run_id(bert_cfg)

print(bert_run_id)

run = mlflow.get_run(bert_run_id)
mlflow.artifacts.download_artifacts(
    f"{run.info.artifact_uri}/0_model", dst_path=Path("./src/service/models/")
)
try:
    shutil.rmtree(Path("./src/service/models/bert"))
except FileNotFoundError:
    pass
Path("./src/service/models/0_model").rename("./src/service/models/bert")


