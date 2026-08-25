import argparse
import re
from dataclasses import dataclass
from typing import Literal
from ir_measures import RR, Recall, nDCG

import ir_datasets
from pyterrier_dr.jpq.utils import merge_queries_into_docpairs
from pyterrier_dr.jpq import JPQTrainer
import os
import pyterrier_dr
import pyterrier as pt
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel
from sentence_transformers import SentenceTransformer
torch.set_float32_matmul_precision('high')

# This script is to make it easy to run JPQ training and evaluation with different models
# The DEFAULT_INIT_BY_NAME dict maps a model name to a python expression that instantiates the model.
# It make require updating with new models, or for the star/adore_star models

DEFAULT_INIT_BY_NAME: dict[str, str] = {
    "tct_colbert": "pyterrier_dr.TctColBert.hnp()",
    "tas_b": "pyterrier_dr.TasB()",
    "star": "pyterrier_dr.STAR('/root/nfs/jpq/provided_models/star/')",
    "adore_star" : "pyterrier_dr.STAR('/root/nfs/jpq/provided_models/adore_star/')",
    "repllama": "pyterrier_dr.RepLLama.v1_7b()",
    "e5": "pyterrier_dr.E5()",
    "dragon" : 'pyterrier_dr.Dragon()',
    "lion" : 'pyterrier_dr.LionLlamaDense()',
}


def instantiate_model(model_name: str):
    """
    Instantiate the model by evaluating the resolved expression in a restricted namespace.
    Only whitelisted globals are visible to eval().
    """

    allowed_globals = {
        "pyterrier_dr": pyterrier_dr,
    }
    expr = DEFAULT_INIT_BY_NAME[model_name]
    try:
        obj = eval(expr, allowed_globals, {})   # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate model with expression: {expr}\n"
            f"Error: {e}"
        ) from e
    return obj


def apply_peft(biencoder: pyterrier_dr.BiEncoder, peft_config):
    model = getattr(biencoder, "model", None)

    if isinstance(model, SentenceTransformer):
        transformer = model[0]
        hf_model = transformer.auto_model
        transformer.model = get_peft_model(hf_model, peft_config)

    elif isinstance(model, PreTrainedModel):
        biencoder.model = get_peft_model(model, peft_config)

    else:
        raise NotImplementedError(
            f"PEFT is not supported for model type: {type(model)}"
        )

    return biencoder

_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")
def _slug(s: str, maxlen: int = 64) -> str:
    s = s.strip().replace(" ", "-")
    s = _slug_re.sub("-", s)
    return s[:maxlen].strip("-")


def compute_index_name(data_cfg, train_cfg) -> str:
    # Compact, human-readable summary of the run
    parts = [
        _slug(data_cfg.train_ds),
        _slug(data_cfg.model_name),
        _slug(train_cfg.pq_impl),
        f"M{train_cfg.M}_nbits{train_cfg.nbits}",
        f"ps{train_cfg.pq_sample_size}",
        f"neg{train_cfg.jpq_negs}" if train_cfg.jpq_negs else "",
        "ibn" if train_cfg.in_batch_negs else "",
        "lr" if train_cfg.lambda_rank else "",
        "frozen" if train_cfg.frozen_query_encoder else "",
    ]
    return "__".join(parts)


@dataclass
class DataConfig:
    base_index: str
    target_dir: str
    model_name: Literal["tct_colbert", "tas_b", "star", "adore_star", "e5", "repllama"] = "tct_colbert"
    train_ds: str = "msmarco-passage/train"
    #eval_ds: str = "msmarco_passage"
    #eval_split: str = "test-2019"
    eval_ds : str = 'irds:msmarco-passage/dev/small'
    eval_split : str|None = None
    test_ds: str = "msmarco_passage"
    test_split: str = "test-2019,test-2020"


@dataclass
class TrainingConfig:
    pq_impl: Literal['faiss', 'sklearn', 'faiss2', 'faiss2opq'] = "faiss2opq"
    M: int = 96
    nbits: int = 8
    pq_sample_size: int = 159_744
    valid_every: int = 500
    patience: int = 3
    total_steps: int = 1_000_000_000
    in_batch_negs: bool = True
    lambda_rank: bool = True
    jpq_negs: int = 200
    pairs_cap: int | None = 2_000_000
    frozen_query_encoder: bool = False
    pq_only: bool = False

def optional_str(value: str):
    if value.lower() == "none":
        return None
    return value
def add_data_args(parser: argparse.ArgumentParser):
    p = parser.add_argument_group("Data")
    p.add_argument("--base-index", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--model-name", choices=["tct_colbert", "tas_b", "star", "repllama", "adore_star", "e5", "dragon", "lion"], default="tct_colbert")
    p.add_argument("--train-ds", default="msmarco-passage/train")
    p.add_argument("--eval-ds", default="irds:msmarco-passage/dev/small")
    p.add_argument("--eval-split", type=optional_str, default=None)
    p.add_argument("--test-ds", default="msmarco_passage")
    p.add_argument("--test-split", type=optional_str, default="test-2019,test-2020")


def add_training_args(parser: argparse.ArgumentParser):
    p = parser.add_argument_group("Training")
    p.add_argument("--pq-impl", choices=['faiss', 'sklearn', 'faiss2', 'faiss2opq'], default="faiss2opq")
    p.add_argument("--M", type=int, default=96)
    p.add_argument("--nbits", type=int, default=8)
    p.add_argument("--pq-sample-size", type=int, default=159744)
    p.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000_000,
        help="Maximum number of JPQ training steps."
    )
    p.add_argument("--valid-every", type=int, default=500)
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in validation executions.",
    )
    p.add_argument("--in-batch-negs", action="store_true", default=True)
    p.add_argument("--no-in-batch", dest="in_batch_negs", action="store_false")
    p.add_argument("--lambda-rank", action="store_true", default=False)
    p.add_argument("--no-lambda-rank", dest="lambda_rank", action="store_false")
    p.add_argument("--jpq-negs", type=int, default=0)
    p.add_argument("--pairs-cap", type=int, default=2_000_000)
    p.add_argument("--frozen-query-encoder", action="store_true", default=False)
    p.add_argument("--pq-only", action="store_true", default=False)


def parse_args():
    parser = argparse.ArgumentParser(description="JPQ CLI")
    add_data_args(parser)
    add_training_args(parser)

    lora_group = parser.add_argument_group("LoRA")

    lora_group.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA for the E5 query encoder.",
    )

    lora_group.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank.",
    )

    lora_group.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA scaling alpha.",
    )

    lora_group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout probability.",
    )

    args = parser.parse_args()

    data = DataConfig(
        base_index=args.base_index,
        target_dir=args.target_dir,
        model_name=args.model_name,
        train_ds=args.train_ds,
        eval_ds=args.eval_ds,
        eval_split=args.eval_split,
        test_ds=args.test_ds,
        test_split=args.test_split,
    )
    train = TrainingConfig(
        pq_impl=args.pq_impl,
        M=args.M,
        nbits=args.nbits,
        pq_sample_size=args.pq_sample_size,
        valid_every=args.valid_every,
        patience=args.patience,
        total_steps=args.total_steps,
        in_batch_negs=args.in_batch_negs,
        lambda_rank=args.lambda_rank,
        jpq_negs=args.jpq_negs,
        pairs_cap=args.pairs_cap,
        frozen_query_encoder=args.frozen_query_encoder,
        pq_only=args.pq_only
    )
    return data, train, args


if __name__ == "__main__":
    data, train, args = parse_args()

    peft_config = None
    if args.use_lora:
        if args.frozen_query_encoder:
            raise ValueError(
                "--use-lora cannot be combined with "
                "--frozen-query-encoder."
            )

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "value"],
            bias="none",
        )

        print(
            "LORA CONFIG:",
            {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": ["query", "value"],
            },
        )

    print("DATA CONFIG:", data)
    print("TRAIN CONFIG:", train)

    target = data.target_dir + "/" + compute_index_name(data, train)
    print("DESTINATION:", target)

    model = instantiate_model(
        data.model_name,
        )
    if peft_config is not None:
        model = apply_peft(model, peft_config)
    print("Instantiated model:", model)

    index = pyterrier_dr.FlexIndex(data.base_index)
    print(f"Index: {data.base_index} has {len(index)} docs")

    train_dataset = ir_datasets.load(data.train_ds)
    print("training dataset loaded")

    eval_dataset = pt.get_dataset(data.eval_ds)
    print("eval dataset loaded")

    if train.lambda_rank and not train.jpq_negs:
        raise ValueError("lambda_rank loss fn currently requires jpq_negs to be set")

    # reproducibilty settings
    import faiss
    thread_count = faiss.omp_get_max_threads()
    import os
    os.environ["MKL_CBWR"] = "COMPATIBLE"
    faiss.omp_set_num_threads(1)

    import torch
    torch.manual_seed(0)

    t = JPQTrainer(model, index, M=train.M, pq_impl=train.pq_impl, nbits=train.nbits, train_query_encoder=not train.frozen_query_encoder)
    t.fit(
        merge_queries_into_docpairs(train_dataset.queries_iter(), train_dataset.docpairs_iter()[:train.pairs_cap]),
        pq_sample_size=train.pq_sample_size,
        total_steps=train.total_steps,
        valid_every=train.valid_every,
        patience=train.patience,
        eval_queries = eval_dataset.get_topics(data.eval_split),
        eval_qrels = eval_dataset.get_qrels(data.eval_split),
        in_batch=train.in_batch_negs,
        lambda_rank=train.lambda_rank,
        jpq_negs=train.jpq_negs,
        pq_only=train.pq_only,
    )
    faiss.omp_set_num_threads(thread_count)
    os.environ["MKL_CBWR"] = "AUTO"

    newindex = t.jpq_index(target)
    t.query_encoder.model.save_pretrained(target) # type: ignore

    oldmodel = instantiate_model(
        data.model_name,
       )

    p = [
        oldmodel >> index.retriever(), # type: ignore
        t.query_encoder >> newindex.retriever_pq()
    ]
    save_dir = target + "/" + 'runs'

    def _do_run(dataset, split, name):
        print(split, name)
        split_savedir = save_dir + "/" + name.replace(":", "_").replace("/", "_")
        os.makedirs(split_savedir, exist_ok=True)

        if split is None:
            topics = dataset.get_topics()
            qrels = dataset.get_qrels()
        else:
            topics = dataset.get_topics(split)
            qrels = dataset.get_qrels(split)


        df = pt.Experiment(
            p,
            # dataset.get_topics(split),
            # dataset.get_qrels(split),
            topics,
            qrels,
            eval_metrics=[RR@10, Recall(rel=2)@100, Recall@100, nDCG@10, "mrt"],
            names=["baseline", "JPQ pq"],
            save_dir = split_savedir,
            baseline=0
        )
        df.to_csv(split_savedir + "/metrics.csv")
        print(df)

    if data.test_split is None:
        _do_run(pt.get_dataset(data.test_ds), None, "test")
    else:
        for split in data.test_split.split(","):
            _do_run(pt.get_dataset(data.test_ds), split, split)
    if data.eval_ds is not None:
        _do_run(pt.get_dataset(data.eval_ds), data.eval_split, "dev")

