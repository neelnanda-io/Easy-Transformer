"""Build the API Documentation."""

import base64
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import warnings
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import pandas as pd  # type: ignore[import-untyped]
import torch
import tqdm  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from muutils.dictmagic import TensorDictFormats, condense_tensor_dict
from muutils.misc import shorten_numerical_to_str
from transformers import AutoTokenizer  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizer

import transformer_lens  # type: ignore[import-untyped]
from transformer_lens import (
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
    loading,
)
from transformer_lens.loading_from_pretrained import (  # type: ignore[import-untyped]
    NON_HF_HOSTED_MODEL_NAMES,
    get_pretrained_model_config,
)

DEVICE: torch.device = torch.device("meta")

# disable the symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN.startswith("hf_"):
        raise ValueError("Invalid Hugging Face token")
except Exception as e:
    warnings.warn(
        f"Failed to get Hugging Face token -- info about certain models will be limited\n{e}"
    )

# Docs Directories
CURRENT_DIR: Path = Path(__file__).parent
SOURCE_PATH: Path = CURRENT_DIR / "../docs/source"
BUILD_PATH: Path = CURRENT_DIR / "../docs/build"
PACKAGE_DIR: Path = CURRENT_DIR.parent
DEMOS_DIR: Path = CURRENT_DIR.parent / "demos"
GENERATED_DIR: Path = CURRENT_DIR.parent / "docs/source/generated"
STATIC_DIR: Path = CURRENT_DIR.parent / "docs/source/_static"


@lru_cache(maxsize=None)
def get_config(model_name: str):
    """Retrieve the configuration of a pretrained model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        dict: Configuration of the pretrained model.
    """
    return loading.get_pretrained_model_config(model_name)


# manually defined known model types
KNOWN_MODEL_TYPES: Sequence[str] = (
    "gpt2",
    "distillgpt2",
    "opt",
    "gpt-neo",
    "gpt-j",
    "gpt-neox",
    "stanford-gpt2",
    "pythia",
    "solu",
    "gelu",
    "attn-only",
    "llama",
    "Llama-2",
    "bert",
    "tiny-stories",
    "stablelm",
    "bloom",
    "qwen",
    "mistral",
    "CodeLlama",
    "phi",
    "gemma",
    "yi",
    "t5",
    "mixtral",
    "Qwen2",
)

MODEL_ALIASES_MAP: dict[str, str] = transformer_lens.loading.make_model_alias_map()

# these will be copied as table columns
CONFIG_ATTRS_COPY: Sequence[str] = (
    "n_params",
    "n_layers",
    "n_heads",
    "d_model",
    "d_vocab",
    "act_fn",
    "positional_embedding_type",
    "parallel_attn_mlp",
    "original_architecture",
    "normalization_type",
)

# modify certain values when saving config
CONFIG_VALUES_PROCESS: dict[str, Callable] = {
    "initializer_range": float,
    "dtype": str,
    "device": str,
}

COLUMNS_ABRIDGED: Sequence[str] = (
    "name.default_alias",
    "name.huggingface",
    "n_params.as_str",
    "n_params.as_int",
    "cfg.n_params",
    "cfg.n_layers",
    "cfg.n_heads",
    "cfg.d_model",
    "cfg.d_vocab",
    "cfg.act_fn",
    "cfg.positional_embedding_type",
    "cfg.parallel_attn_mlp",
    "cfg.original_architecture",
    "cfg.normalization_type",
    "tokenizer.name",
    "tokenizer.class",
    "tokenizer.vocab_size",
    "tokenizer.vocab_hash",
)


def get_tensor_shapes(
    model: HookedTransformer,
    tensor_dims_fmt: TensorDictFormats = "yaml",
    except_if_forward_fails: bool = False,
) -> dict:
    """get the tensor shapes from a model"""
    model_info: dict = dict()
    # state dict
    model_info["tensor_shapes.state_dict"] = condense_tensor_dict(
        model.state_dict(), fmt=tensor_dims_fmt
    )
    model_info["tensor_shapes.state_dict.raw__"] = condense_tensor_dict(
        model.state_dict(), fmt="dict"
    )

    try:
        # input shape for activations -- "847"~="bat", subtract 7 for the context window to make it unique
        input_shape: tuple[int, int] = (847, model.cfg.n_ctx - 7)
        # why? to replace the batch and seq_len dims with "batch" and "seq_len" in the yaml
        dims_names_map: dict[int, str] = {
            input_shape[0]: "batch",
            input_shape[1]: "seq_len",
        }
        # run with cache to activation cache
        with torch.no_grad():
            cache: ActivationCache
            _, cache = model.run_with_cache(
                torch.empty(input_shape, dtype=torch.long, device=DEVICE)
            )
        # condense using muutils and store
        model_info["tensor_shapes.activation_cache"] = condense_tensor_dict(
            cache.cache_dict,
            fmt=tensor_dims_fmt,
            dims_names_map=dims_names_map,
        )
        model_info["tensor_shapes.activation_cache.raw__"] = condense_tensor_dict(
            cache.cache_dict,
            fmt="dict",
            dims_names_map=dims_names_map,
        )
    except Exception as e:
        msg: str = f"Failed to get activation cache for '{model.cfg.model_name}':\n{e}"
        if except_if_forward_fails:
            raise ValueError(msg) from e
        else:
            warnings.warn(msg)

    return model_info


def tokenizer_vocab_hash(tokenizer: PreTrainedTokenizer) -> str:
    # sort
    vocab: dict[str, int]
    try:
        vocab = tokenizer.vocab
    except Exception:
        vocab = tokenizer.get_vocab()

    vocab_hashable: list[tuple[str, int]] = list(
        sorted(
            vocab.items(),
            key=lambda x: x[1],
        )
    )
    # hash it
    hash_obj = hashlib.sha1(bytes(str(vocab_hashable), "UTF-8"))
    # convert to base64
    return base64.b64encode(
        hash_obj.digest(),
        altchars=b"-_",  # - and _ as altchars
    ).decode("UTF-8")


def get_tokenizer_info(model: HookedTransformer) -> dict:
    tokenizer: PreTrainedTokenizer = model.tokenizer
    model_info: dict = dict()
    # basic info
    model_info["tokenizer.name"] = tokenizer.name_or_path
    model_info["tokenizer.vocab_size"] = int(tokenizer.vocab_size)
    model_info["tokenizer.max_len"] = int(tokenizer.model_max_length)
    model_info["tokenizer.class"] = tokenizer.__class__.__name__

    # vocab hash
    model_info["tokenizer.vocab_hash"] = tokenizer_vocab_hash(tokenizer)
    return model_info


def get_model_info(
    model_name: str,
    include_cfg: bool = True,
    include_tensor_dims: bool = True,
    include_tokenizer_info: bool = True,
    tensor_dims_fmt: TensorDictFormats = "yaml",
    allow_warn: bool = True,
) -> tuple[str, dict]:
    """get information about the model from the default alias model name

    # Parameters:
     - `model_name : str`
        the default alias model name
     - `include_cfg : bool`
        whether to include the model config as a yaml string
       (defaults to `True`)
     - `include_tensor_dims : bool`
        whether to include the model tensor shapes
       (defaults to `True`)
     - `include_tokenizer_info : bool`
        whether to include the tokenizer info
        (defaults to `True`)
     - `tensor_dims_fmt : TensorDictFormats`
        the format of the tensor shapes. one of "yaml", "json", "dict"
       (defaults to `"yaml"`)
    """

    # assumes the input is a default alias
    if model_name not in transformer_lens.loading.DEFAULT_MODEL_ALIASES:
        raise ValueError(f"Model name '{model_name}' not found in default aliases")

    # get the names and model types
    official_name: Optional[str] = MODEL_ALIASES_MAP.get(model_name, None)
    model_info: dict = {
        "name.default_alias": model_name,
        "name.huggingface": official_name,
        "name.aliases": ", ".join(
            list(transformer_lens.loading.MODEL_ALIASES.get(official_name, []))  # type: ignore[arg-type]
        ),
        "model_type": None,
    }

    # Split the model name into parts
    parts: list[str] = model_name.split("-")

    # identify model type by known types
    for known_type in KNOWN_MODEL_TYPES:
        if known_type in model_name:
            model_info["model_type"] = known_type
            break

    # search for model size in name
    param_count_from_name: Optional[str] = None
    for part in parts:
        if part[-1].lower() in ["m", "b", "k"] and part[:-1].replace(".", "", 1).isdigit():
            param_count_from_name = part
            break

    # update model info from config
    model_cfg: HookedTransformerConfig = get_pretrained_model_config(model_name)
    model_info.update(
        {
            "name.from_cfg": model_cfg.model_name,
            "n_params.as_str": shorten_numerical_to_str(model_cfg.n_params),  # type: ignore[arg-type]
            "n_params.as_int": model_cfg.n_params,
            "n_params.from_name": param_count_from_name,
            **{f"cfg.{attr}": getattr(model_cfg, attr) for attr in CONFIG_ATTRS_COPY},
        }
    )

    # put the whole config as yaml (for readability)
    if include_cfg:
        # modify certain values to make them pretty-printable
        model_cfg_dict: dict = {
            key: (val if key not in CONFIG_VALUES_PROCESS else CONFIG_VALUES_PROCESS[key](val))
            for key, val in model_cfg.to_dict().items()
        }

        # raw config
        model_info["config.raw__"] = model_cfg_dict
        # dump to yaml
        model_info["config"] = yaml.dump(
            model_cfg_dict,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )

    # get tensor shapes
    if include_tensor_dims or include_tokenizer_info:
        # set default device to meta, so that we don't actually allocate tensors
        # this can't be done at the root level because it would break other tests when we import this file
        # and it has to be done inside this function due to usage of multiprocessing
        with torch.device(DEVICE):
            got_model: bool = False
            try:
                # copy the config, so we can modify it
                model_cfg_copy: HookedTransformerConfig = deepcopy(model_cfg)
                # set device to "meta" -- don't actually initialize the model with real tensors
                model_cfg_copy.device = str(DEVICE)
                if not include_tokenizer_info:
                    # don't need to download the tokenizer
                    model_cfg_copy.tokenizer_name = None
                # init the fake model
                model: HookedTransformer = HookedTransformer(model_cfg_copy, move_to_device=True)
                # HACK: use https://huggingface.co/huggyllama to get tokenizers for original llama models
                if model.cfg.tokenizer_name in NON_HF_HOSTED_MODEL_NAMES:
                    model.set_tokenizer(
                        AutoTokenizer.from_pretrained(
                            f"huggyllama/{model.cfg.tokenizer_name.removesuffix('-hf')}",
                            add_bos_token=True,
                            token=HF_TOKEN,
                            legacy=False,
                        )
                    )
                got_model = True
            except Exception as e:
                msg: str = f"Failed to init model '{model_name}', can't get tensor shapes or tokenizer info"
                if allow_warn:
                    warnings.warn(f"{msg}:\n{e}")
                else:
                    raise ValueError(msg) from e

            if got_model:
                if include_tokenizer_info:
                    try:
                        tokenizer_info: dict = get_tokenizer_info(model)
                        model_info.update(tokenizer_info)
                    except Exception as e:
                        msg = f"Failed to get tokenizer info for model '{model_name}'"
                        if allow_warn:
                            warnings.warn(f"{msg}:\n{e}")
                        else:
                            raise ValueError(msg) from e

                if include_tensor_dims:
                    try:
                        tensor_shapes_info: dict = get_tensor_shapes(model, tensor_dims_fmt)
                        model_info.update(tensor_shapes_info)
                    except Exception as e:
                        msg = f"Failed to get tensor shapes for model '{model_name}'"
                        if allow_warn:
                            warnings.warn(f"{msg}:\n{e}")
                        else:
                            raise ValueError(msg) from e

    return model_name, model_info


def safe_try_get_model_info(
    model_name: str, kwargs: Optional[dict] = None
) -> tuple[str, Union[dict, Exception]]:
    """for parallel processing, to catch exceptions and return the exception instead of raising them"""
    if kwargs is None:
        kwargs = {}
    try:
        return get_model_info(model_name, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to get model info for '{model_name}': {e}")
        return model_name, e


def make_model_table(
    verbose: bool,
    allow_except: bool = False,
    parallelize: Union[bool, int] = True,
    model_names_pattern: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """make table of all models. kwargs passed to `get_model_info()`"""
    model_names: list[str] = list(transformer_lens.loading.DEFAULT_MODEL_ALIASES)
    model_data: list[tuple[str, Union[dict, Exception]]] = list()

    # filter by regex pattern if provided
    if model_names_pattern:
        model_names = [
            model_name for model_name in model_names if model_names_pattern in model_name
        ]

    if parallelize:
        # parallel
        n_processes: int = parallelize if int(parallelize) > 1 else multiprocessing.cpu_count()
        if verbose:
            print(f"running in parallel with {n_processes=}")
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Use imap for ordered results, wrapped with tqdm for progress bar
            imap_results: list[tuple[str, Union[dict, Exception]]] = list(
                tqdm.tqdm(
                    pool.imap(
                        partial(safe_try_get_model_info, **kwargs),
                        model_names,
                    ),
                    total=len(model_names),
                    desc="Loading model info",
                    disable=not verbose,
                )
            )

        model_data = imap_results

    else:
        # serial
        with tqdm.tqdm(
            transformer_lens.loading.DEFAULT_MODEL_ALIASES,
            desc="Loading model info",
            disable=not verbose,
        ) as pbar:
            for model_name in pbar:
                pbar.set_postfix_str(f"model: '{model_name}'")
                try:
                    model_data.append(get_model_info(model_name, **kwargs))
                except Exception as e:
                    if allow_except:
                        # warn and continue if we allow exceptions
                        warnings.warn(f"Failed to get model info for '{model_name}': {e}")
                        model_data.append((model_name, e))
                    else:
                        # raise exception right away if we don't allow exceptions
                        # note that this differs from the parallel version, which will only except at the end
                        raise ValueError(f"Failed to get model info for '{model_name}'") from e

    # figure out what to do with failed models
    failed_models: dict[str, Exception] = {
        model_name: result for model_name, result in model_data if isinstance(result, Exception)
    }
    msg: str = (
        f"Failed to get model info for {len(failed_models)}/{len(model_names)} models: {failed_models}\n"
        + "\n".join(f"\t'{model_name}': {expt}" for model_name, expt in failed_models.items())
    )
    if not allow_except:
        if failed_models:
            # raise exception if we don't allow exceptions
            raise ValueError(msg + "\n\n" + "=" * 80 + "\n\n" + "NO DATA WRITTEN")
    else:
        if failed_models:
            warnings.warn(msg + "\n\n" + "-" * 80 + "\n\n" + "WRITING PARTIAL DATA")

    # filter out failed models if we allow exceptions
    model_data_filtered: list[dict] = [
        result for _, result in model_data if not isinstance(result, Exception)
    ]
    return pd.DataFrame(model_data_filtered)


OutputFormat = Literal["jsonl", "csv", "md"]


def huggingface_name_to_url(df: pd.DataFrame) -> pd.DataFrame:
    """convert the huggingface model name to a url"""
    df_new: pd.DataFrame = df.copy()
    df_new["name.huggingface"] = df_new["name.huggingface"].map(
        lambda x: f"[{x}](https://huggingface.co/{x})" if x else x
    )
    return df_new


MD_TABLE_HEARDER: str = """---
title: Model Properties Table
hide-toc: true
---
# Model Properties Table

also see the [interactive model table](../_static/model_properties_table_interactive.html)
"""


def write_model_table(
    model_table: pd.DataFrame,
    path: Path,
    format: OutputFormat = "jsonl",
    include_TL_version: bool = True,
    md_hf_links: bool = True,
    md_header: str = MD_TABLE_HEARDER,
) -> None:
    """write the model table to disk in the specified format"""

    # make sure the directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    if include_TL_version:
        # get `transformer_lens` version
        tl_version: str = "unknown"
        try:
            from importlib.metadata import PackageNotFoundError, version

            tl_version = version("transformer_lens")
        except PackageNotFoundError as e:
            warnings.warn(f"Failed to get transformer_lens version: package not found\n{e}")
        except Exception as e:
            warnings.warn(f"Failed to get transformer_lens version: {e}")

        with open(path.with_suffix(".version"), "w") as f:
            json.dump({"version": tl_version}, f)

    if format == "jsonl":
        model_table.to_json(path.with_suffix(".jsonl"), orient="records", lines=True)
    elif format == "csv":
        model_table.to_csv(path.with_suffix(".csv"), index=False)
    elif format == "md":
        model_table_processed: pd.DataFrame = model_table
        # convert huggingface name to url
        if md_hf_links:
            model_table_processed = huggingface_name_to_url(model_table_processed)
        model_table_md_text: str = md_header + model_table_processed.to_markdown(index=False)
        with open(path.with_suffix(".md"), "w") as f:
            f.write(model_table_md_text)
    else:
        raise KeyError(f"Invalid format: {format}")


def abridge_model_table(
    model_table: pd.DataFrame,
    columns_keep: Sequence[str] = COLUMNS_ABRIDGED,
    null_to_empty: bool = True,
) -> pd.DataFrame:
    """keep only columns in COLUMNS_ABRIDGED

    primarily used to make the csv and md versions of the table readable

    also replaces `None` with empty string if `null_to_empty` is `True`
    """

    output: pd.DataFrame = model_table.copy()
    # filter columns
    output = output[list(columns_keep)]

    if null_to_empty:
        output = output.fillna("")

    return output


def get_model_table(
    model_table_path: Path,
    verbose: bool = True,
    force_reload: bool = True,
    do_write: bool = True,
    parallelize: Union[bool, int] = True,
    model_names_pattern: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """get the model table either by generating or reading from jsonl file

    # Parameters:
     - `model_table_path : Path`
        the path to the model table file, and the base name for the csv and md files
     - `verbose : bool`
        whether to show progress bar
       (defaults to `True`)
     - `force_reload : bool`
        force creating the table from scratch, even if file exists
       (defaults to `True`)
     - `do_write : bool`
        whether to write the table to disk, if generating
       (defaults to `True`)
     - `model_names_pattern : Optional[str]`
        filter the model names by making them include this string. passed to `make_model_table()`. no filtering if `None`
        (defaults to `None`)
     - `**kwargs`
        passed to `make_model_table()`

    # Returns:
     - `pd.DataFrame`
        the model table. rows are models, columns are model attributes
    """

    # modify the name if a pattern is provided
    if model_names_pattern is not None:
        model_table_path = model_table_path.with_name(
            model_table_path.stem + f"-{model_names_pattern}"
        )

    model_table: pd.DataFrame
    if not model_table_path.exists() or force_reload:
        # generate it from scratch
        model_table = make_model_table(
            verbose=verbose,
            parallelize=parallelize,
            model_names_pattern=model_names_pattern,
            **kwargs,
        )
        if do_write:
            # full data as jsonl
            write_model_table(model_table, model_table_path, format="jsonl")
            # abridged data as csv, md
            abridged_table: pd.DataFrame = abridge_model_table(model_table)
            write_model_table(abridged_table, model_table_path, format="csv")
            write_model_table(abridged_table, model_table_path, format="md")
    else:
        # read the table from jsonl
        model_table = pd.read_json(model_table_path, orient="records", lines=True)

    return model_table


def copy_demos(_app: Optional[Any] = None):
    """Copy demo notebooks to the generated directory."""
    copy_to_dir = GENERATED_DIR / "demos"
    notebooks_to_copy = [
        "Exploratory_Analysis_Demo.ipynb",
        "Main_Demo.ipynb",
    ]

    if copy_to_dir.exists():
        shutil.rmtree(copy_to_dir)

    copy_to_dir.mkdir()
    for filename in notebooks_to_copy:
        shutil.copy(DEMOS_DIR / filename, copy_to_dir)


def build_docs():
    """Build the docs."""
    get_model_table(
        model_table_path=GENERATED_DIR / "model_properties_table.jsonl",
        force_reload=True,
    )
    copy_demos()

    # Generating docs
    subprocess.run(
        [
            "sphinx-build",
            SOURCE_PATH,
            BUILD_PATH,
            # "-n",  # Nitpicky mode (warn about all missing references)
            "-W",  # Turn warnings into errors
        ],
        check=True,
    )


def get_property(name, model_name):
    """Retrieve a specific property of a pretrained model.

    Args:
        name (str): Name of the property to retrieve.
        model_name (str): Name of the pretrained model.

    Returns:
        str: Value of the specified property.
    """
    cfg = get_config(model_name)

    if name == "act_fn":
        if cfg.attn_only:
            return "attn_only"
        if cfg.act_fn == "gelu_new":
            return "gelu"
        if cfg.act_fn == "gelu_fast":
            return "gelu"
        if cfg.act_fn == "solu_ln":
            return "solu"
        return cfg.act_fn
    if name == "n_params":
        n_params = cfg.n_params
        if n_params < 1e4:
            return f"{n_params/1e3:.1f}K"
        if n_params < 1e6:
            return f"{round(n_params/1e3)}K"
        if n_params < 1e7:
            return f"{n_params/1e6:.1f}M"
        if n_params < 1e9:
            return f"{round(n_params/1e6)}M"
        if n_params < 1e10:
            return f"{n_params/1e9:.1f}B"
        if n_params < 1e12:
            return f"{round(n_params/1e9)}B"
        raise ValueError(f"Passed in {n_params} above 1T?")
    return cfg.to_dict()[name]


def docs_hot_reload():
    """Hot reload the docs."""
    get_model_table(
        model_table_path=GENERATED_DIR / "model_properties_table.jsonl", force_reload=False
    )
    copy_demos()

    subprocess.run(
        [
            "sphinx-autobuild",
            "--watch",
            str(PACKAGE_DIR) + "," + str(DEMOS_DIR),
            "--open-browser",
            SOURCE_PATH,
            BUILD_PATH,
        ],
        check=True,
    )
