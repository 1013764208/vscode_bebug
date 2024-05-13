import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datetime import datetime

import os


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "path to pretrained model"}
    )


def main():
    model_args = HfArgumentParser((ModelArguments)).parse_args_into_dataclasses
    print(model_args)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    cur_datetime = datetime.now

    print("*" * 80)

    value = torch.cuda.device_count()
    print(
        f"----> cur_datetime: {cur_datetime},world size: {world_size}, local_rank: {local_rank}, gpu count: {value}"
    )
    return value


if __name__ == "__main__":
    main()

    # ----> cur_datetime: <built-in method now of type object at 0x7f1452674a60>,world size: 1, local_rank: 0, gpu count: 1