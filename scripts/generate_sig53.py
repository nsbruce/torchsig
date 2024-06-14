from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from typing import List
import click
import os
import psutil

NUM_CPUS = len(psutil.Process().cpu_affinity())


def generate(path: str, configs: List[conf.Sig53Config]):
    for config in configs:
        ds = ModulationsDataset(
            level=config.level,
            num_samples=config.num_samples,
            num_iq_samples=config.num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
        )
        loader = DatasetLoader(
            ds,
            seed=12345678,
            num_workers=NUM_CPUS,#os.cpu_count() // 2,
            batch_size=NUM_CPUS,#os.cpu_count() // 2,
        )
        creator = DatasetCreator(
            ds,
            seed=12345678,
            path="{}".format(os.path.join(path, config.name)),
            loader=loader,
        )
        creator.create()


@click.command()
@click.option("--root", default="sig53", help="Path to generate sig53 datasets")
@click.option("--all", default=True, help="Generate all versions of sig53 dataset.")
@click.option("--qa", default=False, help="Generate only QA versions of sig53 dataset.")
@click.option(
    "--impaired",
    default=False,
    help="Generate impaired dataset. Ignored if --all=True (default)",
)
@click.option(
        "--config-index",
        help="Generate only the dataset for the given config index",
)
def main(root: str, all: bool, qa: bool, impaired: bool, config_index:int):
    if not os.path.isdir(root):
        os.mkdir(root)

    configs = [
        conf.Sig53CleanTrainConfig,
        conf.Sig53CleanValConfig,
        conf.Sig53ImpairedTrainConfig,
        conf.Sig53ImpairedValConfig,
        conf.Sig53CleanTrainQAConfig,
        conf.Sig53CleanValQAConfig,
        conf.Sig53ImpairedTrainQAConfig,
        conf.Sig53ImpairedValQAConfig,
    ]
    if qa:
        generate(root, configs[4:])
        return

    if all:
        generate(root, configs[:4])
        return

    if impaired:
        generate(root, configs[2:4])
        return

    if config_index:
        print("config_index",int(config_index))
        generate(root, [configs[int(config_index)]])
        print("done generating")
        return

    generate(root, configs[:2])


if __name__ == "__main__":
    main()
