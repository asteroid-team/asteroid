import argparse

import dlp_mpi
import yaml
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
from sms_wsj.examples.reference_systems import experiment

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="data/sms_wsj.json", help="Full path to sms_wsj.json")


def main(conf):
    experiment.run(
        config_updates=dict(json_path=conf["main_args"]["json_path"], **conf["mm_config"])
    )


if __name__ == "__main__":
    if dlp_mpi.IS_MASTER:
        # We start with opening the config file conf.yml as a dictionary from
        # which we can create parsers. Each top level key in the dictionary defined
        # by the YAML file creates a group in the parser.
        with open("local/conf.yml") as f:
            def_conf = yaml.safe_load(f)
        parser = prepare_parser_from_dict(def_conf, parser=parser)
        # Arguments are then parsed into a hierarchical dictionary (instead of
        # flat, as returned by argparse) to falicitate calls to the different
        # asteroid methods (see in main).
        # plain_args is the direct output of parser.parse_args() and contains all
        # the attributes in an non-hierarchical structure. It can be useful to also
        # have it so we included it here but it is not used.
        arg_dict, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    else:
        arg_dict = None
    arg_dict = dlp_mpi.bcast(arg_dict, root=dlp_mpi.MASTER)
    main(arg_dict)
