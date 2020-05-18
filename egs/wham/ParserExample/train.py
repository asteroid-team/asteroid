import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')

if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
    # Load
    def_conf = yaml.safe_load(open('local/conf.yml'))
    # Create parser
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Parse CLI
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)

    # Arg_dic is a dictionary following the structure of `conf.yml`
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure.
    print(arg_dic)
