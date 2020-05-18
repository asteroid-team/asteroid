import argparse
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_yaml')
parser.add_argument('--prefix', default='')
parser.add_argument('--make_args', action='store_true')


def flat_twolvl(d):
    out = dict()
    key_list = []
    for key in d:
        # Detection to be improved
        if any(k in key_list for k in d[key]):
            raise ValueError("Non-unique second-level keys")
        key_list.append(d[key].keys())
        # One level dictionary
        out.update(d[key])
    return out


def assign(dic, prefix=''):
    for k, v in dic.items():
        # Make assignments bash-friendly
        v = sanitize_assign(v)
        # train_key=value instead of key=value
        k = prefix + '_' + k if prefix else k
        # key=value statement
        print(f'{k}={v}')


def sanitize_assign(value):
    # Empty assignement k=
    if value is None:
        return ''
    # List assign k="1,2"
    if isinstance(value, list):
        joined = ' '.join([str(i) for i in value])
        return f"\'{joined}\'"
    return value


def make_args(dic, prefix=''):
    output_args = []
    for k, _ in dic.items():
        # Passing to argparse
        k = prefix + '_' + k if prefix else k
        v = os.getenv(k, None)  # Get env variable with default
        v = sanitize_make_args(v)
        # v = v if v else '\'\''  # Need to escape the quotes to print
        output_args.append(f'--{k} {v}')
    print(' '.join(output_args))


def sanitize_make_args(value):
    # Print nothing if arg is empty
    if value in [None, '']:
        value = ''
    # Comma separated gets to be list
    elif ',' in value:
        value = ' '.join(value.split(','))
    return value


if __name__ == '__main__':
    args = parser.parse_args()

    conf = yaml.safe_load(open(args.input_yaml))
    conf = flat_twolvl(conf)
    if args.make_args:
        make_args(conf, prefix=args.prefix)
    else:
        assign(conf, prefix=args.prefix)
