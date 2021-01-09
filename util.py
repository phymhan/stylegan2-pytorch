import os
import sys


def set_log_dir(args):
    args.log_dir = os.path.join(args.log_root, args.name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        os.makedirs(os.path.join(args.log_dir, 'sample'))
        os.makedirs(os.path.join(args.log_dir, 'weight'))
    return args


def print_args(parser, args):
    message = ''
    message += '--------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    # print(message)  # suppress messages to std out

    # save to the disk
    exp_dir = args.log_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    file_name = os.path.join(exp_dir, 'args.txt')
    with open(file_name, 'wt') as f:
        f.write(message)
        f.write('\n')

    # save command to disk
    file_name = os.path.join(exp_dir, 'cmd.txt')
    with open(file_name, 'wt') as f:
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write(' python ')
        f.write(' '.join(sys.argv))
        f.write('\n')
