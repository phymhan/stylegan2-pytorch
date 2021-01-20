import os
import sys
import math
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def estimate(netNetwork, tenFirst, tenSecond):
    # Copied from https://github.com/sniklaus/pytorch-pwc/blob/master/run.py
    # Assume tensors are normalized to [-1, 1]
    tenFirst = (tenFirst + 1.) / 2
    tenSecond = (tenSecond + 1.) / 2
    assert(tenFirst.shape[1] == tenSecond.shape[1])
    assert(tenFirst.shape[2] == tenSecond.shape[2])

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = min(int(math.floor(math.ceil(intWidth / 64.0) * 64.0)), 128)
    intPreprocessedHeight = min(int(math.floor(math.ceil(intHeight / 64.0) * 64.0)), 128)

    tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    # tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    tenFlow = 20.0 * netNetwork(tenPreprocessedFirst, tenPreprocessedSecond)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :]
