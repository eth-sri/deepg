import argparse
import csv
import os
import copy
import numpy as np
import sys
import tensorflow as tf
from read_net_file import read_net
sys.path.insert(0, '../ELINA/python_interface/')
from eran import ERAN
from elina_coeff import *
from elina_linexpr0 import *
import time

EPS = 10**(-9)
n_rows, n_cols, n_channels = 0, 0, 0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize(image, means, stds, dataset, is_conv):
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    else:
        for i in range(3072):
            image[i] = (image[i] - means[i % 3]) / stds[i % 3]


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def show_ascii_spec(lb, ub):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')


def main():
    parser = argparse.ArgumentParser(description='Analyze NN.')
    parser.add_argument('--net', type=str, help='Neural network to analyze')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset')
    parser.add_argument('--data_dir', type=str, help='Directory which contains data')
    parser.add_argument('--data_root', type=str, help='Directory which contains data')
    parser.add_argument('--num_params', type=int, default=0, help='Number of transformation parameters')
    parser.add_argument('--num_tests', type=int, default=None, help='Number of images to test')
    parser.add_argument('--from_test', type=int, default=0, help='Number of images to test')
    parser.add_argument('--test_idx', type=int, default=None, help='Index to test')
    parser.add_argument('--debug', action='store_true', help='Whether to display debug info')
    parser.add_argument('--attack', action='store_true', help='Whether to attack')
    parser.add_argument('--timeout_lp', type=float, default=1,  help='timeout for the LP solver')
    parser.add_argument('--timeout_milp', type=float, default=1,  help='timeout for the MILP solver')
    parser.add_argument('--use_area_heuristic', type=str2bool, default=True,  help='whether to use area heuristic for the DeepPoly ReLU approximation')
    args = parser.parse_args()

    global n_rows, n_cols, n_channels
    if args.dataset == 'cifar10':
        n_rows, n_cols, n_channels = 32, 32, 3
    else:
        n_rows, n_cols, n_channels = 28, 28, 1

    filename, file_extension = os.path.splitext(args.net)

    is_trained_with_pytorch = False
    is_saved_tf_model = False

    if(file_extension == ".net" or file_extension == ".pyt"):
        is_trained_with_pytorch = True
    elif(file_extension == ".meta"):
        is_saved_tf_model = True
    elif(file_extension != ".tf"):
        print("file extension not supported")
        exit(1)

    is_conv = False

    if(is_saved_tf_model):
        netfolder = os.path.dirname(args.net)

        tf.logging.set_verbosity(tf.logging.ERROR)

        sess = tf.Session()
        saver = tf.train.import_meta_graph(args.net)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
        eran = ERAN(sess.graph.get_tensor_by_name('logits:0'), sess)
    else:
        if args.dataset == 'mnist' or args.dataset == 'fashion':
            num_pixels = 784
        else:
            num_pixels = 3072
        model, is_conv, means, stds = read_net(args.net, num_pixels, is_trained_with_pytorch)
        eran = ERAN(model)

    csvfile = open('../../code/datasets/{}_test.csv'.format(args.dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')

    total, attacked, standard_correct, tot_time = 0, 0, 0, 0
    correct_box, correct_poly = 0, 0
    cver_box, cver_poly = [], []

    for i, test in enumerate(tests):
        if args.test_idx is not None and i != args.test_idx:
            continue

        attacks_file = os.path.join(args.data_dir, 'attack_{}.csv'.format(i))
        if args.num_tests is not None and i >= args.num_tests:
            break
        print('Test {}:'.format(i))
        
        if args.dataset == 'mnist' or args.dataset == 'fashion':
            image = np.float64(test[1:len(test)])
        else:
            if is_trained_with_pytorch:
                image = np.float64(test[1:len(test)])
            else:
                image = np.float64(test[1:len(test)]) - 0.5

        spec_lb = np.copy(image)
        spec_ub = np.copy(image)

        if(is_trained_with_pytorch):
            normalize(spec_lb, means, stds, args.dataset, is_conv)
            normalize(spec_ub, means, stds, args.dataset, is_conv)

        label, nn, nlb, nub = eran.analyze_box(spec_lb, spec_ub, 'deeppoly', args.timeout_lp, args.timeout_milp, args.use_area_heuristic)
        print('Label: ', label)

        if label != int(test[0]):
            print('Label {}, but true label is {}, skipping...'.format(label, int(test[0])))
            print('Standard accuracy: {} percent'.format(standard_correct/float(i+1)*100))
            continue
        else:
            standard_correct += 1
            print('Standard accuracy: {} percent'.format(standard_correct/float(i+1)*100))

        dim = n_rows * n_cols * n_channels

        ok_box, ok_poly = True, True
        k = args.num_params + 1 + 1 + dim

        attack_imgs, checked, attack_pass = [], [], 0
        cex_found = False
        if args.attack:
            with open(attacks_file, 'r') as fin:
                lines = fin.readlines()
                for j in range(0, len(lines), args.num_params+1):
                    params = [float(line[:-1]) for line in lines[j:j+args.num_params]]
                    tokens = lines[j+args.num_params].split(',')
                    values = np.array(list(map(float, tokens)))

                    attack_lb = values[::2]
                    attack_ub = values[1::2]

                    if is_trained_with_pytorch:
                        normalize(attack_lb, means, stds, args.dataset, is_conv)
                        normalize(attack_ub, means, stds, args.dataset, is_conv)
                    else:
                        attack_lb -= 0.5
                        attack_ub -= 0.5
                    attack_imgs.append((params, attack_lb, attack_ub))
                    checked.append(False)

                    predict_label, _, _, _ = eran.analyze_box(
                        attack_lb[:dim], attack_ub[:dim], 'deeppoly',
                        args.timeout_lp, args.timeout_milp, args.use_area_heuristic, 0)
                    if predict_label != int(test[0]):
                        print('counter-example, params: ', params, ', predicted label: ', predict_label)
                        cex_found = True
                        break
                    else:
                        attack_pass += 1
        print('tot attacks: ', len(attack_imgs))
        specs_file = os.path.join(args.data_dir, '{}.csv'.format(i))
        begtime = time.time()
        with open(specs_file, 'r') as fin:
            lines = fin.readlines()
            print('Number of lines: ', len(lines))
            assert len(lines) % k == 0

            spec_lb = np.zeros(args.num_params + dim)
            spec_ub = np.zeros(args.num_params + dim)

            expr_size = args.num_params
            lexpr_cst, uexpr_cst = [], []
            lexpr_weights, uexpr_weights = [], []
            lexpr_dim, uexpr_dim = [], []

            ver_chunks_box, ver_chunks_poly, tot_chunks = 0, 0, 0
            
            for i, line in enumerate(lines):
                if i % k < args.num_params:
                    # read specs for the parameters
                    values = np.array(list(map(float, line[:-1].split(' '))))
                    assert values.shape[0] == 2
                    param_idx = i % k
                    spec_lb[dim + param_idx] = values[0]
                    spec_ub[dim + param_idx] = values[1]
                    if args.debug:
                        print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
                elif i % k == args.num_params:
                    # read interval bounds for image pixels
                    values = np.array(list(map(float, line[:-1].split(','))))
                    spec_lb[:dim] = values[::2]
                    spec_ub[:dim] = values[1::2]
                    # if args.debug:
                    #     show_ascii_spec(spec_lb, spec_ub)
                elif i % k < k - 1:
                    # read polyhedra constraints for image pixels
                    tokens = line[:-1].split(' ')
                    assert len(tokens) == 2 + 2*args.num_params + 1

                    bias_lower, weights_lower = float(tokens[0]), list(map(float, tokens[1:1+args.num_params]))
                    assert tokens[args.num_params+1] == '|'
                    bias_upper, weights_upper = float(tokens[args.num_params+2]), list(map(float, tokens[3+args.num_params:]))
                    
                    assert len(weights_lower) == args.num_params
                    assert len(weights_upper) == args.num_params
                    
                    lexpr_cst.append(bias_lower)
                    uexpr_cst.append(bias_upper)
                    for j in range(args.num_params):
                        lexpr_dim.append(dim + j)
                        uexpr_dim.append(dim + j)
                        lexpr_weights.append(weights_lower[j])
                        uexpr_weights.append(weights_upper[j])
                else:
                    assert(line == 'SPEC_FINISHED\n')
                    for p_idx in range(args.num_params):
                        lexpr_cst.append(spec_lb[dim + p_idx])
                        for l in range(args.num_params):
                            lexpr_weights.append(0)
                            lexpr_dim.append(dim + l)
                        uexpr_cst.append(spec_ub[dim + p_idx])
                        for l in range(args.num_params):
                            uexpr_weights.append(0)
                            uexpr_dim.append(dim + l)
                    if(is_trained_with_pytorch):
                        normalize(spec_lb[:dim], means, stds, args.dataset, is_conv)
                        normalize(spec_ub[:dim], means, stds, args.dataset, is_conv)
                    normalize_poly(args.num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, args.dataset)

                    for attack_idx, (attack_params, attack_lb, attack_ub) in enumerate(attack_imgs):
                        ok_attack = True
                        for j in range(num_pixels):
                            low, up = lexpr_cst[j], uexpr_cst[j]
                            for idx in range(args.num_params):
                                low += lexpr_weights[j * args.num_params + idx] * attack_params[idx]
                                up += uexpr_weights[j * args.num_params + idx] * attack_params[idx]
                            if low > attack_lb[j] + EPS or attack_ub[j] > up + EPS:
                                ok_attack = False
                        if ok_attack:
                            checked[attack_idx] = True
                            # print('checked ', attack_idx)
                    if args.debug:
                        print('Running the analysis...')

                    t_begin = time.time()
                    perturbed_label_poly, _, _, _ = eran.analyze_box(
                        spec_lb, spec_ub, 'deeppoly',
                        args.timeout_lp, args.timeout_milp, args.use_area_heuristic, 0,
                        lexpr_weights, lexpr_cst, lexpr_dim,
                        uexpr_weights, uexpr_cst, uexpr_dim,
                        expr_size)
                    perturbed_label_box, _, _, _ = eran.analyze_box(
                        spec_lb[:dim], spec_ub[:dim], 'deeppoly',
                        args.timeout_lp, args.timeout_milp, args.use_area_heuristic, 0)
                    t_end = time.time()

                    print('DeepG: ', perturbed_label_poly, '\tInterval: ', perturbed_label_box, '\tlabel: ', label, '[Time: %.4f]' % (t_end - t_begin))

                    tot_chunks += 1
                    if perturbed_label_box != label:
                        ok_box = False
                    else:
                        ver_chunks_box += 1

                    if perturbed_label_poly != label:
                        ok_poly = False
                    else:
                        ver_chunks_poly += 1
                    
                    lexpr_cst, uexpr_cst = [], []
                    lexpr_weights, uexpr_weights = [], []
                    lexpr_dim, uexpr_dim = [], []

        total += 1
        if ok_box:
            correct_box += 1
        if ok_poly:
            correct_poly += 1
        if cex_found:
            assert (not ok_box) and (not ok_poly)
            attacked += 1
        cver_poly.append(ver_chunks_poly / float(tot_chunks))
        cver_box.append(ver_chunks_box / float(tot_chunks))
        tot_time += time.time() - begtime

        print('Verified[box]: {}, Verified[poly]: {}, CEX found: {}'.format(ok_box, ok_poly, cex_found))
        assert not cex_found or not ok_box, 'ERROR! Found counter-example, but image was verified with box!'
        assert not cex_found or not ok_poly, 'ERROR! Found counter-example, but image was verified with poly!'

        print('Attacks found: %.2f percent, %d/%d' % (100.0*attacked/total, attacked, total))
        print('[Box]  Provably robust: %.2f percent, %d/%d' % (100.0*correct_box/total, correct_box, total))
        print('[Poly] Provably robust: %.2f percent, %d/%d' % (100.0*correct_poly/total, correct_poly, total))
        print('Empirically robust: %.2f percent, %d/%d' % (100.0*(total-attacked)/total, total-attacked, total))
        print('[Box]  Average chunks verified: %.2f percent' % (100.0*np.mean(cver_box)))
        print('[Poly]  Average chunks verified: %.2f percent' % (100.0*np.mean(cver_poly)))
        print('Average time: ', tot_time/total)

        
if __name__ == '__main__':
    main()
