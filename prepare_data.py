import argparse
from argparse import ArgumentTypeError
from preprocessing.pre_kuairand import pre_kuairand
from preprocessing.cal_baseline_label import cal_baseline_label
from preprocessing.cal_ground_truth import cal_ground_truth


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="prepare datasets")
    parser.add_argument('-g', '--group_num', type=int, default=30, help="Groups of percentile_label")
    parser.add_argument('-t', '--windows_size', type=int, default=10, help='Windows size of moving average')
    parser.add_argument('-e', '--eps', type=float, default=0.3, help='smooth scale of GMM')
    parser.add_argument('--bias_point', type=float, default=0.15, help='smooth scale of GMM')
    parser.add_argument('--noise_point', type=float, default=10, help='smooth scale of GMM')
    parser.add_argument('--dat_name', type=str, default='KuaiRand', choices=['KuaiRand', 'WeChat', 'KuaiShou2018'])
    parser.add_argument('--is_load', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()

    group_num = args.group_num
    dat_name = args.dat_name
    windows_size = args.windows_size

    print('Load Raw Data...')
    kuairand_dat = pre_kuairand()

    print('Cal Baseline Labels...')
    kuairand_dat = cal_baseline_label(kuairand_dat, group_num, dat_name, windows_size)

    print('Cal Ground Truth Labels...')
    kuairand_dat = cal_ground_truth(kuairand_dat, dat_name)

    kuairand_dat.to_csv('../data/KuaiRand-1K-Process/KuaiRand_subset.csv')
    # todo 可以查看数据
    kuairand_dat.head(100).to_csv('../data/KuaiRand-1K-Process/KuaiRand_subset_head_100.csv')
    print("前10条数据如下", kuairand_dat.head(10))


if __name__ == "__main__":
    main()
