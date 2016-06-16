
import numpy as n
import csv

def calculate_bias_var_err(L):
    # if not os.path.exists(foldername):
    #    print 'no data file, please check it'
    #    return
    # print 'reading data from %s.' % foldername
    g_filename = 'gradient-1/gradient-31-'
    g_L_filename = 'gradient-%d/gradient-31-' % L
    g_hat_L_filename = 'gradient-%d/gradient-1-' % L
    bias = list();
    var = list();
    err = list();

    reclist = n.unique(map(int, filter(lambda x: x < 900, n.logspace(0, 3.35, num=100, base=10.0).tolist()))).tolist()
    for i in reclist :
        with open(g_filename + str(i)) as f:
            g = n.loadtxt(f).T
        with open(g_L_filename + str(i)) as f:
            g_L = n.loadtxt(f).T
        with open(g_hat_L_filename + str(i)) as f:
            g_hat_L = n.loadtxt(f).T

        bias.append(n.sum(n.square(g_L - g)))
        try:
            var.append(n.sum(n.square(g_hat_L - g_L)))
        except:
            print i
        err.append(n.sum(n.square(g_hat_L - g)))

    return (bias, var, err)

def main():
    reclist = n.unique(map(int, filter(lambda x : x < 900, n.logspace(0, 3.35, num=100, base=10.0).tolist()))).tolist()
    for L in [3,10,30,100,300]:
        with open('prop-%d-2nd.csv' % L, 'w') as file:
            wr = csv.writer(file)
            (bias, var, err) = calculate_bias_var_err(L)
            wr.writerow(reclist)
            wr.writerow(bias)
            wr.writerow(var)
            wr.writerow( map (lambda x, y : x +y,  var ,bias) )
            wr.writerow(err)

if __name__ == '__main__':
    main()