import numpy as np
import pylab as pl
import matplotlib

def load_data():
    # x = np.linspace(100, 10, 10)
    x = 100*np.array([1, 0.8201 , 0.6637 , 0.5239 , 0.4037 , 0.3043 , 0.2170 , 0.1509 , 0.1023 , 0.0723])
    y = np.array([
        [0.7118, 0.7023, 0.6962, 0.6793, 0.6666, 0.0796, 0.5818, 0.4895, 0.2836, 0.0190],
        [0.7118, 0.7068, 0.6810, 0.6398, 0.6046, 0.5516, 0.4921, 0.4292, 0.3412, 0.0487],
        [0.7118, 0.7038, 0.7007, 0.6843, 0.6752, 0.6553, 0.5305, 0.4465, 0.3492, 0.0243],
        [0.7118, 0.7011, 0.6909, 0.6814, 0.6628, 0.6381, 0.5832, 0.5140, 0.1854, 0.0091],
        [0.7118, 0.7142, 0.7021, 0.6928, 0.6762, 0.6565, 0.6350, 0.5704, 0.4833, 0.2522],
        [0.7118, 0.7109, 0.7052, 0.6949, 0.6880, 0.6690, 0.6339, 0.6113, 0.5228, 0.3454]
    ])
    return x, y


def main():
    my_font = matplotlib.rcParams.update({'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
    x, y = load_data()

    fig, ax = pl.subplots(1, 1)
    ax.plot(x.reshape(-1), y[0, :].reshape(-1), 'c^-', label='Random')
    ax.plot(x.reshape(-1), y[1, :].reshape(-1), 'gs--', label='Weight sum-$\ell_1$')
    ax.plot(x.reshape(-1), y[2, :].reshape(-1), 'gs-', label='Weight sum-$\ell_2$')
    ax.plot(x.reshape(-1), y[3, :].reshape(-1), 'md-', label='APoZ')
    ax.plot(x.reshape(-1), y[4, :].reshape(-1), 'ro--', label='ThiNet w/o $\mathbf{\hat{w}}$')
    ax.plot(x.reshape(-1), y[5, :].reshape(-1), 'ro-', label='ThiNet')
    pl.xlim(100, 11)  # set axis limits

    ax.set_xticks([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
    ax.set_xticklabels(['100%', '90%', '80%', '70%', '60%', '50%', '40%', '30%','20%', '10%', '0%'])
    ax.legend(loc='best', shadow=True, prop=my_font, fancybox=True)
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax.set_xlabel('FLOPs Reduction')
    ax.set_ylabel('Top-1 Accuracy')

    pl.show()

if __name__ == '__main__':
    main()