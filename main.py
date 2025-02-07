"""quadruplot

Usage:
    quadruplot <name> [--beta=<min,max>] [--gamma=<min,max>] 
        [--frames=<int>] [--delay=<int>]
    quadruplot (-h | --help)

Options:
    --beta=<min,max>    Range for β. [default: 0.3,0.3]
    --gamma=<min,max>   Range for γ in degrees. [default: 0,0]
    --frames=<int>      Number of frames to generate. [default: 1]
    --delay=<int>       Delay between the frames / 10 ms. [default: 15]
"""
# TODO: Better defaults: delat as f(frames)
# TODO: format and single value β γ options

from math import sin, cos, sqrt, pi
from pathlib import Path
import subprocess

from attrs import define, field
from docopt import docopt
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    args = docopt(__doc__)

    name = args['<name>']
    betamin, betamax = [float(v) for v in args['--beta'].split(',')]
    gammamin, gammamax = [float(v)*pi/180 for v in args['--gamma'].split(',')]
    frames = int(args['--frames'])
    delay = int(args['--delay'])

    maxframenumbersize = int(np.log10(frames)) + 1

    loopmap = 0.5*np.sin(np.linspace(0, 2*pi, frames))
    betas = (betamax - betamin)*loopmap + (betamax + betamin)/2
    if (gammamax - gammamin)%(2*pi) < 1e-5: # full loop
        gammas = np.linspace(gammamin, gammamax, frames)
    else:
        gammas = (gammamax - gammamin)*loopmap + (gammamax + gammamin)/2

    q = Quadrupole(0, 0)
    fignames = []
    for i, (beta, gamma) in tqdm(enumerate(zip(betas, gammas)), total=frames):
        q.beta = beta
        q.gamma = gamma
        ax = q.plot()
        fignames.append(f'{name}{i:0>{maxframenumbersize}d}.png')
        plt.savefig(fignames[-1], format='png')
        plt.close()

    if frames == 1:
        return

    subprocess.run(
        ['magick', '-delay', '15', '-loop', '0', f'{name}*.png', f'{name}.gif']
        )

    for figname in fignames:
        Path(f'{figname}').unlink()


@define
class Quadrupole:
    beta: float
    gamma: float

    @property
    def semi_axes(self):
        betaterm = sqrt(5/4/pi)*self.beta
        cosgammaterm = 0.5*cos(self.gamma)
        singammaterm = sqrt(3)/2*sin(self.gamma)
        xyz = np.array([
            1 + betaterm*(-cosgammaterm + singammaterm),
            1 + betaterm*(-cosgammaterm - singammaterm),
            1 + betaterm*2*cosgammaterm
            ])
        return xyz

    def surface(self, npoints):
        theta = np.linspace(0, 2*pi, npoints)
        phi = np.linspace(0, pi, npoints)
        rx, ry, rz = self.semi_axes
        x = rx * np.outer(np.cos(theta), np.sin(phi))
        y = ry * np.outer(np.sin(theta), np.sin(phi))
        z = rz * np.outer(np.ones_like(theta), np.cos(phi))
        return np.array([x, y, z])

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.plot_surface(*self.surface(100), **kwargs)
        ax.set_aspect('equal')
        ax.set_axis_off()
        return ax


if __name__ == '__main__':
    main()
