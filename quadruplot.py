"""# quadruplot

Plot quadrupole deformed spheroid using the parameters β and γ.

```
Usage:
    quadruplot NAME [--beta VAL] [--gamma VAL] [--frames INT] [--delay INT]
    quadruplot (-h | --help)
    quadruplot --version
    quadruplot --license

Options:
    --beta VAL      Range for β. Can be either a number or a range
                    formatted as MIN,MAX. [default: 0.3]
    --gamma VAL     Range for γ in degrees. Can be either a number or a
                    range formatted as MIN,MAX. [default: 0]
    --frames INT    Number of frames to generate.
    --delay INT     Delay between the frames / 10 ms. [default: 15]
    --version       Print version number.
    --license       Print license information.
```
"""
__copying__ = """
Copyright Johannes Sørby Heines 2025
email: johannes.sorby.heines@alumni.cern

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
__version__ = 0.3
# TODO: Better defaults: delat as f(frames)
# TODO: format and single value β γ options

from functools import singledispatch
from math import sin, cos, sqrt, pi
from pathlib import Path
import subprocess
import sys

from attrs import define, field
from docopt import docopt
import cmcrameri as cmc
from matplotlib import pyplot as plt
import numpy as np
#from numpy.typing import NDArray
from tqdm import tqdm

def main():
    args = docopt(__doc__, version=__version__)
    if args['--license']:
        print(__copying__)
        sys.exit()

    name = args['NAME']
    beta, gamma = make_params(args['--beta'], args['--gamma'], args['--frames'])
    delay = int(args['--delay'])

    draw(beta, gamma, name, delay)

@singledispatch
def draw(beta, gamma, **kwargs):
    raise TypeError("beta and gamma should be either float or list")

@draw.register
def _(betas: list, gammas: list, name: str, delay: int):
    frames = len(betas)
    maxframenumbersize = int(np.log10(frames)) + 1

    q = Quadrupole(0, 0)
    fignames = []
    print("Making frames", file=sys.stderr)
    for i, (beta, gamma) in tqdm(enumerate(zip(betas, gammas)), total=frames):
        q.beta = beta
        q.gamma = gamma
        ax = q.plot(cmap='cmc.lipari')
        fignames.append(f'{name}{i:0>{maxframenumbersize}d}.png')
        plt.savefig(fignames[-1], format='png', dpi=200, transparent=True)
        plt.close()

    print("Making gif")
    subprocess.run([
        'magick',
        '-delay', f'{delay}',
        '-loop', '0',
        '-dispose', 'background',
        *fignames, f'{name}.gif'
        ])

    for figname in fignames:
        Path(f'{figname}').unlink()

@draw.register
def _(beta: float, gamma: float, name: str, _):
    q = Quadrupole(beta, gamma)
    q.plot(cmap='cmc.lipari')
    plt.savefig(f'{name}.pdf', format='pdf')
    plt.show()

def make_params(beta_opt, gamma_opt, frame_opt):
    if frame_opt is not None:
        frames = int(frame_opt)
    else:
        if ',' in beta_opt or ',' in gamma_opt:
            frames = 30
        else:
            frames = 1

    if frames == 1:
        beta = float(beta_opt)
        gamma = float(gamma_opt)
        return beta, gamma  # returning scalars

    loopmap = 0.5*np.sin(np.linspace(0, 2*pi, frames))
    if ',' in beta_opt:
        betamin, betamax = [float(v) for v in beta_opt.split(',')]
        betas = (betamax - betamin)*loopmap + (betamax + betamin)/2
    else:
        betas = float(beta_opt)*np.ones(frames)
    if ',' in gamma_opt:
        gammamin, gammamax = [float(v)*pi/180 for v in gamma_opt.split(',')]
        if (gammamax - gammamin)%(2*pi) < 1e-5: # full loop
            gammas = np.linspace(gammamin, gammamax, frames)
        else:
            gammas = (gammamax - gammamin)*loopmap + (gammamax + gammamin)/2
    else:
        gammas = float(gamma_opt)*np.ones(frames)

    return list(betas), list(gammas)  # returning lists


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
            fig = plt.figure(figsize=(6, 6), layout='constrained')
            ax = fig.add_subplot(projection='3d')
        ax.plot_surface(*self.surface(100), **kwargs)
        ax.set_aspect('equal')
        ax.set_axis_off()
        return ax


if __name__ == '__main__':
    main()
