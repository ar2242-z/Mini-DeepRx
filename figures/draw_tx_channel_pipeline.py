from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(20, 6.2))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 2000)
ax.set_ylim(620, 0)
ax.axis('off')
INK, BG, GREY = '#14191f', '#dce5f3', '#c9c9c9'

def label(x, y, value, size=14, bold=False):
    ax.text(x, y, value, ha='center', va='center', fontsize=size,
            color=INK, fontweight='bold' if bold else 'normal', zorder=4)

def panel(x, w, title):
    ax.add_patch(FancyBboxPatch((x, 35), w, 550,
        boxstyle='round,pad=0,rounding_size=28', facecolor=BG, linewidth=0))
    label(x+w/2, 75, title, 19)

def box(x, w, value, grey=False):
    ax.add_patch(Rectangle((x, 150), w, 90, facecolor=GREY if grey else 'white',
                           edgecolor=INK, linewidth=1.25, zorder=3))
    label(x+w/2, 195, value, 14, grey)

def arrow(x1, x2, value=None):
    ax.add_patch(FancyArrowPatch((x1,195), (x2,195), arrowstyle='-|>',
                 mutation_scale=14, linewidth=1.25, color=INK, zorder=2))
    if value:
        label((x1+x2)/2, 126, value, 14)

panel(30, 1370, 'Transmitter (Tx)')
panel(1430, 540, 'Channel')
label(72, 126, 'Bits', 13)
arrow(45, 100)
box(100, 210, 'Channel coding\n& QAM mapping')
arrow(310, 350)
box(350, 230, 'Resource mapping\n& DMRS insertion')
arrow(580, 670, r'$X_k$')
box(670, 150, 'IFFT')
arrow(820, 910, r'$x[n]$')
box(910, 170, 'CP insertion')
arrow(1080, 1190)
box(1190, 140, 'PA', True)
arrow(1330, 1510)
label(1420, 126, r'$s[n]$', 16)
box(1510, 330, 'Multipath channel\n+ AWGN')
arrow(1840, 1940, r'$y[n]$')

# Equations use the user's symbols; CP operation is indicated in the pipeline.
label(385, 326, 'OFDM modulation', 15)
label(385, 397,
      r'$x[n]=\frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}X_k e^{j2\pi kn/N}$', 22)
label(385, 473, r'$n=0,\ldots,N-1.$', 18)

label(1040, 326, 'PA memory-polynomial model', 15)
ax.text(1040, 415,
        r'$s[n]=\sum_{m=0}^{M}\ \sum_{p=1,\ p\ \mathrm{odd}}^{P}'
        r'a_{p,m}\,x[n-m]\left|x[n-m]\right|^{p-1}$',
        ha='center', va='center', fontsize=19, color=INK,
        bbox=dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor=INK, linewidth=1.25))
label(1040, 521, 'PA input uses the CP-extended waveform.', 11)

label(1700, 326, 'Received waveform', 15)
label(1700, 415,
      r'$y[n]=\sum_{\ell=0}^{L-1}h_\ell s[n-\ell]+w[n].$', 20)

for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT/f'tx_channel_pipeline.{ext}', dpi=180, facecolor='white')
plt.close(fig)
