from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(20, 11))
ax.set_xlim(0, 2000)
ax.set_ylim(1100, 0)
ax.axis('off')
fig.subplots_adjust(0, 0, 1, 1)
BG = '#dce5f3'
NN = '#c9c9c9'
INK = '#14191f'

def text(x, y, label, size=12, weight='normal', **kw):
    ax.text(x, y, label, ha='center', va='center', fontsize=size,
            fontweight=weight, color=INK, **kw)

def box(x, y, w, h, label, fill='white', size=12, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK, linewidth=1.25, zorder=3))
    text(x+w/2, y+h/2, label, size, 'bold' if bold else 'normal', zorder=4)

def line(points, arrow=True, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=INK, lw=1.25, ls='--' if dashed else '-', zorder=2)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                     mutation_scale=13, linewidth=1.25, color=INK, zorder=2))

def panel(y, title):
    ax.add_patch(FancyBboxPatch((520, y), 1450, 238, boxstyle='round,pad=0,rounding_size=28',
                              linewidth=0, facecolor=BG, zorder=0))
    text(1245, y+25, title, 16)

def front(y):
    cy=y+100
    line([(485, cy), (555, cy)])
    box(555, y+65, 100, 70, 'CP\nremoval')
    line([(655, cy), (685, cy)])
    box(685, y+65, 90, 70, 'FFT')
    return cy

def end(y, source):
    cy=y+100
    line([(source,cy), (1790,cy)])
    text(1750,cy-17,'LLRs',11)
    box(1790,y+65,115,70,'Channel\ndecoding')
    line([(1905,cy),(1947,cy)])
    text(1926,cy-18,'Bits',10)

def estimate(y, label, target, raw=False):
    box(555,y+168,220,47,'DMRS configuration',size=11)
    box(845,y+168,270,47,label,size=11)
    line([(775,y+191.5),(845,y+191.5)])
    line([(800,y+100),(800,y+180),(845,y+180)])
    line([(target,y+168),(target,y+135)])

# A single transmitter and propagation channel supply all four alternatives.
ax.add_patch(FancyBboxPatch((30,285),220,480,boxstyle='round,pad=0,rounding_size=25',
                          facecolor=BG,linewidth=0))
text(140,315,'Transmitter (Tx)',16)
text(140,353,'Bits',12)
line([(140,366),(140,390)])
box(60,390,160,65,'Channel coding\n& QAM mapping',size=11)
line([(140,455),(140,485)])
box(60,485,160,65,'Resource mapping\n& DMRS insertion',size=11)
line([(140,550),(140,580)])
box(60,580,160,65,'IFFT\n& CP insertion',size=11)
line([(140,645),(140,675)])
box(60,675,160,60,'PA',fill=NN,bold=True)
line([(220,705),(285,705)])
box(285,670,150,70,'Channel\n+ AWGN')
line([(435,705),(485,705)],arrow=False)
line([(485,150),(485,930)],arrow=False)
text(367,632,'Shared link',12)
text(367,776,'Same received\nwaveform',11)

y=50
panel(y,'Conventional Rx (FEQ)')
cy=front(y)
line([(775,cy),(905,cy)])
box(905,y+65,150,70,'Equalization')
line([(1055,cy),(1330,cy)])
box(1330,y+65,170,70,'Demapper')
estimate(y,'Channel estimation\n& interpolation',980)
end(y,1500)

y=570
panel(y,'DeepRx (FCNN)')
cy=front(y)
line([(775,cy),(905,cy)])
box(905,y+65,740,70,'Frequency-domain CNN',fill=NN,bold=True,size=14)
estimate(y,'Raw channel estimation',980,True)
end(y,1645)

y=830
panel(y,'Frequency-domain HybridDeepRx (FTF-CNN)')
cy=front(y)
blocks=[(905,130,'FCNN',NN),(1070,100,'IFFT','white'),(1205,130,'TCNN',NN),
        (1370,100,'FFT','white'),(1505,140,'FCNN',NN)]
last=775
for x,w,label,fill in blocks:
    line([(last,cy),(x,cy)])
    box(x,y+65,w,70,label,fill=fill,bold=fill==NN)
    last=x+w
estimate(y,'Raw channel estimation',980,True)
end(y,last)

y=310
panel(y,'Digital Post-Inverse (DPoI)')
cy=front(y)
blocks=[(845,150,'LMMSE\nequalization','white'),(1025,100,'IFFT','white'),
        (1155,210,'Memory-polynomial\npost-inverse',NN),(1395,90,'FFT','white'),
        (1515,130,'Demapper','white')]
last=775
for x,w,label,fill in blocks:
    line([(last,cy),(x,cy)])
    box(x,y+65,w,70,label,fill=fill,size=11,bold=fill==NN)
    last=x+w
box(555,y+168,220,47,'DMRS configuration',size=11)
box(820,y+168,220,47,'LS channel estimation\n& delay-domain smoothing',size=10)
line([(775,y+191.5),(820,y+191.5)])
line([(800,cy),(800,y+180),(820,y+180)])
line([(920,y+168),(920,y+135)])
box(1155,y+168,210,47,'Pilot-based coefficient fit',size=10)
line([(775,y+203),(790,y+203),(790,y+226),(1260,y+226),(1260,y+215)],dashed=True)
line([(1075,y+135),(1075,y+191.5),(1155,y+191.5)],dashed=True)
line([(1260,y+168),(1260,y+135)],dashed=True)
end(y,last)

for cy in [150,410,670,930]:
    ax.plot(485,cy,'o',color=INK,ms=3,zorder=5)

for ext in ('svg','pdf','png'):
    fig.savefig(OUT/f'receiver_architectures.{ext}',dpi=180,facecolor='white')
plt.close(fig)
