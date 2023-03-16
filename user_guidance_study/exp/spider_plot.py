import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

# MSD Spleen
# python exp/spider_plot.py --guidances disks heatmaps exp_geos edt gdt --metric_dirs phase_2/eval_data/best_disks/ phase_2/eval_data/best_heatmaps/ phase_2/eval_data/best_exp_geos/ phase_2/eval_data/best_edt/ phase_2/eval_data/best_gdt/ --data_dirs phase_1/data/disks_sigma\=1_p\=1.0_enc\=cat/ phase_1/data/heatmaps_sigma\=1_p\=1.0_enc\=cat/ phase_1/data/exp_geos_sigma\=5_p\=1.0_enc\=cat/ phase_1/data/edt_sigma\=1_p\=1.0_enc\=cat_gdt-theta\=10/ phase_1/data/gdt_sigma\=5_p\=1.0_enc\=cat_gdt-theta\=30/

# AutoPET
# python exp/spider_plot.py --guidances disks heatmaps exp_geos edt gdt --metric_dirs phase_2/eval_data_autopet/best_disks/ phase_2/eval_data_autopet/best_heatmaps/ phase_2/eval_data_autopet/best_exp_geos/ phase_2/eval_data_autopet/best_edt/ phase_2/eval_data_autopet/best_gdt/ --data_dir phase_1/data_autopet/disks_sigma\=0_p\=1.0_enc\=cat/ phase_1/data_autopet/heatmaps_sigma\=0_p\=1.0_enc\=cat/ phase_1/data_autopet/exp_geos_sigma\=5_p\=1.0_enc\=cat/ phase_1/data_autopet/edt_sigma\=1_p\=1.0_enc\=cat_gdt-theta\=50/ phase_1/data_autopet/gdt_sigma\=5_p\=1.0_enc\=cat_gdt-theta\=10/
def get_accs(data_dirs, args, guidance):


    all_differences = []
    param_it_accs = []
    print(data_dirs)

    for data_dir in [data_dirs]:
        eval_dir = os.path.join(data_dir, 'eval')
        if True:



            if guidance in eval_dir:
                it_accs = []
                best_ind = np.argmax(np.load(f'{eval_dir}/9.npy'))
                for i in range(10):
                    it = np.load(f'{eval_dir}/{i}.npy')
                    if i < 9:
                        truncate = min(len(it), len(np.load(f'{eval_dir}/{i+1}.npy')))
                        differences = np.load(f'{eval_dir}/{i+1}.npy')[:truncate] - it[:truncate]
                        differences = (differences > 0) * 1.0
                        all_differences += list(differences)

                    all_its = len(it)
                    if 'autopet' in eval_dir or True:
                        it = it[(int(all_its / 102) - 1) * 102 : int(all_its / 102) * 102]
                    else:
                        it = it[(int(all_its / 8) - 1) * 8 : int(all_its / 8) * 8]


                    it_accs.append(np.nanmean(it))
                param_it_accs = it_accs

            it_0 = np.load(f'{eval_dir}/0.npy')
            it_9 = np.load(f'{eval_dir}/9.npy')

            all_its = len(it_9)
            if 'autopet' in eval_dir or True:
                it_0 = it_0[(int(all_its / 102) - 1) * 102 : int(all_its / 102) * 102]
                it_9 = it_9[(int(all_its / 102) - 1) * 102 : int(all_its / 102) * 102]
            else:
                it_0 = it_0[(int(all_its / 8) - 1) * 8 : int(all_its / 8) * 8]
                it_9 = it_9[(int(all_its / 8) - 1) * 8 : int(all_its / 8) * 8]

            init_dice = np.nanmean(it_0)
            final_dice = np.nanmean(it_9)
    return param_it_accs, init_dice, final_dice, all_differences


parser = argparse.ArgumentParser(description="Analyze the effect of the param parameter.")

parser.add_argument('--guidances', nargs='+', help='Type of guidance.')
parser.add_argument('--metric_dirs', nargs='+', help='Path to the overlap and time complexity metrics.')
parser.add_argument('--data_dirs', nargs='+', help='Path to intermediate evaluation.')


args = parser.parse_args()



data_dirs = args.data_dirs
metric_dirs = args.metric_dirs
guidances = args.guidances


epoch = 100

rs = []
for (data_dir, metric_dir, guidance) in zip(data_dirs, metric_dirs, guidances):
    param_it_accs, init_dice, final_dice, all_differences = get_accs(data_dir, args, guidance)


    time = np.load(os.path.join(metric_dir, 'time.npy'))
    #ime = np.clip(time, 0, 1)
    #time = 1 - time

    overlap = np.load(os.path.join(metric_dir, 'overlap.npy'))
    print(overlap)
    #differences = np.array([y - x for x,y in zip(param_it_accs, param_it_accs[1:])])
    #consistent_improvement = np.sum(differences > 0) / len(differences)

    consistent_improvement = np.sum(all_differences) / len(all_differences)

    print(len(time), len(overlap))
    time = np.mean(time)
    overlap = np.nanmean(overlap)

    print(init_dice, final_dice, time, overlap, consistent_improvement)
    exit()
    print(param_it_accs)

    '''
    df = pd.DataFrame(dict(
        r=[init_dice, final_dice, time, overlap, consistent_improvement],
        theta=['Initial Dice','Final Dice','Time in Seconds',
               'Overlap with Ground Truth', 'Consistent Improvement']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    '''
    categories = ['Initial\nDice','Final\nDice','Efficiency',
           'Overlap with\nGround Truth', 'Consistent\nImprovement', 'Initial\nDice']

    '''
    fig.add_trace(go.Scatterpolar(
          r=[init_dice, final_dice, time, overlap, consistent_improvement],
          theta=categories,
         # fill='toself',
          name=guidance
    ))
    '''
    rs.append([init_dice, final_dice, time, overlap, consistent_improvement, init_dice])
title = 'AutoPET' if 'autopet' in data_dirs[0] else 'MSD Spleen'

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(rs[0]))

guidances_mapping = {'disks': 'Disks', 'heatmaps': 'Heatmaps', 'exp_geos': 'Exp. Geodesic Maps', 'gdt': 'Geodesic Maps', 'edt': 'Euclidean Maps', 'adaptive': 'Adaptive Heatmaps'}

if 'autopet' in data_dirs[0]:
    min_autopet = [0.2, 0.25, -0.1, 0.0, 0.1, 0.2]
    max_autopet = [0.6, 0.7, 0.4, 1.0, 0.8, 0.6]
else:
    min_autopet = [0.0, 0.0, -0.1, 0.0, 0.0, 0.0]
    max_autopet = [1.0, 1.0, 0.8, 1.0, 1.0, 1.0]
plt.figure(figsize=(8, 8))
plt.subplot(polar=True)
for i in range(1):
    rsx = rs[i]
    #for j, el in enumerate(rsx):
    #        rsx[j] = (rsx[j] - min_autopet[j]) / max_autopet[j]
    print(rsx)
    plt.plot(label_loc, rsx, label=guidances_mapping[guidances[i]], marker='.', linewidth=2)

plt.title(title, size=20, y=1.05)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, fontsize=0)
#plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylim(0.3, 1.0)


#legend = plt.legend(fontsize=14, loc=(-0.2,0.9))
plt.tight_layout()

''' plotly
fig = go.Figure(
    data=[
        go.Scatterpolar(r=rs[0], theta=categories, name=guidances[0]),
        go.Scatterpolar(r=rs[1], theta=categories, name=guidances[1]),
        go.Scatterpolar(r=rs[2], theta=categories, name=guidances[2]),
        go.Scatterpolar(r=rs[3], theta=categories, name=guidances[3]),
        go.Scatterpolar(r=rs[4], theta=categories, name=guidances[4]),

    ],
    layout=go.Layout(
        title=go.layout.Title(text=title, font={'size':36}),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
)
'''
import matplotlib.pyplot as plt
import numpy as np

def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

#export_legend(legend)

if 'autopet' in args.data_dirs[0]:
    #fig.write_image(f'plots/spider_plot_autopet.pdf')
    plt.savefig('plots/spider_plot_autopet.pdf')
else:
    #fig.write_image(f'plots/spider_plot.pdf')
    plt.savefig(f'plots/spider_plot.pdf')
#plt.ylim(0.92,
