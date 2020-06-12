"""
    @date 19.05.2020
    It builds the series data to run statistical analysis. Analysis of val-loss and ABX scores correlation
"""
import glob
import json
import os
import re
from matplotlib import pyplot
from matplotlib import patches
import numpy as np
import scipy.stats as st


def get_val_epoch_pairs(folder, language):
    models = glob.iglob(os.path.join(folder, language + '*.h5'))

    pairs = []
    for model in models:
        file_name = os.path.split(model)[1]
        # <epoch>_<val-loss>
        data = re.search('\d+_\d+\.\d+', file_name)[0].split('_')
        epoch = int(data[0])
        val_loss = float(data[1])

        pairs.append((epoch, val_loss))

    return pairs


def read_abx_json(path, language):
    with open(path) as f:
        measures = json.load(f)['2017-track1'][language]['1s']
    return (measures['across'], measures['within'])


def get_abx_epoch_pairs(folder, language, series):
    scores_files = glob.iglob(os.path.join(folder, language + '_' + str(series) + '_*.json'))

    pairs = []
    for score_file in scores_files:
        epoch = int(re.search('_\d+\.', score_file)[0].replace('_', '').replace('.', ''))
        scores = read_abx_json(score_file, language)

        pairs.append((epoch, scores))

    return pairs


def get_data_series(models_folder, eval_folder, language, model_name, system=None, total_series=None):
    if system is not None:
        systems = [system]
    else:
        systems = ['diff', 'mix']

    model_diff_folder = os.path.join(models_folder, 'diff')
    model_mix_folder = os.path.join(models_folder, 'mix')

    eval_diff_folder = os.path.join(eval_folder, 'diff')
    eval_mix_folder = os.path.join(eval_folder, 'mix')

    data = {}
    for sys in systems:
        data[sys] = {}

    if total_series is None:
        # default runs
        total_series = 3

    for sys in systems:
        for series in range(1, total_series + 1):
            if sys == 'mix':
                val_loss = get_val_epoch_pairs(os.path.join(model_mix_folder, str(series)), language)
                abx_score = get_abx_epoch_pairs(eval_mix_folder, language, series)
                val_loss.sort(key=lambda t: t[0])
                abx_score.sort(key=lambda t: t[0])
            else:
                val_loss = get_val_epoch_pairs(os.path.join(model_diff_folder, str(series)), language)
                abx_score = get_abx_epoch_pairs(eval_diff_folder, language, series)
                val_loss.sort(key=lambda t: t[0])
                abx_score.sort(key=lambda t: t[0])

            data[sys][series] = []

            # (epoch, val_los, abx_across, abx_within)
            for idx in range(len(val_loss)):
                epoch = val_loss[idx][0]
                val_loss_value = val_loss[idx][1]
                abx_across_value = abx_score[idx][1][0]
                abx_within_value = abx_score[idx][1][1]
                data[sys][series].append((epoch, val_loss_value, abx_across_value, abx_within_value))

    with open(os.path.join('../statistical_analysis', model_name + '_' + language + '_data.json'), 'w') as f:
        json.dump(data, f, indent=2)

    return data


def get_scatter_plots(data, x_idx, y_idx, file_name, title=None, system=None):
    colours = ['#1167B1', '#B73229']
    if system is not None:
        systems = [system]
    else:
        systems = data.keys()

    for system in systems:
        for series in range(1, len(data[system]) + 1):
            colour = '#1167B1' if system == 'mix' else '#B73229'
            # colour = colours[series-1]
            marker = ['+', '.', '*']
            alpha = [1, 0.8, 0.4]
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[y_idx] for x in data[system][series]],
                           c=colour, marker=marker[series-1], alpha=alpha[series-1])

    pyplot.rcParams['legend.fontsize'] = 11
    pyplot.grid(True, which='both', linestyle='--')

    if system is None:
        mix = patches.Patch(color='#1167B1', label='Split B')
        diff = patches.Patch(color='#B73229', label='Split A')
        pyplot.legend(handles=[mix, diff])

    labels = {0: 'Epoch', 1: 'Validation loss (InfoNCE)', 2: 'ABX across-speaker', 3: 'ABX within-speaker'}
    pyplot.ylabel(labels[y_idx], fontsize=11)
    pyplot.xlabel(labels[x_idx], fontsize=11)
    if not title:
        title = file_name.replace('_', ' ').replace('.png', '')
    # pyplot.title(title)
    # pyplot.show()
    pyplot.tight_layout()
    pyplot.savefig(os.path.join('../statistical_analysis', file_name))
    pyplot.close('all')


def get_scatter_plots_both(data, x_idx, file_name):
    colours = ['#1167B1', '#B73229', '#2A9DF4', '#F66257']
    for system in data:
        for series in range(1, len(data[system]) + 1):
            #colour = 'blue' if system == 'mix' else 'red'
            marker = ['+', '.', '*']
            alpha = [1, 0.8, 0.4]
            colour = 0 if system == 'mix' else 1
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[2] for x in data[system][series]],
                           c=colours[colour], marker=marker[series-1], alpha=alpha[series-1])
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[3] for x in data[system][series]],
                           c=colours[colour + 2], marker=marker[series-1], alpha=alpha[series-1])

    pyplot.rcParams['legend.fontsize'] = 11
    pyplot.grid(True, which='both', linestyle='--')
    mix = patches.Patch(color='#1167B1', label='Split B - ABX across-speaker')
    mix2 = patches.Patch(color='#2A9DF4', label='Split B - ABX within-speaker')
    diff = patches.Patch(color='#B73229', label='Split A - ABX across-speaker')
    diff2 = patches.Patch(color='#F66257', label='Split A - ABX within-speaker')
    pyplot.legend(handles=[mix, mix2, diff, diff2])

    labels = {0: 'Epoch', 1: 'Validation loss', 2: 'ABX across-speaker', 3: 'ABX within-speaker'}
    pyplot.ylabel('ABX scores', fontsize=11)
    pyplot.xlabel(labels[x_idx], fontsize=11)

    # pyplot.show()
    pyplot.tight_layout()
    pyplot.savefig(os.path.join('../statistical_analysis', file_name))
    pyplot.close('all')


def get_scatter_plots_two_lang(data, data2, x_idx, file_name, lang1, lang2):
    colours = ['#1167B1', '#B73229', '#2A9DF4', '#F66257']
    for system in ['mix']:
        for series in range(1, len(data[system]) + 1):
            marker = ['+', '.', '*']
            alpha = [1,0.8,0.4]
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[2] for x in data[system][series]],
                           c=colours[0], marker=marker[series-1], alpha=alpha[series-1])
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[3] for x in data[system][series]],
                           c=colours[2], marker=marker[series-1], alpha=alpha[series-1])
            pyplot.scatter([x[x_idx] for x in data2[system][series]], [x[2] for x in data2[system][series]],
                           c=colours[1], marker=marker[series-1], alpha=alpha[series-1])
            pyplot.scatter([x[x_idx] for x in data2[system][series]], [x[3] for x in data2[system][series]],
                           c=colours[3], marker=marker[series-1], alpha=alpha[series-1])

    # Config
    pyplot.rcParams['legend.fontsize']=11
    pyplot.grid(True, which='both', linestyle='--')

    mix = patches.Patch(color='#1167B1', label=lang1 + ' - ABX across-speaker')
    mix2 = patches.Patch(color='#2A9DF4', label=lang1 + ' - ABX within-speaker')
    diff = patches.Patch(color='#B73229', label=lang2 + ' - ABX across-speaker')
    diff2 = patches.Patch(color='#F66257', label=lang2 + ' - ABX within-speaker')
    pyplot.legend(handles=[mix, mix2, diff, diff2])
    labels = {0: 'Epoch', 1: 'Validation loss', 2: 'ABX across-speaker', 3: 'ABX within-speaker'}
    pyplot.ylabel('ABX scores', fontsize=11)
    pyplot.xlabel(labels[x_idx], fontsize=11)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join('../statistical_analysis', file_name), rasterized=True)
    pyplot.close('all')


def get_scatter_plots_one_lang(data, x_idx, file_name, lang1):
    colours = ['#1167B1', '#B73229', '#2A9DF4', '#F66257']
    colour_idx = 0 if lang1 == 'French' else 1
    for system in ['mix']:
        for series in range(1, len(data[system]) + 1):
            marker = ['+', '.', '*']
            alpha = [1, 0.8, 0.4]
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[2] for x in data[system][series]],
                           c=colours[0 + colour_idx], marker=marker[series-1], alpha=alpha[series-1])
            pyplot.scatter([x[x_idx] for x in data[system][series]], [x[3] for x in data[system][series]],
                           c=colours[2 + colour_idx], marker=marker[series-1], alpha=alpha[series-1])

    pyplot.rcParams['legend.fontsize'] = 11
    pyplot.grid(True, which='both', linestyle='--')

    legend = patches.Patch(color='#1167B1', label=lang1 + ' - ABX across-speaker')
    legend2 = patches.Patch(color='#2A9DF4', label=lang1 + ' - ABX within-speaker')
    pyplot.legend(handles=[legend, legend2])

    labels = {0: 'Epoch', 1: 'Validation loss (MAE)', 2: 'ABX across-speaker', 3: 'ABX within-speaker'}
    pyplot.ylabel('ABX scores', fontsize=11)
    pyplot.xlabel(labels[x_idx], fontsize=11)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join('../statistical_analysis', file_name))
    pyplot.close('all')


def get_t_significance_r(r, n):
    t = (r * np.sqrt(n - 2)) / np.sqrt(1 - (r * r))
    return t


def get_t_significance_rs(rs, n):
    # For n>100
    t = rs / np.sqrt((1 - (rs * rs)) / (n - 2))
    return t


def fishers_transformation(r):
    z = 1 / 2 * np.log((1 + r) / (1 - r))
    return z


def calculate_z_obs(z1, z2, n):
    z_obs = (z1 - z2) / np.sqrt(2 / (n - 3))
    return z_obs


def statistical_analysis(data, alpha, language, model):
    means = {'mix': [], 'diff': []}
    conf_interval = {'mix': {}, 'diff': {}}
    correlation = {'mix': {}, 'diff': {}}
    best_models = {'mix': [], 'diff': []}
    t_test = {'abx_across': 0, 'abx_within': 0}
    correlation_series = {'mix': {}, 'diff': {}}
    std = {'mix':{}, 'diff':{}}
    n = 10  # total samples

    for system in data:
        for idx_epoch in range(len(data[system][1])):
            items_epoch = [data[system][s][idx_epoch] for s in data[system]]
            epoch = items_epoch[0][0]
            val_loss = [x[1] for x in items_epoch]
            abx_across = [x[2] for x in items_epoch]
            abx_within = [x[3] for x in items_epoch]
            means[system].append((epoch, np.mean(val_loss), np.mean(abx_across), np.mean(abx_within)))

            # Confidence Interval

            conf_interval[system][epoch] = {'val_loss': st.t.interval(0.95, len(val_loss), loc=np.mean(val_loss),
                                                                      scale=st.sem(val_loss)),
                                            'abx_across': st.t.interval(0.95, len(abx_across), loc=np.mean(abx_across),
                                                                        scale=st.sem(abx_across)),
                                            'abx_within': st.t.interval(0.95, len(abx_within), loc=np.mean(abx_within),
                                                                        scale=st.sem(abx_within))}

        tmp_correlations = {'abx_across_p': [], 'abx_across_s': [], 'abx_within_p': [], 'abx_within_s': []}

        all_abx_across = []
        all_abx_within = []
        for series in data[system]:
            best_models[system].append(sorted(data[system][series], key=lambda t: t[1], reverse=False)[0])
            val_loss = [x[1] for x in data[system][series]]
            abx_across = [x[2] for x in data[system][series]]
            abx_within = [x[3] for x in data[system][series]]

            std[system][series] = {
                'abx_across': np.std(abx_across),
                'abx_within': np.std(abx_within)
            }
            all_abx_across += abx_across
            all_abx_within += abx_within

            correlation_series[system][series] = {
                'abx_across': {'pearson': st.pearsonr(val_loss, abx_across),
                               'spearman': st.spearmanr(val_loss, abx_across)},
                'abx_within': {'pearson': st.pearsonr(val_loss, abx_within),
                               'spearman': st.spearmanr(val_loss, abx_within)}
            }
            tmp_correlations['abx_across_p'].append(correlation_series[system][series]['abx_across']['pearson'][0])
            tmp_correlations['abx_across_s'].append(correlation_series[system][series]['abx_across']['spearman'][0])
            tmp_correlations['abx_within_p'].append(correlation_series[system][series]['abx_within']['pearson'][0])
            tmp_correlations['abx_within_s'].append(correlation_series[system][series]['abx_within']['spearman'][0])
        std[system]['total'] = {
            'abx_across': np.std(all_abx_across),
            'abx_within': np.std(all_abx_within)
        }

        correlation_series[system]['conf_interval'] = {
            'abx_across': {'pearson': st.t.interval(0.95, 3, loc=np.mean(tmp_correlations['abx_across_p']),
                                                    scale=st.sem(tmp_correlations['abx_across_p'])),
                           'spearman': st.t.interval(0.95, 3, loc=np.mean(tmp_correlations['abx_across_s']),
                                                     scale=st.sem(tmp_correlations['abx_across_s']))},
            'abx_within': {'pearson': st.t.interval(0.95, 3, loc=np.mean(tmp_correlations['abx_within_p']),
                                                    scale=st.sem(tmp_correlations['abx_within_p'])),
                           'spearman': st.t.interval(0.95, 3, loc=np.mean(tmp_correlations['abx_within_s']),
                                                     scale=st.sem(tmp_correlations['abx_within_s']))}
        }

        # Correlation Coefficient (Pearson and Spearman) & significance of correlation coefficient
        # val_loss vs abx_across
        val_means = [x[1] for x in means[system]]
        abx_across_means = [x[2] for x in means[system]]
        abx_within_means = [x[3] for x in means[system]]

        correlation[system]['abx_across'] = {'pearson': st.pearsonr(val_means, abx_across_means),
                                             'spearman': st.spearmanr(val_means, abx_across_means)}
        correlation[system]['abx_within'] = {'pearson': st.pearsonr(val_means, abx_within_means),
                                             'spearman': st.spearmanr(val_means, abx_within_means)}

        n = len(val_means)
        for abx in ['abx_across', 'abx_within']:
            correlation[system][abx]['pearson_t'] = get_t_significance_r(
                correlation[system][abx]['pearson'][0], n)
            correlation[system][abx]['spearman_t'] = correlation[system][abx]['spearman'][0]

    # t-test ABX averaged performance
    t_test['abx_across'] = st.ttest_ind([x[2] for x in best_models['mix']], [x[2] for x in best_models['diff']],
                                        equal_var=False)
    t_test['abx_within'] = st.ttest_ind([x[3] for x in best_models['mix']], [x[3] for x in best_models['diff']],
                                        equal_var=False)
    t_test['p_critical'] = (1 - alpha)

    # Z_obs analysis
    critical_t = np.abs(st.t.ppf((1 - alpha), n - 2))
    critical_t_spearman = 0.678
    critical_z = st.norm.ppf(1 - ((1 - alpha) / 2))
    z_obs = {'abx_across': {'pearson': 0, 'spearman': 0}, 'abx_within': {'pearson': 0, 'spearman': 0},
             'critical_value': critical_z, 'critical_t_value': critical_t}

    for abx in ['abx_across', 'abx_within']:
        coeff = {'pearson': [], 'spearman': []}
        for rho in ['pearson', 'spearman']:
            for system in data:
                r, p = correlation[system][abx][rho]
                t = correlation[system][abx][rho + '_t']
                # Confirm significance of correlation
                critical_value = critical_t if rho == 'pearson' else critical_t_spearman
                if np.abs(t) > critical_value and np.abs(p) <= (1 - alpha):
                    coeff[rho].append(r)
            if len(coeff[rho]) == 2:
                z1 = fishers_transformation(coeff[rho][0])
                z2 = fishers_transformation(coeff[rho][1])
                z_obs[abx][rho] = calculate_z_obs(z1, z2, n)

    analysis = {'means': means, 'std': std, 'conf_interval': conf_interval, 'correlation': correlation,
                'correlation_series': correlation_series, 'best_models': best_models,
                't_test': t_test, 'z_obs': z_obs}

    with open(os.path.join('../statistical_analysis', model + '_' + language + '_stats.json'), 'w') as f:
        json.dump(analysis, f, indent=2)

    return analysis

def analyse_change_ratio(data_file, language, model):
    figures = {}
    with open(data_file) as f:
        data = json.load(f)
        for system in data:
            figures[system] = {}
            for series in data[system]:
                figures[system][series] = []
                ratio_val_loss = []
                ratio_across_spkr = []
                ratio_within_spkr = []
                for i in range(len(data[system][series])-1):
                    ratio_val_loss.append(1-(data[system][series][i+1][1] / data[system][series][i][1]))
                    ratio_across_spkr.append(1 - (data[system][series][i + 1][2] / data[system][series][i][2]))
                    ratio_within_spkr.append(1 - (data[system][series][i + 1][3] / data[system][series][i][3]))
                figures[system][series].append(ratio_val_loss)
                figures[system][series].append(ratio_across_spkr)
                figures[system][series].append(ratio_within_spkr)
    # plots
    for system in figures:
        for series in figures[system]:
            pyplot.scatter([x for x in range(len(figures[system][series][0]))], figures[system][series][1], c='red')
            pyplot.scatter([x for x in range(len(figures[system][series][0]))], figures[system][series][2], c='blue')

    across = patches.Patch(color='red', label='across-speaker')
    within = patches.Patch(color='blue', label='within-speaker')
    pyplot.legend(handles=[across, within])
    pyplot.xlabel('steps')
    pyplot.ylabel('ratio val-loss')
    pyplot.title('Ratio changes ' + model + ' ' + language)

    # pyplot.show()
    pyplot.savefig(data_file.replace('data', 'ratio_changes_steps').replace('json', 'png'))
    pyplot.close('all')

    for system in figures:
        for series in figures[system]:
            pyplot.scatter(figures[system][series][0], figures[system][series][1], c='red')
            pyplot.scatter(figures[system][series][0], figures[system][series][2], c='blue')

    across = patches.Patch(color='red', label='across-speaker')
    within = patches.Patch(color='blue', label='within-speaker')
    pyplot.legend(handles=[across, within])
    pyplot.xlabel('ratio val-loss')
    pyplot.ylabel('ratio ABX scores')
    pyplot.title('Ratio changes ' + model + ' ' + language)

    # pyplot.show()
    pyplot.savefig(data_file.replace('data', 'ratio_changes').replace('json', 'png'))
    pyplot.close('all')

    # correlations
    correlations = {}
    for system in figures:
        correlations[system] = {}
        for series in figures[system]:
            correlations[system][series] = {
                'p_across_speaker': st.pearsonr(figures[system][series][0], figures[system][series][1]),
                'p_within_speaker': st.pearsonr(figures[system][series][0], figures[system][series][2]),
                's_across_speaker': st.spearmanr(figures[system][series][0], figures[system][series][1]),
                's_within_speaker': st.spearmanr(figures[system][series][0], figures[system][series][2]),
            }

    with open(data_file.replace('data', 'ratio_changes'), 'w') as f:
        json.dump(correlations, f, indent=2)

# analyse_change_ratio('../statistical_analysis/cpc_2_french_data.json', 'French', 'CPC 2')

data = get_data_series('../models/stats/cpc/', '../evaluation/evaluation_stat/cpc', 'french', 'cpc')
data2 = get_data_series('../models/stats/cpc/', '../evaluation/evaluation_stat/cpc', 'mandarin', 'cpc')
# 0:epoch 1:val-loss 2:ABX across-spkr 3:ABX within-spkr
#get_scatter_plots_one_lang(data, 1, 'cpc_french_abx_scores.eps', 'French')
# get_scatter_plots_two_lang(data, data2, 0, 'apc_epoch_abx_scores.eps', 'French', 'Mandarin')
# get_scatter_plots_both(data, 0, 'cpc_2_10_french_abx_scores.eps')
get_scatter_plots(data, 0, 2, 'cpc_french_epoch_across_speaker.eps', 'CPC French', 'mix')
# statistical_analysis(data2, 0.95, 'mandarin', 'cpc_2')
# statistical_analysis(data, 0.95, 'french', 'cpc_2')

            
