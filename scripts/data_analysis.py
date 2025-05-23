"""
Analyze cleaned per-subject data from data/derivatives/
"""

import pandas as pd
import numpy as np
import os
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Plotting style
sns.set_style("ticks")
sns.set_context("talk")

#%% Define paths and load data
exp_code = "exp_3"
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, "..", "data", exp_code, "derivatives")

# Load all cleaned subject files
all_csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
dfs = []

for file in all_csvs:
    df = pd.read_csv(os.path.join(data_dir, file))
    # if has 195 trials, append, otherwise skip
    if len(df) == 195:
        print(f"✅ Loaded {file} with {len(df)} trials")
        dfs.append(df)
    else:
        print(f"❌ Skipped {file} with {len(df)} trials")
        continue


#%% Combine all into one DataFrame and print descriptive
data = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(dfs)} 'good' participants. Skipped {len(all_csvs) - len(dfs)} participants due to not finishing the experiment.")

age = data.groupby(["subject"], as_index=False).age.mean()
gender = pd.DataFrame(data.groupby(["subject"]).gender.unique())
handedness = pd.DataFrame(data.groupby(["subject"]).handedness.unique())



mean_age = age["age"].mean()
std_dev = age["age"].std()
age_range = [age["age"].min(), age["age"].max()]
output_string = f"Mean Age: {mean_age:.2f} (±{std_dev:.2f}), Range = [{age_range[0]} - {age_range[1]}]"
print(output_string)

gender_counts = gender["gender"].value_counts()
output_string = "Gender: " + \
    ', '.join([f"{count} {gender}" for gender,
                count in gender_counts.items()])
output_string = output_string.replace(
    '[', '').replace(']', '').replace("'", '')
print(output_string)

hand_counts = handedness["handedness"].value_counts()
output_string = "Handedness: " + \
    ', '.join([f"{count} {handedness}" for handedness,
                count in hand_counts.items()])
output_string = output_string.replace(
    '[', '').replace(']', '').replace("'", '')
print(output_string)
# Calculate the number and proportion of unique subjects per nationality
nationality_counts = data.groupby("nationality")["subject"].nunique()
total_subjects = nationality_counts.sum()
proportions = (nationality_counts / total_subjects) * 100

# # Print the results
# print("Nationality:")
# for nationality, count in nationality_counts.items():
#     proportion = proportions[nationality]
#     print(f"{nationality}: {count} subjects ({proportion:.2f}%)")

#%% DOT TASK - RT
grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['dot_acc'] == 1)]
grouped_data = grouped_data[(grouped_data['rt'] > 100) & (grouped_data['rt'] < 4999)]
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).rt.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="phase", y="rt", hue="image_type", col="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True
)

grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['dot_acc'] == 1)]
grouped_data = grouped_data[(grouped_data["image_type"] == 'target')]
grouped_data = grouped_data[(grouped_data['rt'] > 100) & (grouped_data['rt'] < 4000)]
grouped_data = grouped_data[(grouped_data["phase"] != 'color')]

grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).rt.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="phase", y="rt",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True, errorbar='se'
)

aov = pg.rm_anova(data=grouped_data,
                      dv='rt',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

# compute performance relative to catch trials for each manipulation and phase
grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['dot_acc'] == 1)]
grouped_data = grouped_data[(grouped_data['rt'] > 100) & (grouped_data['rt'] < 4999)]

# Separate target and catch trials
target_data = grouped_data[grouped_data['image_type'] == 'target']
catch_data = grouped_data[grouped_data['image_type'] == 'catch']

# Compute mean RT for target and catch trials
target_grouped = target_data.groupby(['subject', 'manipulation', 'phase'], as_index=False).rt.mean()
catch_grouped = catch_data.groupby(['subject', 'manipulation', 'phase'], as_index=False).rt.mean()

# Merge target and catch data on subject, manipulation, and phase
merged_data = pd.merge(target_grouped, catch_grouped, on=['subject', 'manipulation', 'phase'], suffixes=('_target', '_catch'))

# Compute RT relative to catch trials
merged_data['rt_relative'] = merged_data['rt_catch'] - merged_data['rt_target']

# Keep only relevant columns
grouped_data = merged_data[['subject', 'manipulation', 'phase', 'rt_relative']]

g = sns.catplot(
    data=grouped_data, x="phase", y="rt_relative",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True, errorbar='se'
)

aov = pg.rm_anova(data=grouped_data,
                      dv='rt_relative',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")




#%% DOT TASK - ACC
grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['rt'] > 100)]
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).dot_acc.mean()
grouped_data = grouped_data.reset_index()


g = sns.catplot(
    data=grouped_data, x="phase", y="dot_acc", hue="image_type", col="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True
)

grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['rt'] > 100)]
grouped_data = grouped_data[(grouped_data["image_type"] == 'target')]
grouped_data = grouped_data[(grouped_data["phase"] != 'color')]

g = sns.catplot(
    data=grouped_data, x="phase", y="dot_acc",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True, errorbar='se'
)
aov = pg.rm_anova(data=grouped_data,
                      dv='dot_acc',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")


# compute performance relative to catch trials for each manipulation and phase
grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['rt'] > 100)]
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).dot_acc.mean()
grouped_data = grouped_data.reset_index()


# Separate target and catch trials
target_data = grouped_data[grouped_data['image_type'] == 'target']
catch_data = grouped_data[grouped_data['image_type'] == 'catch']

# Compute mean acc for target and catch trials
target_grouped = target_data.groupby(['subject', 'manipulation', 'phase'], as_index=False).dot_acc.mean()
catch_grouped = catch_data.groupby(['subject', 'manipulation', 'phase'], as_index=False).dot_acc.mean()

# Merge target and catch data on subject, manipulation, and phase
merged_data = pd.merge(target_grouped, catch_grouped, on=['subject', 'manipulation', 'phase'], suffixes=('_target', '_catch'))

# Compute acc relative to catch trials
merged_data['acc_relative'] = merged_data['dot_acc_target'] - merged_data['dot_acc_catch']

# Keep only relevant columns
grouped_data = merged_data[['subject', 'manipulation', 'phase', 'acc_relative']]

g = sns.catplot(
    data=grouped_data, x="phase", y="acc_relative",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True, errorbar='se'
)

aov = pg.rm_anova(data=grouped_data,
                      dv='acc_relative',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")


# %% extract learning index by subtracitng pre semantic distance from post semantic distance
grouped_data = data[data['block_type'] == 'dot'].copy()
grouped_data = grouped_data[(grouped_data['rt'] > 100)]
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).dot_acc.mean()
grouped_data = grouped_data.reset_index()
# Pivot so we have one row per subject x manipulation, with pre/post as columns
pivoted = grouped_data.pivot_table(
    index=['subject', 'manipulation'],
    columns='phase',
    values='dot_acc'
).reset_index()

pivoted['learning_index'] = pivoted['post'] - pivoted['pre']

# Convert from wide to long format
long_df = pd.melt(
    pivoted,
    id_vars=['subject', 'manipulation', 'learning_index'],  # keep learning_index with each row
    value_vars=['post'],
    var_name='phase',
    value_name='dot_acc'
)

g = sns.catplot(
    data=long_df, x="manipulation", y="learning_index",
    palette="Set2", kind="point",errorbar="se",
    dodge=True, 
)

aov = pg.rm_anova(data=long_df,
                      dv='learning_index',
                      within='manipulation', subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")
# %% recognition - verbal accuracy

verbal_data = data[data['block_type'] != 'dot'].copy()
grouped_data = verbal_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).verbal_acc_corrected.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="phase", y="verbal_acc_corrected", hue="image_type", col="manipulation",
    palette="Set2", kind="point",order=['pre',  'color','post'],
    dodge=True
)

## How do our manipulations affect verbal id during unambiguous image presentation?
unambiguous_verbal_data = verbal_data[(verbal_data["image_type"] == 'target')]
unambiguous_verbal_data = unambiguous_verbal_data[(unambiguous_verbal_data["phase"] == 'color')]
# combine horizontal_flip and regular into one group (upright)
unambiguous_verbal_data['manipulation'] = np.where(
    unambiguous_verbal_data['manipulation'].isin(['horizontal_flip', 'regular']),
    'upright',
    unambiguous_verbal_data['manipulation']
)
unambiguous_verbal_data = unambiguous_verbal_data.groupby(['subject', 'manipulation'],
                                    as_index=False).verbal_acc_corrected.mean()
unambiguous_verbal_data = unambiguous_verbal_data.reset_index()
g = sns.catplot(
    data=unambiguous_verbal_data, x="manipulation", y="verbal_acc_corrected",
    kind="point",
    dodge=True
)

# one-sided t-test
posthoc = pg.pairwise_tests(data=unambiguous_verbal_data,
                            dv='verbal_acc_corrected',
                            within='manipulation', subject='subject',
                            parametric=True, padjust='fdr_bh', alternative = "greater",
                            effsize='hedges')
pg.print_table(posthoc, floatfmt=".3f")

## How do our manipulations affect verbal id during Mooney images?


grouped_data =data[data['block_type'] != 'dot'].copy()
grouped_data = grouped_data[(grouped_data["image_type"] == 'target')]
grouped_data = grouped_data[(grouped_data["phase"] != 'color')]

grouped_data = grouped_data.groupby(['subject', 'manipulation', 'phase'],
                                    as_index=False).verbal_acc_corrected.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="phase", y="verbal_acc_corrected",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True
)
aov = pg.rm_anova(data=grouped_data,
                      dv='verbal_acc_corrected',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

posthoc = pg.pairwise_tests(data=grouped_data,
                            dv='verbal_acc_corrected',
                            within=['phase', 'manipulation'], subject='subject',
                            parametric=True, padjust='fdr_bh',
                            effsize='hedges')
pg.print_table(posthoc, floatfmt=".3f")




# %% recognition - semantic distance
data_recog = data[data['block_type'] != 'dot'].copy()
# remove trials with nan in semantic distance
data_recog = data_recog[data_recog['semantic_distance'].notna()]
grouped_data = data_recog.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).semantic_distance.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="image_type", y="semantic_distance", hue="phase", col="manipulation",
    palette="Set2",  kind="box", hue_order = ['pre', 'color', 'post'],
    dodge=True
)

grouped_data = data_recog[data_recog['image_type'] == 'target']
grouped_data = grouped_data.groupby(['subject', 'manipulation', 'phase'],
                                    as_index=False).semantic_distance.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, hue="manipulation", y="semantic_distance", x="phase", 
    palette="Set2",  kind="point", order = ['pre', 'color', 'post'],
    dodge=True, errorbar='se'
)

grouped_data = data_recog[data_recog['image_type'] == 'target']
grouped_data = data_recog[data_recog['phase'] != 'color']

grouped_data = grouped_data.groupby(['subject', 'manipulation', 'phase'],
                                    as_index=False).semantic_distance.mean()
grouped_data = grouped_data.reset_index()



g = sns.catplot(
    data=grouped_data, x="phase", y="semantic_distance",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True
)
aov = pg.rm_anova(data=grouped_data,
                      dv='semantic_distance',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

posthoc = pg.pairwise_tests(data=grouped_data,
                            dv='semantic_distance',
                            within=['phase', 'manipulation'], subject='subject',
                            parametric=True, padjust='fdr_bh',
                            effsize='hedges')
pg.print_table(posthoc, floatfmt=".3f")
# %% extract learning index by subtracitng pre semantic distance from post semantic distance
data_recog = data[data['block_type'] != 'dot'].copy()
data_recog = data_recog[data_recog['semantic_distance'].notna()]
grouped_data = data_recog[data_recog['image_type'] == 'target']
grouped_data = data_recog[data_recog['phase'] != 'color']
grouped_data = grouped_data.groupby(['subject', 'manipulation', 'phase'],
                                    as_index=False).semantic_distance.mean()


# Pivot so we have one row per subject x manipulation, with pre/post as columns
pivoted = grouped_data.pivot_table(
    index=['subject', 'manipulation'],
    columns='phase',
    values='semantic_distance'
).reset_index()

pivoted['learning_index'] = pivoted['pre'] - pivoted['post']

# Convert from wide to long format
long_df = pd.melt(
    pivoted,
    id_vars=['subject', 'manipulation', 'learning_index'],  # keep learning_index with each row
    value_vars=['post'],
    var_name='phase',
    value_name='semantic_distance'
)

g = sns.catplot(
    data=long_df, x="manipulation", y="learning_index",
    palette="Set2", kind="point",errorbar="se",
    dodge=True
)

aov = pg.rm_anova(data=long_df,
                      dv='learning_index',
                      within='manipulation', subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

posthoc = pg.pairwise_tests(data=long_df,
                            dv='learning_index',
                            within='manipulation', subject='subject',
                            parametric=True, padjust='fdr_bh',
                            effsize='hedges')
pg.print_table(posthoc, floatfmt=".3f")

#%% Prior usage
grouped_data = data_recog[data_recog['block_type'] != 'dot']
grouped_data = data_recog[data_recog['image_type'] == 'target']
grouped_data = data_recog[data_recog['phase'] != 'pre']
grouped_data = grouped_data.groupby(['subject', 'manipulation', 'phase'],
                                    as_index=False).semantic_distance.mean()
pivoted = grouped_data.pivot_table(
    index=['subject', 'manipulation'],
    columns='phase',
    values='semantic_distance'
).reset_index()

pivoted['prior_usage'] = 1 - (pivoted['post'] - pivoted['color'])

# Convert from wide to long format
long_df = pd.melt(
    pivoted,
    id_vars=['subject', 'manipulation', 'prior_usage'],  # keep learning_index with each row
    value_vars=['post'],
    var_name='phase',
    value_name='semantic_distance'
)

g = sns.catplot(
    data=long_df, x="manipulation", y="prior_usage",
    palette="Set2", kind="point",errorbar="se",
    dodge=True
)

aov = pg.rm_anova(data=long_df,
                      dv='prior_usage',
                      within='manipulation', subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

posthoc = pg.pairwise_tests(data=long_df,
                            dv='prior_usage',
                            within=['manipulation'], subject='subject',
                            parametric=True, padjust='fdr_bh',
                            effsize='hedges')
pg.print_table(posthoc, floatfmt=".3f")
