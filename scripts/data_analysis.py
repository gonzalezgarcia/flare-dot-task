import pandas as pd
import numpy as np
import json
import re
import csv
import os
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")

# %% Config
curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)


#%% list all .txt files in the  designs folder
all_raw = []
for filename in os.listdir('../results'):
    if filename.endswith('300ms.txt'):
        all_raw += [filename]
        print(filename)
        
        
# Step 1: Load the raw lines from the .txt file
with open('../results/'+all_raw[0], 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Step 2: Detect potential header lines (starting with double-quoted field names)
header_candidates = []
for i, line in enumerate(lines):
    if line.startswith('"rt"') or line.startswith('"success"'):
        n_cols = len(line.strip().split(','))
        if n_cols >= 20:  # still sanity check column count
            header_candidates.append((i, line.strip()))

# Step 3: Show all unique headers found
unique_headers = list(set([hdr for _, hdr in header_candidates]))
print(f"🔍 Found {len(header_candidates)} potential headers in total.")
print(f"🧠 Found {len(unique_headers)} unique header versions.\n")

# Step 4: Split the lines into chunks (each from one header to the next)
header_indices = [idx for idx, _ in header_candidates]
chunks = []

for i, start_idx in enumerate(header_indices):
    # End at the next header, or the end of the file
    end_idx = header_indices[i + 1] if i < len(header_indices) - 1 else len(lines)
    chunk = lines[start_idx:end_idx]
    chunks.append(chunk)

print(f"✅ Split data into {len(chunks)} chunks")



# Step 4: # Process each chunk into a DataFrame
dfs = []
weird_chunks = []
for chunk in chunks:
    # Create a CSV reader for the chunk
    reader = csv.reader(chunk)
    # Get the headers for the chunk
    headers = next(reader)
    # Read the chunk into a DataFrame
    df = pd.DataFrame.from_records(reader, columns=headers)
    if len(df) > 195:
        df = df[(df['block_type'] == 'dot') | (df['block_type'] == 'recognition')]

    
    expected_columns = ['subject', 'subject_ProlificID', 'designmatrix_ID','age', 'gender', 'handedness', 'nationality',
                    'trial_number', 'block', 'block_type','image_id','filename',
                    'image_type', 'manipulation','phase','dot_position',
                    'RawResponse','dot_resp','dot_acc','rt', 'verbal_id']

    # Check that all expected columns are present
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print(f"⚠️ Skipping chunk from {filename} — missing columns: {missing}")
        weird_chunks.append({
            "reason": "missing_columns",
            "filename": filename,
            "headers": chunk[0].strip(),
            "sample_rows": chunk[1:6]
        })
        continue

    if len(df) > 180: # if more than 100 trials...==
        df.dropna(inplace=True)
        # reset index to have a fully increasing one
        df.reset_index(drop=True, inplace=True)
        
        # reoder columns
        df = df[expected_columns]  # This will already raise if something is missing
        

        # convert columns to numeric
        cols = ['age', 'trial_number', 'block', 'rt', 'dot_acc']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        #print(f"Subject: {df['subject'].iloc[0]}, Trials: {len(df)}")
        # Add the DataFrame to the list
        dfs.append(df)
    else:# len(df) != 195:
        print(f"⚠️ Skipping subject {df['subject'].iloc[0]} — has {len(df)} trials (expected 195)")
        weird_chunks.append({
            "reason": "wrong_trial_count",
            "subject": df['subject'].iloc[0],
            "n_trials": len(df),
            "sample_rows": df.head().to_dict(orient='records')
        })
        continue
        
        
# Merge all the DataFrames and print some desc
data = pd.concat(dfs, ignore_index=True)

# save to csv
#data.to_csv('../results/merged_all_300ms.csv', index=False)
print("\n\n\nDescriptive information of included participants:\n")


age = data.groupby(["subject"], as_index=False).age.mean()
gender = pd.DataFrame(data.groupby(["subject"]).gender.unique())
handedness = pd.DataFrame(data.groupby(["subject"]).handedness.unique())

print("%s participants went into the analysis\n" % len(age))



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


#%% quick look
# RT
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
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).rt.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="phase", y="rt",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True
)

aov = pg.rm_anova(data=grouped_data,
                      dv='rt',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

# dot acc
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
grouped_data = grouped_data[(grouped_data['rt'] > 100) & (grouped_data['rt'] < 4000)]
grouped_data = grouped_data[(grouped_data["image_type"] == 'target')]

g = sns.catplot(
    data=grouped_data, x="phase", y="dot_acc",hue="manipulation",
    palette="Set2", kind="point",order=['pre', 'post'],
    dodge=True
)
aov = pg.rm_anova(data=grouped_data,
                      dv='dot_acc',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")


# %% extract learning index by subtracitng pre semantic distance from post semantic distance

# Pivot so we have one row per subject x manipulation, with pre/post as columns
pivoted = grouped_data.pivot_table(
    index=['subject', 'manipulation'],
    columns='phase',
    values='dot_acc'
).reset_index()

pivoted['learning_index'] = pivoted['pre'] - pivoted['post']

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
    dodge=True
)

aov = pg.rm_anova(data=long_df,
                      dv='learning_index',
                      within='manipulation', subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

#%% Verbal identification
# Replace underscores with spaces
data['image_id'] = data['image_id'].str.replace('_', ' ')

# raw analysis
data.loc[data["verbal_id"] == data["image_id"], "verbal_acc"] = 1
data.loc[data["verbal_id"] != data["image_id"], "verbal_acc"] = 0


# include synonyms
things_path = '/Users/carlos/Library/CloudStorage/GoogleDrive-cgonzalez@go.ugr.es/.shortcut-targets-by-id/1puRcHFQwdotpIGskjBydD3Z_BWAM767G/WOBC Lab/projects/semper_mooney/THINGS_database/'
things_data = pd.read_csv(things_path + 'things_concepts.csv')
#things_metadata = pd.read_table(things_path + '/from_THINGSPLUS/Concept-specific/category53_longFormat.tsv')
synonyms = list(things_data["WordNet Synonyms"])
# clean up synonyms
for i in range(len(synonyms)):
    # Split the row into individual words using the split() function
    words = synonyms[i].split("_")

    # Join the words together with spaces using the join() function
    synonyms[i] = " ".join(words)

# Define a function to check if a verbal_id is in the list of synonyms
def check_synonyms(row):
    if row['verbal_id'] == '':
        return 0
    elif str(row["verbal_id"]) == 'nan':
        return 0
    else:
        # Loop through the list of synonyms and look for the image label
        for syn in synonyms:
            # Check if the image label is in the current synonym list
            if row['image_id'] in syn:
                # Check if the verbal_id is also in the synonym list
                if row['verbal_id'].split(" ")[0] in syn:
                    return 1
                    break
                else:
                    return 0
                    break
        # If we reach this point, the verbal_id was not found in any synonym list
        return 0


data['verbal_acc_corrected'] = data.apply(check_synonyms, axis=1)


grouped_data = data[data['block_type'] != 'dot'].copy()
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
                                    as_index=False).verbal_acc_corrected.mean()
grouped_data = grouped_data.reset_index()

g = sns.catplot(
    data=grouped_data, x="phase", y="verbal_acc_corrected", hue="image_type", col="manipulation",
    palette="Set2", kind="point",order=['pre',  'post'],
    dodge=True
)

grouped_data =data[data['block_type'] != 'dot'].copy()
grouped_data = grouped_data[(grouped_data["image_type"] == 'target')]
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

# %% SEMANTIC DISTANCE STUFF
from scipy.spatial.distance import pdist, squareform

# load embeddings from THINGS and compute distance matrix
# in this matrix, each row is a concept (in the same order as uniqueID, see below)
# and the columns represent the distances to different concepts
embeddings = pd.read_csv(things_path + 'Metadata/Concept-specific/Semantic Embedding/sensevec_augmented_with_wordvec.csv', header=None)
distances = pdist(embeddings, 'cosine')
dist_matrix = squareform(distances)

## in case you want to plot it, uncomment below
# plt.imshow(dist_matrix, cmap='viridis', interpolation='nearest')
# plt.colorbar(label='Cosine Distance')
# plt.title('Cosine Distance Matrix')
# plt.xlabel('Vector Index')
# plt.ylabel('Vector Index')
# plt.show()

# %% LOAD CONCEPTS AND SYNONYMS FROM THINGS DATABASE AND DO SOME CLEAN UP

concepts = pd.read_csv(things_path +  '/things_concepts.csv')
synonyms = list(concepts["WordNet Synonyms"])
uniqueID = list(concepts["uniqueID"])

# substitute "_" for spaces in synonyms
for i in range(len(synonyms)):
    words = synonyms[i].split("_") # Split the row into individual words using the split() function
    synonyms[i] = " ".join(words) # Join the words together with spaces using the join() function

# add actual filename (from uniqueID) to each row of synonyms
for i in range(len(synonyms)):
    synonyms[i] = synonyms[i] + ", " + uniqueID[i]

# %% LOAD DATA AND DO SOME CLEAN UP

# Replace spaces with underscore (to be able to compare with uniqueID)
data['image_id'] = data['image_id'].str.replace(' ', '_')

# fix a few polysemic labels
def fix_polysemia(raw_data):
    # replace string in "verbal_id" column with correct string
    data = raw_data.copy()

    #data["verbal_id"] = data["verbal_id"].replace("bottle","bottle.n.01")
    data["verbal_id"] = data["verbal_id"].replace('grape vine', 'grapevine')
    data["verbal_id"] = data["verbal_id"].replace('air bag', 'airbag')
    data["verbal_id"] = data["verbal_id"].replace('jean', 'jeans')
    data["verbal_id"] = data["verbal_id"].replace('jeanss', 'jeans')
    data["verbal_id"] = data["verbal_id"].replace('boxing glove', 'boxing gloves')
    data["verbal_id"] = data["verbal_id"].replace('boxing glovess', 'boxing gloves')
    data["verbal_id"] = data["verbal_id"].replace('shell (projectile)', 'shell1')
    data["verbal_id"] = data["verbal_id"].replace('shell (animal housing)', 'shell2')
    data["verbal_id"] = data["verbal_id"].replace('shell (fruit/nut)', 'shell3')
    data["verbal_id"] = data["verbal_id"].replace('mold (form)', 'mold1')
    data["verbal_id"] = data["verbal_id"].replace('mold (fungus)', 'mold2')
    data["verbal_id"] = data["verbal_id"].replace('mouse (animal)', 'mouse1')
    data["verbal_id"] = data["verbal_id"].replace('mouse (computer)', 'mouse2')
    data["verbal_id"] = data["verbal_id"].replace('camera (photos)', 'camera1')
    data["verbal_id"] = data["verbal_id"].replace('camera (videos)', 'camera2')
    data["verbal_id"] = data["verbal_id"].replace('bat (animal)', 'bat1')
    data["verbal_id"] = data["verbal_id"].replace('bat (sports)', 'bat2')
    data["verbal_id"] = data["verbal_id"].replace('baton (conductor', 'baton1')
    data["verbal_id"] = data["verbal_id"].replace('baton (police truncheon)', 'baton2')
    data["verbal_id"] = data["verbal_id"].replace('baton (relay running)', 'baton4')
    data["verbal_id"] = data["verbal_id"].replace('bow (decoration)', 'bow3')
    data["verbal_id"] = data["verbal_id"].replace('bow (weapon)', 'bow2')
    data["verbal_id"] = data["verbal_id"].replace('bow (hair knot)', 'bow1')
    data["verbal_id"] = data["verbal_id"].replace('bracelet (jewelry)', 'bracelet1')
    data["verbal_id"] = data["verbal_id"].replace('bracelet (wristband)', 'bracelet2')
    data["verbal_id"] = data["verbal_id"].replace('button (clothes button)', 'button1')
    data["verbal_id"] = data["verbal_id"].replace('button (device button)', 'button2')
    data["verbal_id"] = data["verbal_id"].replace('calf (animal)', 'calf1')
    data["verbal_id"] = data["verbal_id"].replace('calf (leg)', 'calf2')
    data["verbal_id"] = data["verbal_id"].replace('chest (body)', 'chest1')
    data["verbal_id"] = data["verbal_id"].replace('chest (box)', 'chest2')
    data["verbal_id"] = data["verbal_id"].replace('chicken (meat)', 'chicken1')
    data["verbal_id"] = data["verbal_id"].replace('chicken (animal)', 'chicken2')
    data["verbal_id"] = data["verbal_id"].replace('clipper (garden)', 'clipper1')
    data["verbal_id"] = data["verbal_id"].replace('clipper (nails)', 'clipper2')
    data["verbal_id"] = data["verbal_id"].replace('crystal (material)', 'crystal')
    data["verbal_id"] = data["verbal_id"].replace('crystal (rock)', 'crystal')
    data["verbal_id"] = data["verbal_id"].replace('hook (door hook)', 'hook1')
    data["verbal_id"] = data["verbal_id"].replace('hook (attachment hook)', 'hook2')
    data["verbal_id"] = data["verbal_id"].replace('juicer (juice squeezer)', 'juicer1')
    data["verbal_id"] = data["verbal_id"].replace('juicer (machine)', 'juicer2')
    data["verbal_id"] = data["verbal_id"].replace('file (card index)', 'file1')
    data["verbal_id"] = data["verbal_id"].replace('file (tool)', 'file2')
    data["verbal_id"] = data["verbal_id"].replace('pepper (seasoning)', 'pepper1')
    data["verbal_id"] = data["verbal_id"].replace('pepper (vegetable)', 'pepper2')
    data["verbal_id"] = data["verbal_id"].replace('pipe (smoking)', 'pipe1')
    data["verbal_id"] = data["verbal_id"].replace('pipe (tube)', 'pipe2')
    data["verbal_id"] = data["verbal_id"].replace('punch (beverage)', 'punch1')
    data["verbal_id"] = data["verbal_id"].replace('punch (hole punch)', 'punch2')
    data["verbal_id"] = data["verbal_id"].replace('screen (projection)', 'screen1')
    data["verbal_id"] = data["verbal_id"].replace('screen (net)', 'screen2')
    data["verbal_id"] = data["verbal_id"].replace('rack (shelf)', 'rack1')
    data["verbal_id"] = data["verbal_id"].replace('rack (stand)', 'rack2')
    data["verbal_id"] = data["verbal_id"].replace('stamp (postage stamp)', 'stamp1')
    data["verbal_id"] = data["verbal_id"].replace('stamp (pattern stamp)', 'stamp2')
    data["verbal_id"] = data["verbal_id"].replace('stove (food oven)', 'stove1')
    data["verbal_id"] = data["verbal_id"].replace('stove (heating oven)', 'stove2')
    data["verbal_id"] = data["verbal_id"].replace('straw (grain)', 'straw1')
    data["verbal_id"] = data["verbal_id"].replace('straw (drinking straw)', 'straw2')
    data["verbal_id"] = data["verbal_id"].replace('tank (storage)', 'tank1')
    data["verbal_id"] = data["verbal_id"].replace('tank (vehicle)', 'tank2')
    data["verbal_id"] = data["verbal_id"].replace('walker (for older adults)', 'walker1')
    data["verbal_id"] = data["verbal_id"].replace('walker (for toddlers)', 'walker2')
    data["verbal_id"] = data["verbal_id"].replace('chewing gum', 'gum')
    
    return data

data = fix_polysemia(data)
data = data.reset_index()

# %% COMPUTE SEMANTIC DISTANCES

# check if "merged_all_1002p.csv" file in +data contains the column "semantic_distance"
semantic_distances = []
# create empty column in "data" collad "semantic_distance
data["semantic_distance"] = np.nan

data_recog = data[data['block_type'] != 'dot'].copy()
# reset index to have a fully increasing one
data_recog.reset_index(drop=True, inplace=True)

print(len(data_recog))
for i in range(len(data_recog)): # for each row in data
    # find index of image in uniqueID
    index_true = uniqueID.index(data_recog["image_id"][i])
    
    # find index of actual verbal response in uniqueID
    # if response is NaN, set distance to NaN
    if pd.isnull(data["verbal_id"][i]):
        semantic_distances.append(np.nan)
        # set semantic distance in the data pd to NaN
        data_recog.loc[i, "semantic_distance"] = np.nan
        continue
    else: # check if verbal_id is in any row of synonyms
        # Loop through the list of synonyms and look for the image label
        for idx,syn in enumerate(synonyms):
            # Check if the verbal label is in the current synonym list
            if data_recog["verbal_id"][i] in syn:
                index_response = idx # if so, take the index of that row
                semantic_distances.append(dist_matrix[index_true,index_response])
                data_recog.loc[i, "semantic_distance"]  = dist_matrix[index_true,index_response]
                continue
        
    #print(f"Row {i}: {data_recog['image_id'][i]} - {data_recog['verbal_id'][i]} - {semantic_distances[i]}")
    
grouped_data = data_recog[data_recog['block_type'] != 'dot']
grouped_data = grouped_data.groupby(['subject','image_type', 'manipulation', 'phase'],
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
    palette="Set2",  kind="box", order = ['pre', 'color', 'post'],
    dodge=True
)

grouped_data = data_recog[data_recog['image_type'] == 'target']
grouped_data = data_recog[data_recog['phase'] != 'color']

grouped_data = grouped_data.groupby(['subject', 'manipulation', 'phase'],
                                    as_index=False).semantic_distance.mean()
grouped_data = grouped_data.reset_index()

aov = pg.rm_anova(data=grouped_data,
                      dv='semantic_distance',
                      within=['phase','manipulation'], subject='subject',
                      detailed=True)
pg.print_table(aov, floatfmt=".3f")

posthoc = pg.pairwise_tests(data=grouped_data,
                            dv='semantic_distance',
                            within=['manipulation','phase'], subject='subject',
                            parametric=True, padjust='fdr_bh',
                            effsize='hedges')
pg.print_table(posthoc, floatfmt=".3f")

g = sns.catplot(
    data=grouped_data, hue="manipulation", y="semantic_distance", x="phase", 
    palette="Set2",  kind="box", order = ['pre',  'post'],
    dodge=True
)




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
# %% extract learning index by subtracitng pre semantic distance from post semantic distance
grouped_data = data_recog[data_recog['block_type'] != 'dot']
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