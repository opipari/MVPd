import os
import json
import math
import clip
import torch
import numpy as np
import pandas as pd
import pickle
import random

from PIL import Image

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.cluster import KMeans
from adjustText import adjust_text
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.patheffects as pe
from wordcloud import WordCloud



rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
hex2rgb = lambda hexs: list(int(hexs[i:i+2], 16) for i in (0, 2, 4))


#
# Must have all panoptic instances available in MVPd/zero_shot/{train|val}
# Must have all training videos available in MVPd/train
#


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-L/14', device, jit=True)
total_params = sum(p.numel() for p in model.parameters())




train_data = json.load(open('MVPd/zero_shot/train/panoptic_train.json','r'))
val_data = json.load(open('MVPd/zero_shot/val/panoptic_val.json','r'))



def get_instance_map(instance_list, valid_cats=range(41)):
    instance_map = {}
    for inst in instance_list:
        inst_cat = inst['category_id']
        inst_name = inst['raw_category']
        inst_color = inst['color']
        inst_video = inst['video_name']
        inst_scene = inst['scene_name']
        if inst_cat not in valid_cats:
            continue
        color = rgb2hex(*inst_color)
        inst_key = inst_scene+'.'+color
        if inst_key not in instance_map:
            instance_map[inst_key] = {'scene': set(), 'videos': set()}
        else:
            assert instance_map[inst_key]['name'] == inst_name
        instance_map[inst_key]['scene'].add(inst_scene)
        instance_map[inst_key]['videos'].add(inst_video)
        instance_map[inst_key]['name'] = inst_name
    for inst_key in instance_map.keys():
        assert len(instance_map[inst_key]['scene'])==1
        instance_map[inst_key]['scene'] = next(iter(instance_map[inst_key]['scene']))
    return instance_map


train_instances = get_instance_map(train_data['instances'], valid_cats=[39,40])
val_instances = get_instance_map(val_data['instances'], valid_cats=[39,40])
print(f"Number of training object instances: {len(train_instances.keys())}")
print(f"Number of validation object intstances: {len(val_instances.keys())}")
print()

train_names = [train_instances[k]['name'] for k in train_instances]
val_names = [val_instances[k]['name'] for k in val_instances]
all_object_names = train_names + val_names
print(f"Number of training names: {len(set(train_names))}")
print(f"Number of validation names: {len(set(val_names))}")
print()
    
batch_size = 100
clip_text_data = sorted(list(set(all_object_names)))
clip_features = np.zeros((len(clip_text_data),768))

with torch.no_grad():
    for batch_i in range(math.ceil(len(clip_text_data)/batch_size)):
        batch_text = clip_text_data[batch_i*batch_size:(batch_i+1)*batch_size]
        batch_text = clip.tokenize(batch_text).to(device)
        batch_features = model.encode_text(batch_text)
        clip_features[batch_i*batch_size:(batch_i+1)*batch_size, :] = batch_features.cpu().numpy()



n_clusters = 20
clip_kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(clip_features)
clip_clusters = clip_kmeans.predict(clip_features)
clip_cluster_colors = [sns.color_palette("husl",n_clusters)[i] for i in clip_clusters]

# clip_tsne = TSNE(n_components=2, n_iter=10000, learning_rate='auto', init='random', perplexity=75).fit(clip_features)
# clip_embedded = clip_tsne.transform(clip_features)
clip_nca = NeighborhoodComponentsAnalysis(n_components=2, max_iter=1000, random_state=seed).fit(clip_features, clip_clusters)
clip_embedded = clip_nca.transform(clip_features)



all_object_counts = {'Raw Category Name': {},
                     'Object Instances': {},
                     'Videos': {},
                     'Scenes': {},
                     'Count of Scenes': {},
                     'Count of Videos': {},
                     'Count of Instances': {},
                     'Set': {},
                     'Cluster': {}
                     }

for i, (name, cluster) in enumerate(zip(clip_text_data, clip_clusters)):
    train_index = i*2
    train_instances_of_name = set()
    train_videos_of_name = set()
    train_scenes_of_name = set()
    for train_object_key in train_instances.keys():
        if train_instances[train_object_key]['name'] == name:
            train_scenes_of_name.add(train_instances[train_object_key]['scene'])
            train_videos_of_name.update(train_instances[train_object_key]['videos'])
            train_instances_of_name.add(train_object_key)
    all_object_counts['Raw Category Name'][train_index] = name
    all_object_counts['Object Instances'][train_index] = train_instances_of_name
    all_object_counts['Videos'][train_index] = train_videos_of_name
    all_object_counts['Scenes'][train_index] = train_scenes_of_name
    all_object_counts['Count of Scenes'][train_index] = len(all_object_counts['Scenes'][train_index])
    all_object_counts['Count of Videos'][train_index] = len(all_object_counts['Videos'][train_index])
    all_object_counts['Count of Instances'][train_index] = len(all_object_counts['Object Instances'][train_index])
    all_object_counts['Set'][train_index] = 'Train'
    all_object_counts['Cluster'][train_index] = cluster

    val_index = i*2+1
    val_instances_of_name = set()
    val_videos_of_name = set()
    val_scenes_of_name = set()
    for val_object_key in val_instances.keys():
        if val_instances[val_object_key]['name'] == name:
            val_scenes_of_name.add(val_instances[val_object_key]['scene'])
            val_videos_of_name.update(val_instances[val_object_key]['videos'])
            val_instances_of_name.add(val_object_key)
    all_object_counts['Raw Category Name'][val_index] = name
    all_object_counts['Object Instances'][val_index] = val_instances_of_name
    all_object_counts['Videos'][val_index] = val_videos_of_name
    all_object_counts['Scenes'][val_index] = val_scenes_of_name
    all_object_counts['Count of Scenes'][val_index] = len(all_object_counts['Scenes'][val_index])
    all_object_counts['Count of Videos'][val_index] = len(all_object_counts['Videos'][val_index])
    all_object_counts['Count of Instances'][val_index] = len(all_object_counts['Object Instances'][val_index])
    all_object_counts['Set'][val_index] = 'Val'
    all_object_counts['Cluster'][val_index] = cluster




fig,ax = plt.subplots(figsize=(25.6,19.2))
ax.set_xlabel("NCA Component 1", fontsize=28)
ax.set_ylabel("NCA Component 2", fontsize=28)
ax.grid(color='black', linestyle='-', linewidth=2, alpha=0.15)
ax.tick_params(axis='both', which='major', labelsize=23)
ax.scatter(clip_embedded[:,0], clip_embedded[:,1], s=250.0, c=clip_cluster_colors)
texts = []
texts_per_cluster = [np.random.choice(np.arange(len(clip_clusters))[clip_clusters==ci], size=5, replace=False) for ci in range(n_clusters)]
for i, txt in enumerate(clip_text_data):
    if i in texts_per_cluster[clip_clusters[i]]:
        texts.append(ax.text(clip_embedded[i,0], clip_embedded[i,1], txt, size=23, color=clip_cluster_colors[i], path_effects=[pe.withStroke(linewidth=3, foreground=(1.,1.,1.,0.85))]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=2))
xlim, ylim = ax.get_xlim(), ax.get_ylim()
plt.savefig(f'CLIP_Embedding_Scatter_{seed}.png')
plt.close()


fig,ax = plt.subplots(figsize=(25.6,19.2))
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("NCA Component 1", fontsize=28)
ax.set_ylabel("NCA Component 2", fontsize=28)
ax.grid(color='black', linestyle='-', linewidth=2, alpha=0.15)
ax.tick_params(axis='both', which='major', labelsize=23)
clip_cluster_centers = clip_nca.transform(clip_kmeans.cluster_centers_)

clip_cluster_wordclouds = ["" for _ in range(n_clusters)]
for text, cluster_i in zip(clip_text_data, clip_clusters):
    clip_cluster_wordclouds[cluster_i] += text+' '


x, y = np.ogrid[:1500, :1500]
mask = (x - 750) ** 2 + (y - 750) ** 2 > 750 ** 2
mask = 255 * mask.astype(int)

clip_cluster_wordclouds = [WordCloud(background_color=None, mode="RGBA", mask=mask).generate(text) for text in clip_cluster_wordclouds]
clip_cluster_recolors = [(lambda clr: (lambda *args, **kwargs: tuple(int(255*c) for c in clr)))(color) for color in list(sns.color_palette("husl",n_clusters))]
for i, (wc, recolor, cntr) in enumerate(zip(clip_cluster_wordclouds, clip_cluster_recolors, clip_cluster_centers)):
    wc.recolor(color_func=recolor)
    # wc.to_file(f'wordcloud_{i}.png')
    imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(wc.to_array(), zoom=0.15), cntr, frameon=False
        )
    imagebox.set(zorder=1)
    ax.add_artist(imagebox)


plt.savefig(f'CLIP_Embedding_Words_{seed}.png')
plt.close()


df = pd.DataFrame.from_dict(all_object_counts)



all_cluster_counts = {'Raw Category Names': {},
                     'Object Instances': {},
                     'Videos': {},
                     'Scenes': {},
                     'Count of Scenes': {},
                     'Count of Videos': {},
                     'Count of Instances': {},
                     'Set': {},
                     'Cluster': {}
                     }

for i in range(n_clusters):
    train_index = i*2
    all_cluster_counts['Raw Category Names'][train_index] = set(df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Raw Category Name'].tolist())
    all_cluster_counts['Object Instances'][train_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Object Instances'].tolist())
    all_cluster_counts['Videos'][train_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Videos'].tolist())
    all_cluster_counts['Scenes'][train_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Scenes'].tolist())
    all_cluster_counts['Count of Scenes'][train_index] = len(all_cluster_counts['Scenes'][train_index])
    all_cluster_counts['Count of Videos'][train_index] = len(all_cluster_counts['Videos'][train_index])
    all_cluster_counts['Count of Instances'][train_index] = len(all_cluster_counts['Object Instances'][train_index])
    all_cluster_counts['Set'][train_index] = 'Train'
    all_cluster_counts['Cluster'][train_index] = i

    val_index = i*2+1
    all_cluster_counts['Raw Category Names'][val_index] = set(df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Raw Category Name'].tolist())
    all_cluster_counts['Object Instances'][val_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Object Instances'].tolist())
    all_cluster_counts['Videos'][val_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Videos'].tolist())
    all_cluster_counts['Scenes'][val_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Scenes'].tolist())
    all_cluster_counts['Count of Scenes'][val_index] = len(all_cluster_counts['Scenes'][val_index])
    all_cluster_counts['Count of Videos'][val_index] = len(all_cluster_counts['Videos'][val_index])
    all_cluster_counts['Count of Instances'][val_index] = len(all_cluster_counts['Object Instances'][val_index])
    all_cluster_counts['Set'][val_index] = 'Val'
    all_cluster_counts['Cluster'][val_index] = i


# with open('all_cluster_counts.pickle', 'wb') as handle:
#     pickle.dump(all_cluster_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)


cluster_labels = {
                    0: 'Fixtures',
                    1: 'Artwork',
                    2: 'Electronics',
                    3: 'Soap',
                    4: 'Object Storage',
                    5: 'Hard Containers',
                    6: 'Hangers',
                    7: 'Soft Containers',
                    8: 'Wall Features',
                    9: 'Doors & Windows',
                    10: 'Food Containers',
                    11: 'HVAC',
                    12: 'Stands',
                    13: 'Cooking',
                    14: 'Clutter',
                    15: 'Hobby',
                    16: 'Stairs',
                    17: 'Liquid Containers',
                    18: 'Bathing',
                    19: 'Clothes Storage',
                }


df_cluster = pd.DataFrame.from_dict(all_cluster_counts)

palette = ['#1f78b4', '#fc8e62']
plt.figure()
bar = sns.barplot(data=df_cluster, x="Cluster", y="Count of Videos", hue="Set", palette=palette)
bar.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right', rotation_mode='anchor')
for xtick, color in zip(bar.get_xticklabels(), list(sns.color_palette("husl",n_clusters))):
    xtick.set_color(color)
fig = bar.get_figure()
fig.savefig(f"Cluster_Bar_VideoCount{seed}.png", bbox_inches='tight')
plt.close()


plt.figure()
bar = sns.barplot(data=df_cluster, x="Cluster", y="Count of Instances", hue="Set", palette=palette)
bar.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right', rotation_mode='anchor')
for xtick, color in zip(bar.get_xticklabels(), list(sns.color_palette("husl",n_clusters))):
    xtick.set_color(color)
fig = bar.get_figure()
fig.savefig(f"Cluster_Bar_InstanceCount{seed}.png", bbox_inches='tight')
plt.close()





train_data = json.load(open('MVPd/train/panoptic_train.json','r'))

n_zero_shot_cat = 4
zero_shot_clusters = df_cluster[df_cluster['Set']=='Train'].sort_values('Count of Videos')['Cluster'][:n_zero_shot_cat].tolist()


zero_shot_train_scenes = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Train')]['Scenes'].tolist())
zero_shot_train_videos_to_ignore = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Train')]['Videos'].tolist())
zero_shot_train_instances_to_ignore = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Train')]['Object Instances'].tolist())
zero_shot_train_videos = set([v['video_name'] for v in train_data['videos']]).difference(zero_shot_train_videos_to_ignore)

zero_shot_train_data = {"accepted": sorted(list(zero_shot_train_videos)), "rejected": sorted(list(zero_shot_train_videos_to_ignore))}
with open('MVPd/train/zero_shot.json', 'w') as f:
    json.dump(zero_shot_train_data, f)

print(f"Zero-Shot Train Scenes: {len(zero_shot_train_scenes)}")
print(f"Zero-Shot Train Videos: {len(zero_shot_train_videos_to_ignore)}")
print(f"Remaining Train Videos: {len(zero_shot_train_videos)}")
print(f"Zero-Shot Train Instances: {len(zero_shot_train_instances_to_ignore)}")




val_original_scenes = ['00800-TEEsavR23oF',
                        '00802-wcojb4TFT35',
                        '00803-k1cupFYWXJ6',
                        '00808-y9hTuugGdiq',
                        '00810-CrMo8WxCyVb',
                        '00813-svBbv1Pavdk',
                        '00814-p53SfW6mjZe',
                        '00815-h1zeeAwLh9Z',
                        '00820-mL8ThkuaVTM',
                        '00821-eF36g7L6Z9M',
                        '00823-7MXmsvcQjpJ',
                        '00824-Dd4bFSTQ8gi',
                        '00827-BAbdmeyTvMZ',
                        '00829-QaLdnwvtxbs',
                        '00831-yr17PDCnDDW',
                        '00832-qyAac8rV8Zk',
                        '00835-q3zU7Yy5E5s',
                        '00839-zt1RVoi7PcG',
                        '00843-DYehNKdT76V',
                        '00844-q5QZSEeHe5g',
                        '00847-bCPU9suPUw9',
                        '00848-ziup5kvtCCR',
                        '00849-a8BtkwhxdRV',
                        '00853-5cdEh9F2hJL',
                        '00861-GLAQ4DNUx5U',
                        '00862-LT9Jq6dN3Ea',
                        '00869-MHPLjHsuG27',
                        '00871-VBzV5z6i1WS',
                        '00873-bxsVRursffK',
                        '00876-mv2HUxq3B53',
                        '00877-4ok3usBNeis',
                        '00878-XB4GS9ShBRE',
                        '00880-Nfvxx8J5NCo',
                        '00890-6s7QHgap2fW',
                        '00891-cvZr5TUy5C5',
                        '00894-HY1NcmCgn3n'
                        ]




zero_shot_val_original_scenes = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Val')]['Scenes'].tolist())
non_zero_shot_val_original_scenes = set(val_original_scenes).difference(zero_shot_val_original_scenes)
zero_shot_val_original_videos = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Val')]['Videos'].tolist())
zero_shot_val_original_instances = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Val')]['Object Instances'].tolist())

print(f"Zero-Shot Val (original) Scenes: {len(zero_shot_val_original_scenes)}")
print(f"Zero-Shot Val (original) Videos: {len(zero_shot_val_original_videos)}")
print(f"Zero-Shot Val (original) Instances: {len(zero_shot_val_original_instances)}")



val_test_split = 0.5
zero_shot_val_original_scenes, non_zero_shot_val_original_scenes = sorted(list(zero_shot_val_original_scenes)), sorted(list(non_zero_shot_val_original_scenes))
np.random.shuffle(zero_shot_val_original_scenes)
np.random.shuffle(non_zero_shot_val_original_scenes)

zero_shot_split = int(len(zero_shot_val_original_scenes)*val_test_split)
non_zero_shot_split = int(len(non_zero_shot_val_original_scenes)*val_test_split)
zero_shot_val_new_scenes = sorted(zero_shot_val_original_scenes[:zero_shot_split]+non_zero_shot_val_original_scenes[:non_zero_shot_split])
zero_shot_test_new_scenes = sorted(zero_shot_val_original_scenes[zero_shot_split:]+non_zero_shot_val_original_scenes[non_zero_shot_split:])


with open('data/val.txt', 'w') as f:
    for scene in zero_shot_val_new_scenes:
        f.write(f"{scene}\n")


with open('data/test.txt', 'w') as f:
    for scene in zero_shot_test_new_scenes:
        f.write(f"{scene}\n")


zero_shot_val_new_videos = set()
zero_shot_val_new_instances = set()

zero_shot_test_new_videos = set()
zero_shot_test_new_instances = set()


for val_object_key in val_instances.keys():
    if val_object_key in zero_shot_val_original_instances:
        if val_instances[val_object_key]['scene'] in zero_shot_val_new_scenes:
            zero_shot_val_new_videos.update(val_instances[val_object_key]['videos'])
            zero_shot_val_new_instances.add(val_object_key)
        if val_instances[val_object_key]['scene'] in zero_shot_test_new_scenes:
            zero_shot_test_new_videos.update(val_instances[val_object_key]['videos'])
            zero_shot_test_new_instances.add(val_object_key)

zero_shot_val_data = {"zero-shot": [{"scene_name": el.split('.')[0], "color": list(hex2rgb(el.split('.')[1]))} for el in sorted(list(zero_shot_val_new_instances))] }
with open('MVPd/val/zero_shot.json', 'w') as f:
    json.dump(zero_shot_val_data, f)

os.makedirs('MVPd/test', exist_ok=True)
zero_shot_test_data = {"zero-shot": [{"scene_name": el.split('.')[0], "color": list(hex2rgb(el.split('.')[1]))} for el in sorted(list(zero_shot_test_new_instances))] }
with open('MVPd/test/zero_shot.json', 'w') as f:
    json.dump(zero_shot_test_data, f)

print(f"Val (new) Scenes: {len(zero_shot_val_new_scenes)}")
print(f"Zero-Shot Val (new) Videos: {len(zero_shot_val_new_videos)}")
print(f"Zero-Shot Val (new) Instances: {len(zero_shot_val_new_instances)}")


print(f"Test (new) Scenes: {len(zero_shot_test_new_scenes)}")
print(f"Zero-Shot Test (new) Videos: {len(zero_shot_test_new_videos)}")
print(f"Zero-Shot Test (new) Instances: {len(zero_shot_test_new_instances)}")



all_object_counts_newsplit = {'Raw Category Name': {},
                     'Object Instances': {},
                     'Videos': {},
                     'Scenes': {},
                     'Count of Scenes': {},
                     'Count of Videos': {},
                     'Count of Instances': {},
                     'Set': {},
                     'Cluster': {}
                     }

for i, (name, cluster) in enumerate(zip(clip_text_data, clip_clusters)):
    train_index = i*3
    train_instances_of_name = set()
    train_videos_of_name = set()
    train_scenes_of_name = set()
    for train_object_key in train_instances.keys():
        if train_instances[train_object_key]['name'] == name:
            train_videos_of_name_not_ignored = train_instances[train_object_key]['videos'].difference(zero_shot_train_data["rejected"])
            if len(train_videos_of_name_not_ignored)>0:
                train_scenes_of_name.add(train_instances[train_object_key]['scene'])
                train_videos_of_name.update(train_videos_of_name_not_ignored)
                train_instances_of_name.add(train_object_key)
    all_object_counts_newsplit['Raw Category Name'][train_index] = name
    all_object_counts_newsplit['Object Instances'][train_index] = train_instances_of_name
    all_object_counts_newsplit['Videos'][train_index] = train_videos_of_name
    all_object_counts_newsplit['Scenes'][train_index] = train_scenes_of_name
    all_object_counts_newsplit['Count of Scenes'][train_index] = len(all_object_counts_newsplit['Scenes'][train_index])
    all_object_counts_newsplit['Count of Videos'][train_index] = len(all_object_counts_newsplit['Videos'][train_index])
    all_object_counts_newsplit['Count of Instances'][train_index] = len(all_object_counts_newsplit['Object Instances'][train_index])
    all_object_counts_newsplit['Set'][train_index] = 'Train'
    all_object_counts_newsplit['Cluster'][train_index] = cluster

    val_index = i*3+1
    val_instances_of_name = set()
    val_videos_of_name = set()
    val_scenes_of_name = set()
    for val_object_key in val_instances.keys():
        if val_instances[val_object_key]['name'] == name:
            if val_instances[val_object_key]['scene'] in zero_shot_val_new_scenes:
                val_scenes_of_name.add(val_instances[val_object_key]['scene'])
                val_videos_of_name.update(val_instances[val_object_key]['videos'])
                val_instances_of_name.add(val_object_key)
    all_object_counts_newsplit['Raw Category Name'][val_index] = name
    all_object_counts_newsplit['Object Instances'][val_index] = val_instances_of_name
    all_object_counts_newsplit['Videos'][val_index] = val_videos_of_name
    all_object_counts_newsplit['Scenes'][val_index] = val_scenes_of_name
    all_object_counts_newsplit['Count of Scenes'][val_index] = len(all_object_counts_newsplit['Scenes'][val_index])
    all_object_counts_newsplit['Count of Videos'][val_index] = len(all_object_counts_newsplit['Videos'][val_index])
    all_object_counts_newsplit['Count of Instances'][val_index] = len(all_object_counts_newsplit['Object Instances'][val_index])
    all_object_counts_newsplit['Set'][val_index] = 'Val'
    all_object_counts_newsplit['Cluster'][val_index] = cluster


    test_index = i*3+2
    test_instances_of_name = set()
    test_videos_of_name = set()
    test_scenes_of_name = set()
    for test_object_key in val_instances.keys():
        if val_instances[test_object_key]['name'] == name:
            if val_instances[test_object_key]['scene'] in zero_shot_test_new_scenes:
                test_scenes_of_name.add(val_instances[test_object_key]['scene'])
                test_videos_of_name.update(val_instances[test_object_key]['videos'])
                test_instances_of_name.add(test_object_key)
    all_object_counts_newsplit['Raw Category Name'][test_index] = name
    all_object_counts_newsplit['Object Instances'][test_index] = test_instances_of_name
    all_object_counts_newsplit['Videos'][test_index] = test_videos_of_name
    all_object_counts_newsplit['Scenes'][test_index] = test_scenes_of_name
    all_object_counts_newsplit['Count of Scenes'][test_index] = len(all_object_counts_newsplit['Scenes'][test_index])
    all_object_counts_newsplit['Count of Videos'][test_index] = len(all_object_counts_newsplit['Videos'][test_index])
    all_object_counts_newsplit['Count of Instances'][test_index] = len(all_object_counts_newsplit['Object Instances'][test_index])
    all_object_counts_newsplit['Set'][test_index] = 'Test'
    all_object_counts_newsplit['Cluster'][test_index] = cluster




df = pd.DataFrame.from_dict(all_object_counts_newsplit)



all_cluster_counts_newsplit = {'Raw Category Names': {},
                     'Object Instances': {},
                     'Videos': {},
                     'Scenes': {},
                     'Count of Scenes': {},
                     'Count of Videos': {},
                     'Count of Instances': {},
                     'Set': {},
                     'Cluster': {}
                     }

for i in range(n_clusters):
    train_index = i*3
    all_cluster_counts_newsplit['Raw Category Names'][train_index] = set(df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Raw Category Name'].tolist())
    all_cluster_counts_newsplit['Object Instances'][train_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Object Instances'].tolist())
    all_cluster_counts_newsplit['Videos'][train_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Videos'].tolist())
    all_cluster_counts_newsplit['Scenes'][train_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Train')]['Scenes'].tolist())
    all_cluster_counts_newsplit['Count of Scenes'][train_index] = len(all_cluster_counts_newsplit['Scenes'][train_index])
    all_cluster_counts_newsplit['Count of Videos'][train_index] = len(all_cluster_counts_newsplit['Videos'][train_index])
    all_cluster_counts_newsplit['Count of Instances'][train_index] = len(all_cluster_counts_newsplit['Object Instances'][train_index])
    all_cluster_counts_newsplit['Set'][train_index] = 'Train'
    all_cluster_counts_newsplit['Cluster'][train_index] = i

    val_index = i*3+1
    all_cluster_counts_newsplit['Raw Category Names'][val_index] = set(df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Raw Category Name'].tolist())
    all_cluster_counts_newsplit['Object Instances'][val_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Object Instances'].tolist())
    all_cluster_counts_newsplit['Videos'][val_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Videos'].tolist())
    all_cluster_counts_newsplit['Scenes'][val_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Val')]['Scenes'].tolist())
    all_cluster_counts_newsplit['Count of Scenes'][val_index] = len(all_cluster_counts_newsplit['Scenes'][val_index])
    all_cluster_counts_newsplit['Count of Videos'][val_index] = len(all_cluster_counts_newsplit['Videos'][val_index])
    all_cluster_counts_newsplit['Count of Instances'][val_index] = len(all_cluster_counts_newsplit['Object Instances'][val_index])
    all_cluster_counts_newsplit['Set'][val_index] = 'Val'
    all_cluster_counts_newsplit['Cluster'][val_index] = i


    test_index = i*3+2
    all_cluster_counts_newsplit['Raw Category Names'][test_index] = set(df.loc[(df['Cluster']==i) & (df['Set']=='Test')]['Raw Category Name'].tolist())
    all_cluster_counts_newsplit['Object Instances'][test_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Test')]['Object Instances'].tolist())
    all_cluster_counts_newsplit['Videos'][test_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Test')]['Videos'].tolist())
    all_cluster_counts_newsplit['Scenes'][test_index] = set.union(*df.loc[(df['Cluster']==i) & (df['Set']=='Test')]['Scenes'].tolist())
    all_cluster_counts_newsplit['Count of Scenes'][test_index] = len(all_cluster_counts_newsplit['Scenes'][test_index])
    all_cluster_counts_newsplit['Count of Videos'][test_index] = len(all_cluster_counts_newsplit['Videos'][test_index])
    all_cluster_counts_newsplit['Count of Instances'][test_index] = len(all_cluster_counts_newsplit['Object Instances'][test_index])
    all_cluster_counts_newsplit['Set'][test_index] = 'Test'
    all_cluster_counts_newsplit['Cluster'][test_index] = i


df_cluster = pd.DataFrame.from_dict(all_cluster_counts_newsplit)

palette = ['#1f78b4', '#fc8e62', '#b2df8a']
plt.figure()
bar = sns.barplot(data=df_cluster, x="Cluster", y="Count of Videos", hue="Set", palette=palette)
bar.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right', rotation_mode='anchor')
for xtick, color in zip(bar.get_xticklabels(), list(sns.color_palette("husl",n_clusters))):
    xtick.set_color(color)
fig = bar.get_figure()
fig.savefig(f"Cluster_Bar_VideoCount{seed}_newsplit.png", bbox_inches='tight')
plt.close()


plt.figure()
bar = sns.barplot(data=df_cluster, x="Cluster", y="Count of Instances", hue="Set", palette=palette)
bar.set_xticklabels([cluster_labels[i] for i in range(n_clusters)], rotation=45, ha='right', rotation_mode='anchor')
for xtick, color in zip(bar.get_xticklabels(), list(sns.color_palette("husl",n_clusters))):
    xtick.set_color(color)
fig = bar.get_figure()
fig.savefig(f"Cluster_Bar_InstanceCount{seed}_newsplit.png", bbox_inches='tight')
plt.close()


zero_shot_val_new_scenes = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Val')]['Scenes'].tolist())
zero_shot_val_new_videos = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Val')]['Videos'].tolist())
zero_shot_val_new_instances = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Val')]['Object Instances'].tolist())

print(f"Zero-Shot Val (new) Scenes: {len(zero_shot_val_new_scenes)}")
print(f"Zero-Shot Val (new) Videos: {len(zero_shot_val_new_videos)}")
print(f"Zero-Shot Val (new) Instances: {len(zero_shot_val_new_instances)}")

zero_shot_test_new_scenes = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Test')]['Scenes'].tolist())
zero_shot_test_new_videos = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Test')]['Videos'].tolist())
zero_shot_test_new_instances = set.union(*df_cluster[df_cluster['Cluster'].isin(zero_shot_clusters) & (df_cluster['Set']=='Test')]['Object Instances'].tolist())

print(f"Zero-Shot Test (new) Scenes: {len(zero_shot_test_new_scenes)}")
print(f"Zero-Shot Test (new) Videos: {len(zero_shot_test_new_videos)}")
print(f"Zero-Shot Test (new) Instances: {len(zero_shot_test_new_instances)}")