# test-program
## クラスタリング用プログラム（python）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
sns.set()

data = pd.read_csv('data.csv')

def Clustering(data, x):
    feature_x = data.columns[data.dtypes != np.object] # 数値カラムの名前を抽出
    kind = data.columns[data.dtypes == np.object].values # 商品名を変数
    scaler = StandardScaler() # 数値データを正規化
    n_df = scaler.fit_transform(data[feature_x])
    n_df = pd.DataFrame(n_df, columns=feature_x) # DataFrame型に直す
    n_df['customer_id'] = kind # 商品名を付与
    n_df = n_df.loc[:, data.columns] # 元データとカラム順をそろえる
    dist = hierarchy.distance.pdist(n_df[feature_x], metric='euclidean') # 距離計算
    linkage = hierarchy.linkage(dist, method='ward') # クラスタリング
    threshold = x # デンドログラム作成
    plt.figure(figsize=(25, 15), dpi=300)
    dg = hierarchy.dendrogram(linkage, labels=kind, color_threshold=threshold, leaf_font_size=6)
    plt.hlines(threshold, 0, 100000, linestyles='dashed')
    plt.tight_layout()
    plt.tick_params(labelsize=15)
    plt.show()    
    cluster = hierarchy.fcluster(linkage, t=x, criterion='distance') # クラスタ数を確認
    n_df['Cluster'] = cluster
    data2 = data.copy()
    data2['Cluster'] = n_df['Cluster']
    print('クラスタ数 :', len(data2['Cluster'].value_counts()))
    print('クラスタ内容 :', data2['Cluster'].value_counts())
    data2.sort_values(['Cluster', 'customer_id'], inplace=True)
    return data2
