# src/features.py
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def load_logs(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def window_aggregate(df, window_minutes=60):
    # produces per-MSISDN aggregated features over sliding/rolling windows
    df = df.copy()
    df = df.sort_values('timestamp')
    df['window'] = (df['timestamp'].astype('int64') // (window_minutes*60*10**9))
    groups = df.groupby(['caller','window'])
    feats=[]
    for (caller,window), g in groups:
        call_count = (g['event']=='CALL').sum()
        sms_count = (g['event']=='SMS').sum()
        data_count = (g['event']=='DATA').sum()
        unique_callees = g['callee'].nunique()
        avg_dur = g['duration'].replace(0,np.nan).mean() if call_count>0 else 0
        interarrival = g['timestamp'].diff().dt.total_seconds().dropna()
        inter_mean = interarrival.mean() if len(interarrival)>0 else np.nan
        domain_entropy = compute_domain_entropy(g['domain'].fillna(''))
        imei_shared = len(g['imei'].unique())
        feats.append({
            'msisdn': caller,
            'window': window,
            'call_count': call_count,
            'sms_count': sms_count,
            'data_count': data_count,
            'unique_callees': unique_callees,
            'avg_duration': avg_dur,
            'interarrival_mean': inter_mean,
            'domain_entropy': domain_entropy,
            'imei_shared_count': imei_shared,
            'label': g['label'].mode().iloc[0] if 'label' in g.columns else 'normal'
        })
    return pd.DataFrame(feats)

def compute_domain_entropy(domains):
    if domains.empty: return 0.0
    s = domains[domains!='']
    if s.empty: return 0.0
    counts = s.value_counts(normalize=True)
    return -(counts * np.log2(counts)).sum()

def build_call_graph(df, window_minutes=60):
    # build aggregated graph aggregated by window; returns a NetworkX multigraph per window (or combined)
    df = df.copy()
    df['window'] = (df['timestamp'].astype('int64') // (window_minutes*60*10**9))
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        src = row['caller']
        dst = row['callee'] if row['callee']!='' else None
        if dst is None: continue
        if not G.has_node(src):
            G.add_node(src)
        if not G.has_node(dst):
            G.add_node(dst)
        # add or update edge attributes
        if G.has_edge(src,dst):
            G[src][dst]['count'] += 1
            G[src][dst]['dur_sum'] += row['duration']
        else:
            G.add_edge(src,dst, count=1, dur_sum=row['duration'])
    # compute node features
    for n in G.nodes():
        G.nodes[n]['degree'] = G.degree(n)
        G.nodes[n]['in_deg'] = G.in_degree(n)
        G.nodes[n]['out_deg'] = G.out_degree(n)
    return G

def extract_graph_features(G):
    # return a DataFrame of node features from graph G
    feats=[]
    for n in G.nodes():
        d = G.nodes[n]
        feats.append({
            'msisdn': n,
            'degree': d.get('degree',0),
            'in_deg': d.get('in_deg',0),
            'out_deg': d.get('out_deg',0),
            'clustering': nx.clustering(nx.Graph(G), n)
        })
    return pd.DataFrame(feats)

def prepare_tabular_features(windowed_df, graph_feats):
    df = windowed_df.merge(graph_feats, on='msisdn', how='left').fillna(0)
    # basic transforms
    df['call_sms_ratio'] = df['call_count'] / (df['sms_count']+1e-6)
    df['calls_per_callee'] = df['call_count'] / (df['unique_callees']+1e-6)
    # fill na
    df = df.fillna(0)
    # scale numeric columns for models that need it
    num_cols = ['call_count','sms_count','data_count','unique_callees','avg_duration','interarrival_mean','domain_entropy','imei_shared_count','degree','in_deg','out_deg','clustering','call_sms_ratio','calls_per_callee']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

if __name__ == "__main__":
    df = load_logs("data/raw_logs.csv").head(9000)
    win = window_aggregate(df, window_minutes=60)
    G = build_call_graph(df)
    gfeats = extract_graph_features(G)
    final = prepare_tabular_features(win, gfeats)
    final.to_csv("data/engineered_features.csv", index=False)
    print("Saved engineered features to data/engineered_features.csv")
