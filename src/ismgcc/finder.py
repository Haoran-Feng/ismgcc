from astropy import log
log.setLevel("INFO")
import pickle
import numpy as np
import os
from typing import List
from regions import Regions, PolygonSkyRegion, PolygonPixelRegion, PixCoord
from skimage.morphology import flood_fill
from skimage.measure import find_contours
import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MeanShift
from pathlib import Path
import hashlib
from scipy.spatial import KDTree
from itertools import combinations
import networkx as nx
from scipy.stats import norm, poisson
import time
from functools import wraps
import pyarrow
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import tqdm_joblib

def log_start_and_end(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        t1 = time.time()
        t1s = time.asctime()
        log.info(f"Start {name} at {t1s}")
        r = func(*args, **kwargs)
        t2 = time.time()
        t2s = time.asctime()
        elapse_time = t2 - t1
        log.info(f"Done at {t2s}, elapse time: {elapse_time:.2f} seconds")
        return r

    return wrapper

def read_gpickle(file):
    with open(file, "rb") as f:
        g = pickle.load(f)

    return g

def save_gpickle(file, obj):
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    

class DecomposedPPVStructureFinder:
    def _load_default_params(self):
        self.params = {
            "r" : 3,
            "bandwidth_coef": 0.5, 
            "spatial_distance_threshold": 1.5,
            "snr_th0": 0,
            "snr_th1": 5,
            "decision_boundary": 0.5,
            "community_resolution": 0.01,
            "minimal_number_of_pixels": 16,
        }

    # decorator
    def _save_or_load_cache_file(dependent_params: List[str], name_suffix: str, file_format:str):
        params = sorted(dependent_params)
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                cache_path = Path("./.cache/")
                if not cache_path.exists():
                    cache_path.mkdir()

                hv = self.decomp_table_hash_value
                param_strings = []
                for p in params:
                    if p not in self.params.keys():
                        raise ValueError(f"No such parameter: {p}")

                    value = self.params[p]
                    value_string = f"{value:.2f}"
                    param_strings.append(p + value_string)
                
                identity_string = hv + "_" + "_".join(param_strings) + "_" + name_suffix

                file = Path(cache_path / (identity_string + file_format))
                # ===================
                # create a lock for file written
                # to avoid duplicated computing of the same cache file
                # in multi process 
                # 
                lock_file = Path(str(file) + ".lock")
                while lock_file.exists():
                    log.info(f"Locked by: {lock_file}, waitting...")
                    time.sleep(10) # if there's another process trying to create the cache file now, wait.
                
                log.info(f"No lock file: {lock_file}")
                #
                if file.exists():
                    # read cached file
                    reader = {
                        ".arrow": pd.read_feather,
                        ".gpickle": read_gpickle,
                        ".pickle": read_gpickle,
                    }[file_format]
                    log.info(f"Read cached {name_suffix} file: {file}")
                    r = reader(file)
                else: 
                    # ==========
                    # if we are creating the cache file, create a lock file
                    lock_file.touch(exist_ok=False)
                    log.info(f"Create lock file: {lock_file}")

                    #
                    r = func(self, *args, **kwargs)
                    writer = {
                        ".arrow": lambda file, obj: obj.to_feather(file),
                        ".gpickle": save_gpickle,
                        ".pickle": save_gpickle,
                    }[file_format]
                    log.info(f"Save cached {name_suffix} file: {file}")
                    writer(file, r)
                    # ======
                    # after we have save the cache file, release the lock
                    lock_file.unlink(missing_ok=False)
                    log.info(f"Release lock file: {lock_file}")
                    # 

                return r
            return wrapper
        return decorator
            
    def __init__(self, decomp_table: pd.DataFrame, params: dict=None, n_jobs: int=None) -> None:
        self._load_default_params()
        if params:
            for k, v in params.items():
                if k not in self.params.keys():
                    raise ValueError(f"Param '{k}' is not a valid parameter!")
                self.params[k] = v

        self.n_jobs = n_jobs if n_jobs else (int(os.cpu_count() * 0.5) - 1) 
        self.decomp_table = decomp_table
        self.decomp_table_hash_value = hashlib.sha256(pd.util.hash_pandas_object(self.decomp_table, index=True).values).hexdigest()[:5]


    @log_start_and_end
    @_save_or_load_cache_file(["spatial_distance_threshold"], "kdtree_and_adj_table", ".pickle")
    def _prepare_kdtree_and_adj_table(self):
        xy = self.decomp_table[["x_pos", "y_pos"]].values
        kdtree = KDTree(xy)
        pairs = kdtree.query_pairs(self.params["spatial_distance_threshold"])
        adj_table = {
            i: set() for i in self.decomp_table.index.tolist()
        }
        for a, b in pairs:
            adj_table[a].add(b)
            adj_table[b].add(a)

        return kdtree, adj_table
    
    def find_structures(self):
        self.kdtree, self.adj_table = self._prepare_kdtree_and_adj_table()
        self.vcluster_table = self._prepare_v_cluster_table()
        self.v_graph = self._prepare_v_graph()
        self.graph = self._prepare_ppv_weight_graph()

        log.info("Start finding connected components...")
        indices, ids = self.ppv_clusters_from_graph(self.graph)

        vgdf = pd.DataFrame({"cluster_id0": ids, "index": indices})
        outdf = self.decomp_table.join(vgdf.set_index("index"))
        s = self.check_multi_vclusters_in_single_pixel(outdf, self.v_graph, "cluster_id0")
        outdf = outdf.join(s)

        to_split = outdf.query("multi_vclusters_of_cluster_id0")["cluster_id0"].unique()
        outdf["cluster_id1"] = 0

        n1 = len(np.unique(ids))
        n2 = len(to_split)
        log.info(f"Found {n1} connected components, in which {n2} need(s) splitting with modularity communities.")

        parallel_indices = []
        for i in to_split:
            indices = outdf.query("cluster_id0 == @i").index.tolist()
            parallel_indices.append(indices)

        with tqdm_joblib(tqdm(desc=f"Finding communities...", total=len(to_split))) as progress_bar:
            rst = Parallel(n_jobs=self.n_jobs)(delayed(self.split_by_community)(self.graph, indices, self.params["community_resolution"]) for indices in parallel_indices)

        n3 = 0
        for i, indices in enumerate(parallel_indices):
            new_id = rst[i]
            outdf.loc[indices, "cluster_id1"] = new_id
            n3 += len(np.unique(new_id))
        
        log.info(f"The {n2} complex connected components are splitted into {n3} communities.")

        uids = self.cantor_paring(outdf["cluster_id0"], outdf["cluster_id1"])
        outdf["uid"] = uids

        unique_uids = np.sort(outdf["uid"].unique())
        id_map = {
            unique_uids[i]: i + 1
            for i in range(len(unique_uids))
        }
        outdf["serial_id"] = outdf["uid"].apply(lambda x: id_map[x])
        s = self.check_multi_vclusters_in_single_pixel(outdf, self.v_graph, "serial_id")
        outdf = outdf.join(s)

        df = self.format_output_table(outdf, "serial_id", self.params["minimal_number_of_pixels"])
        return df

    def get_output_suffix(self):
        """Return a string that contains values of all parameters
        """
        keys = sorted(list(self.params.keys()))
        kv_paris = []
        for k in keys:
            value = self.params[k]
            s = f"{k}={value}"
            kv_paris.append(s)
        return "-".join(kv_paris)

    @log_start_and_end
    @_save_or_load_cache_file(["r", "bandwidth_coef"], "vcluster_table", ".arrow")
    def _prepare_v_cluster_table(self):
        r = self.params["r"]
        bw_coef = self.params["bandwidth_coef"]
        xydf = self.decomp_table[["x_pos", "y_pos"]].drop_duplicates()
        xytodo = [(row[0], row[1]) for row in xydf.values]
        indices = []
        Xs = []
        xys = []
        bws = []
        for x, y in tqdm(xytodo):
            adf = self._get_aperture_df(x, y)
            X = adf[["VLSR"]].values
            if len(X) > 0:
                Xs.append(X)
                idx = adf.index.to_numpy()
                indices.append(idx)
                xys.append((x, y))
                bw = bw_coef * np.average(adf["vel_disp"], weights=adf["amp"])
                bws.append(bw)

        def func(X, bw):
            model = MeanShift(bandwidth=bw)
            labels = model.fit_predict(X)

            return labels
        
        results = []
        with tqdm_joblib(tqdm(desc=f"Run Mean-Shift, r={r}", total=len(Xs))) as progress_bar:
            rst = Parallel(n_jobs=self.n_jobs)(delayed(func)(X, bw) for X, bw in zip(Xs, bws))

        results += rst
            
        outdf = pd.DataFrame({"xy": xys * 1, "indices": indices * 1, "labels": results})
        outdf["r"] = r
        outdf["x"] = outdf["xy"].apply(lambda x: x[0])
        outdf["y"] = outdf["xy"].apply(lambda x: x[1])
        outdf = outdf.drop(columns=["xy"])
        outdf["n_comps_in_ap"] = outdf["indices"].apply(len)

        return outdf

    @log_start_and_end
    @_save_or_load_cache_file(["r", "bandwidth_coef"], "v_graph", ".gpickle")
    def _prepare_v_graph(self):
        nodes = self.decomp_table.index.tolist()
        edges = []
        for i, row in tqdm(self.vcluster_table.iterrows(), total=len(self.vcluster_table)):
            indices = row["indices"]
            labels = row["labels"]
            ulabels = np.unique(labels)
            for j in ulabels:
                cindices = indices[labels == j]
                edges += list(combinations(cindices, 2))

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        return g

    def _get_aperture_df(self, x, y):
        r = self.params["r"]
        indices = self.kdtree.query_ball_point([x, y], r)
        return self.decomp_table.loc[indices, :]

    @log_start_and_end
    @_save_or_load_cache_file(["r", "bandwidth_coef", "spatial_distance_threshold", "decision_boundary", "snr_th0", "snr_th1"], "ppv_weight_graph", ".gpickle")
    def _prepare_ppv_weight_graph(self):
        nodes = self.decomp_table.index.tolist()
        n_connected, n_share_apertures, connected_df = self._prepare_vc_n_connect_table()


        edges_w = []
        edge_attr_dict = {}

        db = self.params["decision_boundary"]
        snr_th0 = self.params["snr_th0"]
        snr_th1 = self.params["snr_th1"]

        poisson_lambda = connected_df["n_connected"].mean()
        log.info(f"Poisson Lambda={poisson_lambda:.4f}")
        for ab in tqdm(n_connected.keys(), total=len(n_connected)):
            #############################################################
            # Calculate the probability of two VCs are coherent with each other
            #############################################################
            i1, i2 = tuple(ab)
            amp1 = self.decomp_table.loc[i1, "amp"]
            amp2 = self.decomp_table.loc[i2, "amp"]
            rms1 = self.decomp_table.loc[i1, "rms"]
            rms2 = self.decomp_table.loc[i2, "rms"]
            snr1 = amp1 / rms1
            snr2 = amp2 / rms2

            # 1. whether two VCs are assigned into the same cluster during the MeanShift Velocity Clustering Operation for at least 80% of the shared apertures.
            prob1 = 0.0 if (n_connected[ab] / n_share_apertures[ab]) < db else 1.0
            if prob1 == 0.0:
                continue

            # 2. how sure the two VCs are real signal
            # snr_th0 = 3
            # snr_th1 = 3

            # prob2vc1 = sigmoid(snr1 -snr_th1) * (np.sign(snr1 - snr_th0) + 1) / 2
            prob2vc1 = norm.cdf(snr1 - snr_th1, loc=0, scale=1) * (np.sign(snr1 - snr_th0) + 1) / 2
            prob2vc2 = norm.cdf(snr2 - snr_th1, loc=0, scale=1) * (np.sign(snr2 - snr_th0) + 1) / 2

            prob2 = prob2vc1 * prob2vc2
            if np.isclose(prob2, 0.0):
                continue

            # 3. how confident are we sure they are physically connected 
            n = n_connected[ab]
            prob3 = poisson.cdf(n, mu=poisson_lambda)

            weight = prob1 * prob2 * prob3
            #############################################################
            edges_w.append((*tuple(ab), weight))

            edge_attr_dict[(i1, i2)] = {
                "p_real1": prob2vc1,
                "p_real2": prob2vc2,
                "p_v_coh": prob3,
                "n_connected": n,
                "n_shared": n_share_apertures[ab]
            }

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_weighted_edges_from(edges_w, "weight")
        nx.set_edge_attributes(g, edge_attr_dict)

        return g

    @log_start_and_end
    @_save_or_load_cache_file(["r", "bandwidth_coef"], "n_connected", ".pickle")
    def _prepare_vc_n_connect_table(self):
        n_connected = {}       # record the number of times that two nodes are clustered into the same vel cluster
        n_share_apertures = {} # record the number of times that two nodes are circled by the same aperture
        for i, row in tqdm(self.vcluster_table.iterrows(), total=len(self.vcluster_table)):
            indices = row["indices"]
            labels = row["labels"]
            ulabels = np.unique(labels)
            for j in ulabels:
                cindices = indices[labels == j]
                for a, b in combinations(cindices, 2):
                    if b in self.adj_table[a]:
                        k = frozenset((a, b))
                        n_connected[k] = n_connected.get(k, 0) + 1

            for a, b in combinations(indices, 2):
                if b in self.adj_table[a]:
                    k = frozenset((a, b))
                    n_share_apertures[k] = n_share_apertures.get(k, 0) + 1


        t1 = pd.Series(n_share_apertures)
        t2 = pd.Series(n_connected)
        connected_df = pd.DataFrame({"n_shared": t1, "n_connected": t2})
        ratio = connected_df["n_connected"] / connected_df["n_shared"]
        connected_df["ratio"] = ratio

        return n_connected, n_share_apertures, connected_df

    @staticmethod
    def cantor_paring(a, b):
        return (a + b) / 2 * (a + b + 1) + b
    
    @staticmethod
    def split_by_community(graph, indices, resolution):
        g = nx.subgraph(graph, indices)
        communities = nx.community.greedy_modularity_communities(g, weight="weight",resolution=resolution)
        # communities = nx.community.louvain_communities(g, weight="weight",resolution=resolution, seed=1234)
        result = {}
        for i, cc_indices in enumerate(communities):
            for j in cc_indices:
                result[j] = i + 1

        return [result[i] for i in indices]

    @staticmethod
    def format_output_table(table, cluster_id_col: str, min_n_pix: int):
        counts = table[cluster_id_col].value_counts()
        counts.name = "vc_count"
        outdf= table.join(counts, on=cluster_id_col)
        t = outdf.groupby(cluster_id_col)[["x_pos", "y_pos"]]
        pix_count = t.apply(lambda x: len(x.drop_duplicates()))
        pix_count.name = "pix_count"
        outdf = outdf.join(pix_count, on=cluster_id_col)
        outdf[f"has{min_n_pix}pix"] = outdf["pix_count"] >= min_n_pix
        outdf = outdf.reset_index(drop=False)

        n_valid_structures = len(outdf.loc[outdf[f"has{min_n_pix}pix"], cluster_id_col].unique())
        n_total = len(outdf[cluster_id_col].unique())
        log.info(f"Total number of structures: {n_total}, Number of structures with at least {min_n_pix} pixels: {n_valid_structures}.")
        
        uids = np.sort(outdf[cluster_id_col].unique())
        uids2 = uids.copy()
        np.random.shuffle(uids2)
        id_map = {
            uids[i]: uids2[i]
            for i in range(len(uids))
        }
        id_map
        outdf["shuffle_id"] = outdf[cluster_id_col].replace(id_map)

        cols = outdf.columns.to_list()
        if "GLON" in cols and "GLAT" in cols:
            first_cols = ["GLON", "GLAT", "VLSR", "index", cluster_id_col, "shuffle_id"]
        elif "RA" in cols and "DEC" in cols:
            first_cols = ["RA", "DEC", "VLSR", "index", cluster_id_col, "shuffle_id"]
        else:
            return outdf
        
        for c in first_cols:
            cols.remove(c)

        outdf = outdf[first_cols + cols]

        return outdf

    @staticmethod
    def ppv_clusters_from_graph(graph: nx.Graph):
        ccs = list(nx.connected_components(graph)) # connected components of the graph
        cluster_ids = []
        indices = []
        for i, c in enumerate(ccs):
            cluster_ids.append([i + 1] * len(c))
            indices.append(list(c))

        ids = np.concatenate(cluster_ids)
        indices = np.concatenate(indices)
        return indices, ids

    @staticmethod
    def check_multi_vclusters_in_single_pixel(table: pd.DataFrame, v_graph: nx.Graph, cluster_id_col_name: str):
        cols = ["x_pos", "y_pos", cluster_id_col_name]
        outdf = table[cols].copy()

        def func(g):
            indices = g.index.tolist()
            for a, b in combinations(indices, 2):
                if b not in v_graph[a]:
                    return True

            return False
            
        i = outdf[cols].duplicated(keep=False)
        # --------------------------------------------------------------
        if not i.any():
            outdf[f"multi_vclusters_of_{cluster_id_col_name}"] = False
            return outdf[f"multi_vclusters_of_{cluster_id_col_name}"]
        # modified with Ou Xiangyu
        # --------------------------------------------------------------

        rst = outdf.loc[i, cols].groupby(cols).apply(func)
        rst.name = f"multi_vclusters_of_{cluster_id_col_name}"
        outdf = outdf.join(rst, on=cols, how="left")
        outdf[rst.name] = outdf[rst.name].apply(lambda x: False if np.isnan(x) else x).astype(bool)
        return outdf[rst.name]

