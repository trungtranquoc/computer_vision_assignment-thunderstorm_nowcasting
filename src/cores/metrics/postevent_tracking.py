import numpy as np
from sklearn.linear_model import TheilSenRegressor
from typing import Annotated
from tqdm.notebook import tqdm

class TrackCluster:
    def __init__(self, id: int):
        self.id = id

        # Use for early stopping criteria, if centroids do not change, we stop updating
        self.is_changed = True      # Indicates whether the centroids have been updated
        self.is_actived = True     # Indicates whether the cluster is active

        self.centroids = np.empty((0, 3))  # Initialize empty centroids

    def inactive(self):
        """
        Mark this cluster as inactive
        """
        self.is_actived = False
        self.is_changed = False
        self.slope_x = 0
        self.slope_y = 0
        self.intercept_x = float('inf')           # Use inf to set all distance to this cluster to inf
        self.intercept_y = float('inf')

    def update_centroids(self, centroids: np.ndarray):
        # Update centroids and check if they have changed
        if not self.is_actived:
            return
        if set([tuple(row) for row in self.centroids.tolist()]) == set([tuple(row) for row in centroids.tolist()]):
            self.is_changed = False
            return
        
        try:
            self.time_window = (centroids[:, 2].min(), centroids[:, 2].max())
        except Exception as e:
            print("Error in updating centroids:", self.id, len(centroids), e)
            raise e

        self.centroids = centroids
        self.is_changed = True

    def _fit(self, points: np.ndarray, times: np.ndarray):
        model = TheilSenRegressor()
        model.fit(y=times, X=points)        # Orginally form: t = coef * x + intercept

        # Normalize to get slope and intercept in form: x = slope * t + intercept
        slope = 1 / model.coef_[0]
        intercept = -model.intercept_ / model.coef_[0]

        return slope, intercept

    def fit_trajectory(self):
        if not self.is_actived:
            return
        points_x = self.centroids[:, 0]      # Only get (x)
        points_y = self.centroids[:, 1]      # Only get (y)
        times = self.centroids[:, 2].reshape(-1, 1)  # Get (t)

        self.slope_x, self.intercept_x = self._fit(points_x.reshape(-1, 1), times)
        self.slope_y, self.intercept_y = self._fit(points_y.reshape(-1, 1), times)

    def get_params(self):
        return np.array([self.slope_x, self.slope_y, self.intercept_x, self.intercept_y], dtype=float)
    
    def get_distance(self, point: np.ndarray) -> float:
        """
        Get distance from a point to this cluster's trajectory
        """
        if not self.is_actived:
            return float('inf')
        t = point[2]
        x_pred =  (t - self.intercept_x) / self.slope_x
        y_pred = (t - self.intercept_y) / self.slope_y

        distance = np.sqrt((point[0] - x_pred) ** 2 + (point[1] - y_pred) ** 2)
        return distance
    
    def merge(self, other_cluster: 'TrackCluster'):
        """
        Merge another cluster into this cluster
        """
        if not self.is_actived or not other_cluster.is_actived:
            return
        combined_centroids = np.vstack((self.centroids, other_cluster.centroids))
        self.update_centroids(combined_centroids)

class PostEventClustering:
    """
    Perform post-event clustering on storm centroids to identify storm tracks, reference to the paper "A Method for Extracting Postevent Storm Tracks"

    Arguments:
        centroids (np.ndarray): np.ndarray of shape (N, 3), where each row is (x, y, t)
        max_window_time (int): Maximum time window to consider a point belongs to a cluster
        spatial_distance_threshold (float): Maximum spatial distance to consider a point belongs to a cluster
        clusters_merged_dict (dict[int, int]): A dictionary to keep track of merged clusters. This will used for scoring later by mapping original cluster IDs to merged cluster IDs.

    ### Algorithms
    1. Initialize clusters based on initial assignments.
    2. Iteratively perform the following until convergence or maximum epochs reached:
        - 2.1. For each cluster, fit the trajectory by Theil-Sen
        - 2.2. Re-associate points to nearest clusters.
        - 2.3. Perform post-processing: 
            - Drop clusters that have small number of points and re-associate those points.
            - Merge clusters that identical: those clusters such that all of the points of two clusters are in less than $D$ to both of the clusters.
    3. Finally, drop those points that are far from any clusters.
    """
    centroids: np.ndarray           # np.ndarray of shape (N, 3)
    clusters_assigned: list[int]    # List of length N, indicating which cluster each point is assigned to
    clusters: list[TrackCluster]
    max_window_time: Annotated[int, "Maximum time window to consider a point belongs to a cluster"]
    spatial_distance_threshold: Annotated[float, "Maximum spatial distance to consider a point belongs to a cluster"]
    clusters_merged_dict: dict[int, int]

    def __init__(self, centroids: np.ndarray, max_window_time: int = 5, spatial_distance_threshold: float = 5.0):
        self.centroids = centroids          # np.ndarray of shape (N, 3)
        self.max_window_time = max_window_time
        self.spatial_distance_threshold = spatial_distance_threshold
        self.clusters_merged_dict = {}

    def _get_distance_fast(self, clusters_params: np.ndarray, include_masked: bool = True) -> np.ndarray:
        """
        Compute distance matrix between each point and each cluster. Clusters params are represented as (slope_x, slope_y, intercept_x, intercept_y)
        This is the faster version that computes distances in a vectorized way.

        Returns:
            distance_matrix (np.ndarray): np.ndarray of shape (N, num_initial_clusters)
        """
        x_distanced = (self.centroids[:, 0].reshape(-1, 1) - (clusters_params[:, 0] * self.centroids[:, 2].reshape(-1, 1) + clusters_params[:, 2].reshape(1, -1))) ** 2
        y_distanced = (self.centroids[:, 1].reshape(-1, 1) - (clusters_params[:, 1] * self.centroids[:, 2].reshape(-1, 1) + clusters_params[:, 3].reshape(1, -1))) ** 2
        distance_matrix = np.sqrt(x_distanced + y_distanced)    # Shape (N, num_initial_clusters)

        # Mask distances based on time windows -- If point's time is outside cluster's time window +/- max_window_time, set distance to inf
        if include_masked:
            distance_matrix = np.where(
                (self.centroids[:, 2].reshape(-1, 1) >= np.array([cluster.time_window[0] for cluster in self.clusters]) - self.max_window_time) &
                (self.centroids[:, 2].reshape(-1, 1) <= np.array([cluster.time_window[1] for cluster in self.clusters]) + self.max_window_time),
                distance_matrix,
                float('inf')
            )

        return distance_matrix

    def _epoch_update(self) -> bool:
        # 2.1 Fit trajectories for each cluster
        for cluster in self.clusters:
            if not cluster.is_actived:
                continue
            cluster.update_centroids(self.centroids[np.array(self.clusters_assigned) == cluster.id])
            cluster.fit_trajectory()
        
        # 2.2 Re-assign points to nearest clusters
        # First, only move points to a new cluster if the spatial distance is less than threshold D, otherwise, keep the old assignment
        # masked_old_assignment = 
        # masked_reassigned = masked_spatial_distance

        clusters_params = np.array([cluster.get_params() for cluster in self.clusters])       # Stack them together for efficient computation
        distance_matrix = self._get_distance_fast(clusters_params)
        new_assignments = np.argmin(distance_matrix, axis=1).tolist()

        # Check if assignments have changed ---> If no, early stopping
        if new_assignments == self.clusters_assigned:
            return False

        # 2.3 Prunning & Merging: 
        # 2.3.1 Remove those clusters that have too few points
        counts = np.bincount(new_assignments, minlength=self.num_initial_clusters)
        for cluster_id, count in enumerate(counts):
            if count < 2:
                self.clusters[cluster_id].inactive()
                distance_matrix[:, cluster_id] = float('inf')  # Set distances to this cluster to inf

        # Reupdate assignments after pruning
        new_assignments = np.argmin(distance_matrix, axis=1).tolist()

        # 2.3.2 Merging: Merge clusters that have similar trajectories
        # First, from distance matrix, masking those points that are within spatial distance threshold of each cluster
        masked_spatial_distance = distance_matrix <= self.spatial_distance_threshold
        for i in range(self.num_initial_clusters):
            if not self.clusters[i].is_actived:
                continue
            
            centroids_idx = np.where(np.array(new_assignments) == i)[0]
            for j in range(i + 1, self.num_initial_clusters):
                if not self.clusters[j].is_actived:
                    continue
                
                # Not in the time boundary
                if not (self.clusters[i].time_window[1] < self.clusters[j].time_window[0] - self.max_window_time or
                        self.clusters[j].time_window[1] < self.clusters[i].time_window[0] - self.max_window_time):
                    continue

                other_centroids_idx = np.where(np.array(new_assignments) == j)[0]
                
                # Check if all points in cluster i are within spatial distance threshold to cluster j
                if np.all(masked_spatial_distance[centroids_idx, j]) and np.all(masked_spatial_distance[other_centroids_idx, i]):
                    print("\tMerge clusters:", i, "and", j)
                    self.clusters[i].merge(self.clusters[j])
                    self.clusters[j].inactive()
                    new_assignments = [i if assign == j else assign for assign in new_assignments]
                    
                    # Iteratively update the merged dict
                    self.clusters_merged_dict[j] = i
                    for key in list(self.clusters_merged_dict.keys()):
                        if self.clusters_merged_dict[key] == j:
                            self.clusters_merged_dict[key] = i

        # Do the update centroids after pruning and merging
        self.clusters_assigned = new_assignments

        return True

    def fit_transform(self, num_clusters: int, clusters_assigned: list[int], 
                      max_epochs: int = 100, show_progress: bool = True) -> list[int]:
        """
        Fit transform the clustering assignments. It return the final cluster assignments after post-event clustering. Some of the points may be marked as -1, indicating that they are not assigned to any cluster (dropped).

        Arguments:
            num_clusters (int): Initial number of clusters
            clusters_assigned (list[int]): Initial cluster assignments for each point
            max_epochs (int): Maximum number of epochs to run
            show_progress (bool): Whether to show progress bar
        """
        self.num_initial_clusters = num_clusters
        self.clusters = [TrackCluster(id) for id in range(num_clusters)]
        self.clusters_assigned = clusters_assigned

        pbar = tqdm(range(max_epochs), desc="Post-event clustering") if show_progress else range(max_epochs)
        for _ in pbar:
            changed = self._epoch_update()
            if not changed:
                print("No change detected, stopping early !")
                break

        # Finally, drop those points that are far from any clusters
        clusters_params = np.array([cluster.get_params() for cluster in self.clusters])
        distance_matrix = self._get_distance_fast(clusters_params)
        min_distances = np.min(distance_matrix, axis=1)
        for idx, distance in enumerate(min_distances):
            if distance > self.spatial_distance_threshold:
                self.clusters_assigned[idx] = -1    # Mark as unassigned
        
        return self.clusters_assigned