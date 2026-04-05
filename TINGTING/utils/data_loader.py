# bold_dataset_cv.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io
import pandas as pd
class BoldSequenceDatasetOneLabel(Dataset):

    def __init__(self, data_dir, label_csv, data_ext="*.csv"):
        super().__init__()
        self.data_dir = data_dir

        pattern = os.path.join(data_dir, data_ext)
        self.data_paths = sorted(glob.glob(pattern))
        if len(self.data_paths) == 0:
            raise RuntimeError(f"No data files found in {data_dir} with pattern {pattern}")

        y = np.loadtxt(label_csv, delimiter=',')
        if y.ndim > 1:
            y = y.squeeze()
        self.labels = torch.from_numpy(y.astype(np.int64))   # [T]

        x0 = np.loadtxt(self.data_paths[0], delimiter=',', dtype=np.float32)
        if x0.ndim == 1:
            x0 = x0[:, None]  # (T,) -> (T,1)
        T_data, N_data = x0.shape
        T_label = self.labels.shape[0]

        if T_data != T_label and N_data == T_label:
            x0 = x0.T
            T_data, N_data = x0.shape

        if T_data != T_label:
            raise RuntimeError(f"Time length mismatch between data ({T_data}) and label ({T_label})")

        self.T = T_data
        self.N = N_data

        self._first_sample = torch.from_numpy(x0)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        if idx == 0 and self._first_sample is not None:
            x = self._first_sample.clone()
        else:
            x_np = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
            if x_np.ndim == 1:
                x_np = x_np[:, None]
            T_data, N_data = x_np.shape
            if T_data != self.T and N_data == self.T:
                x_np = x_np.T
            x = torch.from_numpy(x_np)

        return x, self.labels.clone()   


class BoldSequenceDatasetOneLabelFC(Dataset):

    def __init__(
        self,
        data_dir,
        label_csv,
        data_ext="*.csv",
        fc_type="corr",
        edge_ratio=0.4,
        cache_fc=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.fc_type = fc_type
        self.edge_ratio = edge_ratio
        self.cache_fc = cache_fc

        pattern = os.path.join(data_dir, data_ext)
        self.data_paths = sorted(glob.glob(pattern))
        if len(self.data_paths) == 0:
            raise RuntimeError(f"No data files found in {data_dir} with pattern {pattern}")

        y = np.loadtxt(label_csv, delimiter=",")
        if y.ndim > 1:
            y = y.squeeze()
        self.labels = torch.from_numpy(y.astype(np.int64))  # [T]

        x0 = np.loadtxt(self.data_paths[0], delimiter=",", dtype=np.float32)
        if x0.ndim == 1:
            x0 = x0[:, None]  # (T,) -> (T,1)
        T_data, N_data = x0.shape
        T_label = self.labels.shape[0]

        if T_data != T_label and N_data == T_label:
            x0 = x0.T
            T_data, N_data = x0.shape

        if T_data != T_label:
            raise RuntimeError(
                f"Time length mismatch between data ({T_data}) and label ({T_label})"
            )

        self.T = T_data
        self.N = N_data

        self._first_sample = torch.from_numpy(x0)

        self.fc_cache = [None] * len(self.data_paths) if cache_fc else None

    def __len__(self):
        return len(self.data_paths)

    def _compute_fc(self, x_np: np.ndarray) -> np.ndarray:

        T, N = x_np.shape

        if self.fc_type == "cov":
            x_center = x_np - x_np.mean(axis=0, keepdims=True)  # [T, N]
            # (N, T) @ (T, N) / (T-1) -> (N, N)
            fc = x_center.T @ x_center / max(T - 1, 1)
        else:
            fc = np.corrcoef(x_np, rowvar=False)

        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

        iu = np.triu_indices(N, k=1)
        abs_edges = np.abs(fc[iu])  

        if abs_edges.size == 0:
            return fc.astype(np.float32)

        k = int(np.floor(self.edge_ratio * abs_edges.size))
        if k < 1:
            k = 1

        threshold = np.partition(abs_edges, -k)[-k]


        mask = np.abs(fc) >= threshold

        np.fill_diagonal(mask, False)

        fc_thr = fc * mask  

        return fc_thr.astype(np.float32)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

 
        if idx == 0 and self._first_sample is not None:
            x = self._first_sample.clone()
        else:
            x_np = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
            if x_np.ndim == 1:
                x_np = x_np[:, None]
            T_data, N_data = x_np.shape
            if T_data != self.T and N_data == self.T:
                x_np = x_np.T
            x = torch.from_numpy(x_np)


        if self.fc_cache is not None and self.fc_cache[idx] is not None:
            fc = self.fc_cache[idx]
        else:
            fc_np = self._compute_fc(x.numpy())  # [N, N]
            fc = torch.from_numpy(fc_np)
            if self.fc_cache is not None:
                self.fc_cache[idx] = fc

        # [T, N], [T], [N, N]
        return x, self.labels.clone(), fc


import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io


class BoldSequenceDatasetOneLabelSC(Dataset):
    

    def __init__(
        self,
        data_dir,
        label_csv,
        sc_root_dir,
        fallback_sc_path,
        data_ext="*.csv",
        required_T=405,
        cache_sc=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.required_T = required_T
        self.cache_sc = cache_sc

        pattern = os.path.join(data_dir, data_ext)
        all_paths = sorted(glob.glob(pattern))

        bold_paths = []
        for p in all_paths:
            name = os.path.basename(p)

            if "task-WM" not in name or "acq-RL" not in name:
                continue

            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T_data, N_data = x_np.shape
            if T_data != required_T:
                continue

            bold_paths.append(p)

        if len(bold_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files found in {data_dir} with pattern {pattern}, "
                f"task-WM & acq-RL & T={required_T} (after removing first row/column)."
            )

        self.data_paths = bold_paths

        y = np.loadtxt(label_csv, delimiter=",")
        if y.ndim > 1:
            y = y.squeeze()
        self.labels = torch.from_numpy(y.astype(np.int64))  # [T_label]

        T_label = self.labels.shape[0]
        if T_label != required_T:
            raise RuntimeError(
                f"Label length {T_label} != required_T {required_T}"
            )

        self.T = required_T

        x0_np = self._load_bold_array(self.data_paths[0])
        if x0_np is None:
            raise RuntimeError("First BOLD sample failed to load / clean.")

        T0, N0 = x0_np.shape
        if T0 != self.T:
            raise RuntimeError(
                f"First sample time length {T0} != {self.T} after cleaning."
            )

        self.N = N0
        self._first_sample = torch.from_numpy(x0_np.astype(np.float32))

        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        fb_sc = fb_mat["aal116_sift_radius2_count_connectivity"]  # [N_sc,N_sc]
        fb_sc = self._normalize_sc(fb_sc)
        self.fallback_sc = fb_sc  # torch.FloatTensor

        self.sc_cache = [None] * len(self.data_paths) if cache_sc else None

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _load_bold_array(path: str) -> np.ndarray | None:

        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            print(f"[WARN] Invalid CSV (dim<2): {path}")
            return None

        data = raw[1:, 1:]  # [R-1, C-1]

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        vfunc = np.vectorize(to_float)
        data = vfunc(data).astype(np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0:
            print(f"[WARN] Empty after cleaning: {path}")
            return None

        return data

    @staticmethod
    def _extract_subj_id(path):

        name = os.path.basename(path)
        m = re.search(r"(sub-[A-Za-z0-9]+)", name)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        """
        sc_np: [N_sc, N_sc] numpy array
        返回: row-normalized 的 torch.FloatTensor
        """
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)  # [N_sc,1]
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    def _load_sc_for_subject(self, subj_id: str) -> torch.Tensor:

        if subj_id is None:
            return self.fallback_sc.clone()

        subj_dir = os.path.join(self.sc_root_dir, subj_id)
        if not os.path.isdir(subj_dir):
            return self.fallback_sc.clone()

        pattern = os.path.join(
            subj_dir, f"{subj_id}_space-T1w_desc-preproc_msmtconnectome.mat"
        )
        mat_paths = glob.glob(pattern)

        if len(mat_paths) == 0:
            mat_paths = glob.glob(os.path.join(subj_dir, "*.mat"))

        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_file_path = mat_paths[0]
        sc_data = scipy.io.loadmat(sc_file_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            # 字段不存在，fallback
            return self.fallback_sc.clone()

        sc_np = sc_data[key]
        sc_tensor = self._normalize_sc(sc_np)
        return sc_tensor

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        # ---------- BOLD: x: [T, N] ----------
        if idx == 0 and self._first_sample is not None:
            x = self._first_sample.clone()
        else:
            x_np = self._load_bold_array(data_path)
            if x_np is None:
                x_np = self._first_sample.numpy()
            T_data, N_data = x_np.shape
            if T_data != self.T:
                x_np = x_np[: self.T, :]
            x = torch.from_numpy(x_np.astype(np.float32))

        # ---------- SC: [N_sc, N_sc] ----------
        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            subj_id = self._extract_subj_id(data_path)
            sc = self._load_sc_for_subject(subj_id)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        #  [T, N], [T], [N_sc, N_sc]
        return x, self.labels.clone(), sc.clone()


class BoldSequenceDatasetOneLabelSC_re(Dataset):

    def __init__(
        self,
        data_dir,
        label_csv,
        sc_root_dir,
        fallback_sc_path,
        data_ext="*.csv",
        required_T=405,
        cache_sc=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.required_T = required_T
        self.cache_sc = cache_sc

        pattern = os.path.join(data_dir, data_ext)
        all_paths = sorted(glob.glob(pattern))

        bold_paths = []
        for p in all_paths:
            name = os.path.basename(p)

            if "task-WM" not in name or "acq-LR" not in name:
                continue

            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T_data, N_data = x_np.shape
            if T_data != required_T:
                continue

            bold_paths.append(p)

        if len(bold_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files found in {data_dir} with pattern {pattern}, "
                f"task-WM & acq-LR & T={required_T} (after removing first row/column)."
            )

        self.data_paths = bold_paths

        y = np.loadtxt(label_csv, delimiter=",")
        if y.ndim > 1:
            y = y.squeeze()
        self.labels = torch.from_numpy(y.astype(np.int64))  # [T_label]

        T_label = self.labels.shape[0]
        if T_label != required_T:
            raise RuntimeError(
                f"Label length {T_label} != required_T {required_T}"
            )

        self.T = required_T

        x0_np = self._load_bold_array(self.data_paths[0])
        if x0_np is None:
            raise RuntimeError("First BOLD sample failed to load / clean.")

        T0, N0 = x0_np.shape
        if T0 != self.T:
            raise RuntimeError(
                f"First sample time length {T0} != {self.T} after cleaning."
            )

        self.N = N0
        self._first_sample = torch.from_numpy(x0_np.astype(np.float32))

        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        fb_sc = fb_mat["aal116_sift_radius2_count_connectivity"]  # [N_sc,N_sc]
        fb_sc = self._normalize_sc(fb_sc)
        self.fallback_sc = fb_sc  # torch.FloatTensor

        self.sc_cache = [None] * len(self.data_paths) if cache_sc else None

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _load_bold_array(path: str) -> np.ndarray | None:

        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            print(f"[WARN] Invalid CSV (dim<2): {path}")
            return None

        data = raw[1:, 1:]  # [R-1, C-1]

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        vfunc = np.vectorize(to_float)
        data = vfunc(data).astype(np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0:
            print(f"[WARN] Empty after cleaning: {path}")
            return None

        return data

    @staticmethod
    def _extract_subj_id(path):

        name = os.path.basename(path)
        m = re.search(r"(sub-[A-Za-z0-9]+)", name)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:

        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)  # [N_sc,1]
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    def _load_sc_for_subject(self, subj_id: str) -> torch.Tensor:

        if subj_id is None:
            return self.fallback_sc.clone()

        subj_dir = os.path.join(self.sc_root_dir, subj_id)
        if not os.path.isdir(subj_dir):
            return self.fallback_sc.clone()

        pattern = os.path.join(
            subj_dir, f"{subj_id}_space-T1w_desc-preproc_msmtconnectome.mat"
        )
        mat_paths = glob.glob(pattern)

        if len(mat_paths) == 0:
            mat_paths = glob.glob(os.path.join(subj_dir, "*.mat"))

        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_file_path = mat_paths[0]
        sc_data = scipy.io.loadmat(sc_file_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            return self.fallback_sc.clone()

        sc_np = sc_data[key]
        sc_tensor = self._normalize_sc(sc_np)
        return sc_tensor

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        # ---------- BOLD: x: [T, N] ----------
        if idx == 0 and self._first_sample is not None:
            x = self._first_sample.clone()
        else:
            x_np = self._load_bold_array(data_path)
            if x_np is None:
                x_np = self._first_sample.numpy()
            T_data, N_data = x_np.shape
            if T_data != self.T:
                x_np = x_np[: self.T, :]
            x = torch.from_numpy(x_np.astype(np.float32))

        # ---------- SC: [N_sc, N_sc] ----------
        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            subj_id = self._extract_subj_id(data_path)
            sc = self._load_sc_for_subject(subj_id)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        #  [T, N], [T], [N_sc, N_sc]
        return x, self.labels.clone(), sc.clone()



import os
import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io


class HCPYA(Dataset):
    """
    HCP-YA multi-task BOLD dataset (subject-level classification).

    Rules:
      - label parsed from filename: task-{EMOTION,GAMBLING,LANGUAGE,MOTOR,RELATIONAL,SOCIAL,WM} (7 classes)
      - output sequence length fixed to T_fix=220
      - if cleaned T < 220: drop this sample
      - if cleaned T > 220: center-crop to 220 by removing both head & tail
      - BOLD CSV cleaning:
          * remove first row & first col
          * str->float, invalid->NaN->0
          * remove all-zero rows
      - __getitem__ returns:
          x     : [220, N] float32
          label : scalar int64
          sc    : [N_sc, N_sc] float32 row-normalized
    """

    TASKS = ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
    TASK2ID = {t: i for i, t in enumerate(TASKS)}

    def __init__(
        self,
        data_dir,
        label_csv=None,  # kept for backward compatibility, not used
        sc_root_dir=None,
        fallback_sc_path=None,
        data_ext="*.csv",
        T_fix=220,
        cache_sc=True,
        verbose=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.T_fix = int(T_fix)
        self.cache_sc = bool(cache_sc)
        self.verbose = bool(verbose)

        if self.fallback_sc_path is None:
            raise ValueError("fallback_sc_path must be provided.")

        # ---- preload fallback SC ----
        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in fb_mat:
            raise KeyError(f"Key '{key}' not found in fallback SC mat: {self.fallback_sc_path}")
        self.fallback_sc = self._normalize_sc(fb_mat[key])  # torch.FloatTensor

        # ---- scan all BOLD files ----
        pattern = os.path.join(self.data_dir, self.data_ext)
        all_paths = sorted(glob.glob(pattern))

        data_paths = []
        labels = []
        lengths = []

        for p in all_paths:
            name = os.path.basename(p)

            # parse task label
            task = self._extract_task_name(name)
            if task is None or task not in self.TASK2ID:
                continue
            y = self.TASK2ID[task]

            # load + clean
            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T, N = x_np.shape
            lengths.append(T)

            # drop too-short
            if T < self.T_fix:
                continue

            data_paths.append(p)
            labels.append(y)

        if len(data_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files found in {data_dir} with pattern {pattern}. "
                f"Need task-* in {self.TASKS} and cleaned T >= {self.T_fix}."
            )

        self.data_paths = data_paths
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))  # [num_samples]
        self.num_classes = len(self.TASKS)

        # ---- determine N from first valid sample (after crop) ----
        x0 = self._load_bold_array(self.data_paths[0])
        if x0 is None:
            raise RuntimeError("First valid BOLD sample failed to load.")
        if x0.shape[0] < self.T_fix:
            raise RuntimeError("Internal error: first valid sample shorter than T_fix.")

        self.N = int(x0.shape[1])
        self.T = self.T_fix  # fixed output length

        # ---- SC cache ----
        self.sc_cache = [None] * len(self.data_paths) if self.cache_sc else None

        # ---- optional summary ----
        if self.verbose:
            # class counts
            counts = {t: 0 for t in self.TASKS}
            for lab in labels:
                counts[self.TASKS[int(lab)]] += 1

            lengths_np = np.asarray(lengths, dtype=np.int32) if len(lengths) > 0 else None
            print(f"[HCPYA] Dataset size: {len(self.data_paths)} samples")
            print(f"[HCPYA] Class counts: {counts}")
            print(f"[HCPYA] Unique labels: {sorted(list(set(labels)))}")
            if lengths_np is not None and lengths_np.size > 0:
                print(
                    f"[HCPYA] Cleaned length stats (before filtering/cropping): "
                    f"min={lengths_np.min()}  p25={np.percentile(lengths_np,25):.0f}  "
                    f"median={np.percentile(lengths_np,50):.0f}  p75={np.percentile(lengths_np,75):.0f}  "
                    f"max={lengths_np.max()}"
                )
            print(f"[HCPYA] Output fixed length T_fix={self.T_fix}")

    def __len__(self):
        return len(self.data_paths)

    # --------- parse task from filename ---------
    @staticmethod
    def _extract_task_name(filename: str):
        m = re.search(r"task-([A-Za-z0-9]+)", filename)
        return m.group(1).upper() if m else None

    # --------- load + clean BOLD CSV ---------
    @staticmethod
    def _load_bold_array(path: str):
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            print(f"[WARN] Invalid CSV (dim<2): {path}")
            return None

        # remove first row/col
        data = raw[1:, 1:]

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        data = np.vectorize(to_float)(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # remove all-zero rows
        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0 or data.ndim != 2:
            print(f"[WARN] Empty/invalid after cleaning: {path}")
            return None

        return data

    # --------- subject id from filename ---------
    @staticmethod
    def _extract_subj_id(path: str):
        name = os.path.basename(path)
        m = re.search(r"(sub-[A-Za-z0-9]+)", name)
        return m.group(1) if m else None

    # --------- row-normalize SC ---------
    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    # --------- load SC for subject (fallback if missing) ---------
    def _load_sc_for_subject(self, subj_id: str) -> torch.Tensor:
        if subj_id is None or self.sc_root_dir is None:
            return self.fallback_sc.clone()

        subj_dir = os.path.join(self.sc_root_dir, subj_id)
        if not os.path.isdir(subj_dir):
            return self.fallback_sc.clone()

        pattern = os.path.join(subj_dir, f"{subj_id}_space-T1w_desc-preproc_msmtconnectome.mat")
        mat_paths = glob.glob(pattern)

        if len(mat_paths) == 0:
            mat_paths = glob.glob(os.path.join(subj_dir, "*.mat"))
        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_file_path = mat_paths[0]
        sc_data = scipy.io.loadmat(sc_file_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            return self.fallback_sc.clone()

        return self._normalize_sc(sc_data[key])

    # --------- center-crop to T_fix ---------
    def _fix_length_to_T(self, x_np: np.ndarray) -> np.ndarray:
        """
        Make time length exactly T_fix:
        - if T > T_fix: center crop
        - if T < T_fix: zero-pad at the end
        """
        T, N = x_np.shape

        if T == self.T_fix:
            return x_np

        # ---- case 1: longer than T_fix → center crop ----
        if T > self.T_fix:
            start = (T - self.T_fix) // 2
            end = start + self.T_fix
            return x_np[start:end, :]

        # ---- case 2: shorter than T_fix → zero pad ----
        pad_len = self.T_fix - T
        pad = np.zeros((pad_len, N), dtype=x_np.dtype)
        return np.concatenate([x_np, pad], axis=0)


    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        x_np = self._load_bold_array(data_path)
        if x_np is None:
            # extreme fallback: use first sample
            x_np = self._load_bold_array(self.data_paths[0])

        x_np = self._fix_length_to_T(x_np)
        if x_np is None:
            # if still too short (should not), pad or fallback; here fallback to zeros
            x_np = np.zeros((self.T_fix, self.N), dtype=np.float32)

        # ensure float32 tensor [T_fix, N]
        x = torch.from_numpy(x_np.astype(np.float32))

        # label scalar
        label = self.labels[idx].clone()

        # SC
        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            subj_id = self._extract_subj_id(data_path)
            sc = self._load_sc_for_subject(subj_id)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        return x, label, sc.clone()



class HCPA(Dataset):
    """
    HCP-YA multi-task BOLD dataset (subject-level classification).

    Rules:
      - label parsed from filename: task-{EMOTION,GAMBLING,LANGUAGE,MOTOR,RELATIONAL,SOCIAL,WM} (7 classes)
      - output sequence length fixed to T_fix=220
      - if cleaned T < 220: drop this sample
      - if cleaned T > 220: center-crop to 220 by removing both head & tail
      - BOLD CSV cleaning:
          * remove first row & first col
          * str->float, invalid->NaN->0
          * remove all-zero rows
      - __getitem__ returns:
          x     : [220, N] float32
          label : scalar int64
          sc    : [N_sc, N_sc] float32 row-normalized
    """

    TASKS = ["CARIT", "FACENAME", "REST", "VISMOTOR"]
    TASK2ID = {t: i for i, t in enumerate(TASKS)}

    def __init__(
        self,
        data_dir,
        label_csv=None,  # kept for backward compatibility, not used
        sc_root_dir=None,
        fallback_sc_path=None,
        data_ext="*.csv",
        T_fix=220,
        cache_sc=True,
        verbose=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.T_fix = int(T_fix)
        self.cache_sc = bool(cache_sc)
        self.verbose = bool(verbose)

        if self.fallback_sc_path is None:
            raise ValueError("fallback_sc_path must be provided.")

        # ---- preload fallback SC ----
        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in fb_mat:
            raise KeyError(f"Key '{key}' not found in fallback SC mat: {self.fallback_sc_path}")
        self.fallback_sc = self._normalize_sc(fb_mat[key])  # torch.FloatTensor

        # ---- scan all BOLD files ----
        pattern = os.path.join(self.data_dir, self.data_ext)
        all_paths = sorted(glob.glob(pattern))

        data_paths = []
        labels = []
        lengths = []

        for p in all_paths:
            name = os.path.basename(p)

            # parse task label
            task = self._extract_task_name(name)
            if task is None or task not in self.TASK2ID:
                continue
            y = self.TASK2ID[task]

            # load + clean
            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T, N = x_np.shape
            lengths.append(T)

            # drop too-short
            if T < self.T_fix:
                continue

            data_paths.append(p)
            labels.append(y)

        if len(data_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files found in {data_dir} with pattern {pattern}. "
                f"Need task-* in {self.TASKS} and cleaned T >= {self.T_fix}."
            )

        self.data_paths = data_paths
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))  # [num_samples]
        self.num_classes = len(self.TASKS)

        # ---- determine N from first valid sample (after crop) ----
        x0 = self._load_bold_array(self.data_paths[0])
        if x0 is None:
            raise RuntimeError("First valid BOLD sample failed to load.")
        if x0.shape[0] < self.T_fix:
            raise RuntimeError("Internal error: first valid sample shorter than T_fix.")

        self.N = int(x0.shape[1])
        self.T = self.T_fix  # fixed output length

        # ---- SC cache ----
        self.sc_cache = [None] * len(self.data_paths) if self.cache_sc else None

        # ---- optional summary ----
        if self.verbose:
            # class counts
            counts = {t: 0 for t in self.TASKS}
            for lab in labels:
                counts[self.TASKS[int(lab)]] += 1

            lengths_np = np.asarray(lengths, dtype=np.int32) if len(lengths) > 0 else None
            print(f"[HCPYA] Dataset size: {len(self.data_paths)} samples")
            print(f"[HCPYA] Class counts: {counts}")
            print(f"[HCPYA] Unique labels: {sorted(list(set(labels)))}")
            if lengths_np is not None and lengths_np.size > 0:
                print(
                    f"[HCPYA] Cleaned length stats (before filtering/cropping): "
                    f"min={lengths_np.min()}  p25={np.percentile(lengths_np,25):.0f}  "
                    f"median={np.percentile(lengths_np,50):.0f}  p75={np.percentile(lengths_np,75):.0f}  "
                    f"max={lengths_np.max()}"
                )
            print(f"[HCPYA] Output fixed length T_fix={self.T_fix}")

    def __len__(self):
        return len(self.data_paths)

    # --------- parse task from filename ---------
    @staticmethod
    def _extract_task_name(filename: str):
        m = re.search(r"task-([A-Za-z0-9]+)", filename)
        return m.group(1).upper() if m else None

    # --------- load + clean BOLD CSV ---------
    @staticmethod
    def _load_bold_array(path: str):
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            print(f"[WARN] Invalid CSV (dim<2): {path}")
            return None

        # remove first row/col
        data = raw[1:, 1:]

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        data = np.vectorize(to_float)(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # remove all-zero rows
        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0 or data.ndim != 2:
            print(f"[WARN] Empty/invalid after cleaning: {path}")
            return None

        return data

    # --------- subject id from filename ---------
    @staticmethod
    def _extract_subj_id(path: str):
        name = os.path.basename(path)
        m = re.search(r"(sub-[A-Za-z0-9]+)", name)
        return m.group(1) if m else None

    # --------- row-normalize SC ---------
    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    # --------- load SC for subject (fallback if missing) ---------
    def _load_sc_for_subject(self, subj_id: str) -> torch.Tensor:
        if subj_id is None or self.sc_root_dir is None:
            return self.fallback_sc.clone()

        subj_dir = os.path.join(self.sc_root_dir, subj_id)
        if not os.path.isdir(subj_dir):
            return self.fallback_sc.clone()

        pattern = os.path.join(subj_dir, f"{subj_id}_space-T1w_desc-preproc_msmtconnectome.mat")
        mat_paths = glob.glob(pattern)

        if len(mat_paths) == 0:
            mat_paths = glob.glob(os.path.join(subj_dir, "*.mat"))
        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_file_path = mat_paths[0]
        sc_data = scipy.io.loadmat(sc_file_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            return self.fallback_sc.clone()

        return self._normalize_sc(sc_data[key])

    # --------- center-crop to T_fix ---------
    def _fix_length_to_T(self, x_np: np.ndarray) -> np.ndarray:
        """
        Make time length exactly T_fix:
        - if T > T_fix: center crop
        - if T < T_fix: zero-pad at the end
        """
        T, N = x_np.shape

        if T == self.T_fix:
            return x_np

        # ---- case 1: longer than T_fix → center crop ----
        if T > self.T_fix:
            start = (T - self.T_fix) // 2
            end = start + self.T_fix
            return x_np[start:end, :]

        # ---- case 2: shorter than T_fix → zero pad ----
        pad_len = self.T_fix - T
        pad = np.zeros((pad_len, N), dtype=x_np.dtype)
        return np.concatenate([x_np, pad], axis=0)


    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        x_np = self._load_bold_array(data_path)
        if x_np is None:
            # extreme fallback: use first sample
            x_np = self._load_bold_array(self.data_paths[0])

        x_np = self._fix_length_to_T(x_np)
        if x_np is None:
            # if still too short (should not), pad or fallback; here fallback to zeros
            x_np = np.zeros((self.T_fix, self.N), dtype=np.float32)

        # ensure float32 tensor [T_fix, N]
        x = torch.from_numpy(x_np.astype(np.float32))

        # label scalar
        label = self.labels[idx].clone()

        # SC
        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            subj_id = self._extract_subj_id(data_path)
            sc = self._load_sc_for_subject(subj_id)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        return x, label, sc.clone()

class UKB(Dataset):
    """
    UKB multi-task BOLD dataset (subject-level classification).

    Rules:
      - label parsed from filename: task-{hariri,rest} (2 classes)
      - output sequence length fixed to T_fix (default 220)
      - BOLD CSV cleaning:
          * remove first row & first col
          * str->float, invalid->NaN->0
          * remove all-zero rows
      - __getitem__ returns:
          x     : [T_fix, N] float32
          label : scalar int64
          sc    : [N_sc, N_sc] float32 row-normalized
    """

    TASKS = ["hariri", "rest"]
    TASK2ID = {t: i for i, t in enumerate(TASKS)}

    def __init__(
        self,
        data_dir,
        label_csv=None,  # kept for backward compatibility, not used
        sc_root_dir=None,
        fallback_sc_path=None,
        data_ext="*.csv",
        T_fix=220,
        cache_sc=True,
        verbose=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.T_fix = int(T_fix)
        self.cache_sc = bool(cache_sc)
        self.verbose = bool(verbose)

        if self.fallback_sc_path is None:
            raise ValueError("fallback_sc_path must be provided.")

        # ---- preload fallback SC ----
        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in fb_mat:
            raise KeyError(f"Key '{key}' not found in fallback SC mat: {self.fallback_sc_path}")
        self.fallback_sc = self._normalize_sc(fb_mat[key])  # torch.FloatTensor

        # ---- scan all BOLD files ----
        pattern = os.path.join(self.data_dir, self.data_ext)
        all_paths = sorted(glob.glob(pattern))

        data_paths = []
        labels = []
        lengths = []

        for p in all_paths:
            name = os.path.basename(p)

            # parse task label 
            task = self._extract_task_name(name)
            if task is None or task not in self.TASK2ID:
                continue
            y = self.TASK2ID[task]

            # load + clean
            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T, N = x_np.shape
            lengths.append(T)

            data_paths.append(p)
            labels.append(y)

        if len(data_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files found in {data_dir} with pattern {pattern}. "
                f"Need task-* in {self.TASKS}. "
                f"(Tip: check task parsing + CSV cleaning.)"
            )

        self.data_paths = data_paths
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))  # [num_samples]
        self.num_classes = len(self.TASKS)

        # ---- determine N from first valid sample (after cleaning) ----
        x0 = self._load_bold_array(self.data_paths[0])
        if x0 is None:
            raise RuntimeError("First valid BOLD sample failed to load.")
        self.N = int(x0.shape[1])
        self.T = self.T_fix  # fixed output length

        # ---- SC cache ----
        self.sc_cache = [None] * len(self.data_paths) if self.cache_sc else None

        # ---- optional summary ----
        if self.verbose:
            counts = {t: 0 for t in self.TASKS}
            for lab in labels:
                counts[self.TASKS[int(lab)]] += 1

            lengths_np = np.asarray(lengths, dtype=np.int32) if len(lengths) > 0 else None
            print(f"[UKB] Dataset size: {len(self.data_paths)} samples")
            print(f"[UKB] Class counts: {counts}")
            print(f"[UKB] Unique labels: {sorted(list(set(labels)))}")
            if lengths_np is not None and lengths_np.size > 0:
                print(
                    f"[UKB] Cleaned length stats (before padding/cropping): "
                    f"min={lengths_np.min()}  p25={np.percentile(lengths_np,25):.0f}  "
                    f"median={np.percentile(lengths_np,50):.0f}  p75={np.percentile(lengths_np,75):.0f}  "
                    f"max={lengths_np.max()}"
                )
            print(f"[UKB] Output fixed length T_fix={self.T_fix}")

    def __len__(self):
        return len(self.data_paths)

    # --------- parse task from filename ---------
    @staticmethod
    def _extract_task_name(filename: str):
        """
        Parse task-xxx from filename and return lower-case task name.
        e.g. task-rest -> "rest"
        """
        m = re.search(r"task-([A-Za-z0-9]+)", filename)
        return m.group(1).lower() if m else None

    # --------- load + clean BOLD CSV ---------
    @staticmethod
    def _load_bold_array(path: str):
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            print(f"[WARN] Invalid CSV (dim<2): {path}")
            return None

        # remove first row/col
        data = raw[1:, 1:]

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        data = np.vectorize(to_float)(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # remove all-zero rows
        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0 or data.ndim != 2:
            # keep warning but do not crash
            # print(f"[WARN] Empty/invalid after cleaning: {path}")
            return None

        return data

    # --------- subject id from filename ---------
    @staticmethod
    def _extract_subj_id(path: str):
        name = os.path.basename(path)
        m = re.search(r"(sub-[A-Za-z0-9]+)", name)
        return m.group(1) if m else None

    # --------- row-normalize SC ---------
    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    # --------- load SC for subject (fallback if missing) ---------
    def _load_sc_for_subject(self, subj_id: str) -> torch.Tensor:

        if subj_id is None or self.sc_root_dir is None:
            return self.fallback_sc.clone()

        subj_dir = os.path.join(self.sc_root_dir, "ses-2")
        if not os.path.isdir(subj_dir):
            return self.fallback_sc.clone()

        pattern = os.path.join(subj_dir, f"{subj_id}_space-T1w_desc-preproc_msmtconnectome.mat")
        mat_paths = glob.glob(pattern)

        if len(mat_paths) == 0:
            mat_paths = glob.glob(os.path.join(subj_dir, "*.mat"))
        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_file_path = mat_paths[0]
        sc_data = scipy.io.loadmat(sc_file_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            return self.fallback_sc.clone()

        return self._normalize_sc(sc_data[key])

    # --------- fix length to T_fix ---------
    def _fix_length_to_T(self, x_np: np.ndarray) -> np.ndarray:
        """
        Make time length exactly T_fix:
        - if T > T_fix: center crop
        - if T < T_fix: zero-pad at the end
        """
        T, N = x_np.shape

        if T == self.T_fix:
            return x_np

        if T > self.T_fix:
            start = (T - self.T_fix) // 2
            end = start + self.T_fix
            return x_np[start:end, :]

        pad_len = self.T_fix - T
        pad = np.zeros((pad_len, N), dtype=x_np.dtype)
        return np.concatenate([x_np, pad], axis=0)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]

        x_np = self._load_bold_array(data_path)
        if x_np is None:
            # extreme fallback: try first sample
            x_np = self._load_bold_array(self.data_paths[0])
        if x_np is None:
            # final fallback: zeros
            x_np = np.zeros((self.T_fix, self.N), dtype=np.float32)

        x_np = self._fix_length_to_T(x_np)

        x = torch.from_numpy(x_np.astype(np.float32))  # [T_fix, N]

        label = self.labels[idx].clone()               # scalar int64

        # SC
        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            subj_id = self._extract_subj_id(data_path)
            sc = self._load_sc_for_subject(subj_id)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        return x, label, sc.clone()


class ADNI(Dataset):
    """
    ADNI BOLD dataset (subject-level classification).

    - BOLD CSV filename contains subject id like: sub-002S0295_...
    - labels read from a CSV:
        Subject column: 002_S_0295  (need remove '_' => 002S0295)
        Group column  : AD / MCI / CN (3 classes)

    Output:
      x     : [T_fix, N] float32
      label : scalar int64 (CN=0, MCI=1, AD=2 by default)
      sc    : [N_sc, N_sc] float32 row-normalized
    """

    GROUPS = ["CN", "MCI", "AD"]
    GROUP2ID = {g: i for i, g in enumerate(GROUPS)}

    def __init__(
        self,
        data_dir,
        label_csv,                 
        sc_root_dir=None,
        fallback_sc_path=None,
        data_ext="*.csv",
        T_fix=220,
        cache_sc=True,
        verbose=True,
        subject_col="Subject",
        group_col="Group",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.T_fix = int(T_fix)
        self.cache_sc = bool(cache_sc)
        self.verbose = bool(verbose)

        if label_csv is None:
            raise ValueError("label_csv must be provided for ADNI dataset.")
        self.label_csv = label_csv
        self.subject_col = subject_col
        self.group_col = group_col

        if self.fallback_sc_path is None:
            raise ValueError("fallback_sc_path must be provided.")

        # ---- preload fallback SC ----
        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in fb_mat:
            raise KeyError(f"Key '{key}' not found in fallback SC mat: {self.fallback_sc_path}")
        self.fallback_sc = self._normalize_sc(fb_mat[key])

        # ---- read label table ----
        df = pd.read_csv(self.label_csv)

        if self.subject_col not in df.columns or self.group_col not in df.columns:
            raise KeyError(f"CSV must contain columns: {self.subject_col}, {self.group_col}")

        # Build map: "002S0295" -> label_id
        subj2label = {}
        dropped_unknown_group = 0
        dropped_missing = 0

        for _, row in df.iterrows():
            subj_raw = str(row[self.subject_col])
            grp_raw = str(row[self.group_col]).strip().upper()

            # normalize group
            if grp_raw not in self.GROUP2ID:
                dropped_unknown_group += 1
                continue

            # normalize subject: 002_S_0295 -> 002S0295
            subj_norm = self._normalize_subject_from_csv(subj_raw)
            if subj_norm is None:
                dropped_missing += 1
                continue

            subj2label[subj_norm] = self.GROUP2ID[grp_raw]

        if len(subj2label) == 0:
            raise RuntimeError("No valid subject labels parsed from label_csv.")

        # ---- scan BOLD files and match to subj2label ----
        pattern = os.path.join(self.data_dir, self.data_ext)
        all_paths = sorted(glob.glob(pattern))

        data_paths = []
        labels = []
        lengths = []
        unmatched = 0

        for p in all_paths:
            name = os.path.basename(p)

            subj_norm = self._extract_subj_id_from_bold_filename(name)  # "002S0295" or None
            if subj_norm is None:
                continue

            if subj_norm not in subj2label:
                unmatched += 1
                continue

            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T, N = x_np.shape
            lengths.append(T)

            data_paths.append(p)
            labels.append(subj2label[subj_norm])

        if len(data_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files matched labels.\n"
                f"data_dir={self.data_dir}, pattern={pattern}\n"
                f"label_csv={self.label_csv}\n"
                f"Tip: check filename subject pattern 'sub-002S0295' and CSV Subject '002_S_0295'."
            )

        self.data_paths = data_paths
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.num_classes = len(self.GROUPS)

        # determine N from first sample 
        x0 = self._load_bold_array(self.data_paths[0])
        if x0 is None:
            raise RuntimeError("First valid BOLD sample failed to load.")
        self.N = int(x0.shape[1])
        self.T = self.T_fix

        # SC cache
        self.sc_cache = [None] * len(self.data_paths) if self.cache_sc else None

        # summary
        if self.verbose:
            # counts
            cnt = {g: 0 for g in self.GROUPS}
            for lab in labels:
                cnt[self.GROUPS[int(lab)]] += 1
            lengths_np = np.asarray(lengths, dtype=np.int32) if len(lengths) > 0 else None

            print(f"[ADNI] Dataset size: {len(self.data_paths)} samples")
            print(f"[ADNI] Class counts: {cnt}")
            print(f"[ADNI] Unique labels: {sorted(list(set(labels)))}")
            print(f"[ADNI] Unmatched BOLD files (no label): {unmatched}")
            print(f"[ADNI] Dropped CSV rows (unknown group): {dropped_unknown_group}")
            print(f"[ADNI] Dropped CSV rows (bad subject): {dropped_missing}")
            if lengths_np is not None and lengths_np.size > 0:
                print(
                    f"[ADNI] Cleaned length stats (before padding/cropping): "
                    f"min={lengths_np.min()}  p25={np.percentile(lengths_np,25):.0f}  "
                    f"median={np.percentile(lengths_np,50):.0f}  p75={np.percentile(lengths_np,75):.0f}  "
                    f"max={lengths_np.max()}"
                )
            print(f"[ADNI] Output fixed length T_fix={self.T_fix}")

    def __len__(self):
        return len(self.data_paths)

    # ---------- subject normalize (CSV) ----------
    @staticmethod
    def _normalize_subject_from_csv(subj: str):
        """
        CSV Subject: "002_S_0295" -> "002S0295"
        """
        if subj is None:
            return None
        s = str(subj).strip()
        if s == "" or s.lower() == "nan":
            return None
        s = s.replace("_", "")
        # some CSVs might have spaces
        s = s.replace(" ", "")
        return s

    # ---------- subject extract (BOLD filename) ----------
    @staticmethod
    def _extract_subj_id_from_bold_filename(filename: str):
        """
        Find subject id from filename:
          "sub-002S0295_task-..." -> "002S0295"
          "sub-{002S0295}_..."    -> "002S0295"  (handles braces)
        """
        m = re.search(r"sub-\{?([0-9]{3}S[0-9]{4})\}?", filename)
        if m:
            return m.group(1)
        return None

    # ---------- load + clean BOLD ----------
    @staticmethod
    def _load_bold_array(path: str):
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            return None

        data = raw[1:, 1:]  # remove first row/col

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        data = np.vectorize(to_float)(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # remove all-zero rows
        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0 or data.ndim != 2:
            return None

        return data

    # ---------- row-normalize SC ----------
    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    # ---------- SC load ----------
    def _load_sc_for_subject(self, subj_norm: str) -> torch.Tensor:

        if subj_norm is None or self.sc_root_dir is None:
            return self.fallback_sc.clone()

        subj_dir = self.sc_root_dir
        if not os.path.isdir(subj_dir):
            return self.fallback_sc.clone()

        mat_paths = glob.glob(os.path.join(subj_dir, f"*{subj_norm}*.mat"))
        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_data = scipy.io.loadmat(mat_paths[0])
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            return self.fallback_sc.clone()

        return self._normalize_sc(sc_data[key])

    # ---------- fix length ----------
    def _fix_length_to_T(self, x_np: np.ndarray) -> np.ndarray:
        T, N = x_np.shape
        if T == self.T_fix:
            return x_np
        if T > self.T_fix:
            start = (T - self.T_fix) // 2
            end = start + self.T_fix
            return x_np[start:end, :]
        pad_len = self.T_fix - T
        pad = np.zeros((pad_len, N), dtype=x_np.dtype)
        return np.concatenate([x_np, pad], axis=0)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        name = os.path.basename(data_path)

        x_np = self._load_bold_array(data_path)
        if x_np is None:
            x_np = np.zeros((self.T_fix, self.N), dtype=np.float32)
        x_np = self._fix_length_to_T(x_np)
        x = torch.from_numpy(x_np.astype(np.float32))

        label = self.labels[idx].clone()

        # SC: get subj id from filename
        subj_norm = self._extract_subj_id_from_bold_filename(name)

        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            sc = self._load_sc_for_subject(subj_norm)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        return x, label, sc.clone()


class PPMI(Dataset):
    """
    PPMI BOLD dataset (subject-level classification).

    - subject id from filename:
        try patterns:
          1) sub-<PATNO>  (e.g., sub-1234_...)
          2) patno-<PATNO>
          3) plain number token inside filename (fallback)
    - labels read from CSV:
        PATNO column -> subject id (int)
        COHORT_DEFINITION column -> label string:
            Healthy Control / Prodromal / Parkinson's Disease

    Output:
      x     : [T_fix, N] float32
      label : scalar int64 (HC=0, Prodromal=1, PD=2)
      sc    : [N_sc, N_sc] float32 row-normalized
    """

    COHORTS = ["Healthy Control", "Prodromal", "Parkinson's Disease"]
    COHORT2ID = {c.upper(): i for i, c in enumerate(COHORTS)}  # case-insensitive

    def __init__(
        self,
        data_dir,
        label_csv,                
        sc_root_dir=None,
        fallback_sc_path=None,
        data_ext="*.csv",
        T_fix=220,
        cache_sc=True,
        verbose=True,
        patno_col="PATNO",
        label_col="COHORT_DEFINITION",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.T_fix = int(T_fix)
        self.cache_sc = bool(cache_sc)
        self.verbose = bool(verbose)

        if label_csv is None:
            raise ValueError("label_csv must be provided for PPMI dataset.")
        self.label_csv = label_csv
        self.patno_col = patno_col
        self.label_col = label_col

        if self.fallback_sc_path is None:
            raise ValueError("fallback_sc_path must be provided.")

        # ---- preload fallback SC ----
        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        key = "aal116_sift_radius2_count_connectivity"
        if key not in fb_mat:
            raise KeyError(f"Key '{key}' not found in fallback SC mat: {self.fallback_sc_path}")
        self.fallback_sc = self._normalize_sc(fb_mat[key])

        # ---- read label table ----
        df = pd.read_csv(self.label_csv)
        if self.patno_col not in df.columns or self.label_col not in df.columns:
            raise KeyError(f"CSV must contain columns: {self.patno_col}, {self.label_col}")

        # Build map: "<PATNO as string>" -> label_id
        subj2label = {}
        dropped_unknown = 0
        dropped_missing = 0

        for _, row in df.iterrows():
            patno = row[self.patno_col]
            cohort = row[self.label_col]

            if pd.isna(patno) or pd.isna(cohort):
                dropped_missing += 1
                continue

            # PATNO normalize to string digits
            try:
                patno_str = str(int(patno))
            except Exception:
                patno_str = str(patno).strip()

            cohort_str = str(cohort).strip()
            cohort_key = cohort_str.upper()

            if cohort_key not in self.COHORT2ID:
                dropped_unknown += 1
                continue

            subj2label[patno_str] = self.COHORT2ID[cohort_key]

        if len(subj2label) == 0:
            raise RuntimeError("No valid subject labels parsed from label_csv.")

        # ---- scan BOLD files and match to subj2label ----
        pattern = os.path.join(self.data_dir, self.data_ext)
        all_paths = sorted(glob.glob(pattern))

        data_paths = []
        labels = []
        lengths = []
        unmatched = 0
        no_subj = 0

        for p in all_paths:
            name = os.path.basename(p)

            patno_str = self._extract_patno_from_filename(name)
            if patno_str is None:
                no_subj += 1
                continue

            if patno_str not in subj2label:
                unmatched += 1
                continue

            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T, N = x_np.shape
            lengths.append(T)

            data_paths.append(p)
            labels.append(subj2label[patno_str])

        if len(data_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files matched labels.\n"
                f"data_dir={self.data_dir}, pattern={pattern}\n"
                f"label_csv={self.label_csv}\n"
                f"Tip: check filename contains PATNO (e.g., sub-1234) and CSV PATNO matches."
            )

        self.data_paths = data_paths
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.num_classes = len(self.COHORTS)

        # determine N from first sample (cleaned)
        x0 = self._load_bold_array(self.data_paths[0])
        if x0 is None:
            raise RuntimeError("First valid BOLD sample failed to load.")
        self.N = int(x0.shape[1])
        self.T = self.T_fix

        # SC cache
        self.sc_cache = [None] * len(self.data_paths) if self.cache_sc else None

        # summary
        if self.verbose:
            cnt = {c: 0 for c in self.COHORTS}
            for lab in labels:
                cnt[self.COHORTS[int(lab)]] += 1
            lengths_np = np.asarray(lengths, dtype=np.int32) if len(lengths) > 0 else None

            print(f"[PPMI] Dataset size: {len(self.data_paths)} samples")
            print(f"[PPMI] Class counts: {cnt}")
            print(f"[PPMI] Unique labels: {sorted(list(set(labels)))}")
            print(f"[PPMI] Files without PATNO parsed: {no_subj}")
            print(f"[PPMI] Unmatched BOLD files (no label): {unmatched}")
            print(f"[PPMI] Dropped CSV rows (unknown cohort): {dropped_unknown}")
            print(f"[PPMI] Dropped CSV rows (missing): {dropped_missing}")
            if lengths_np is not None and lengths_np.size > 0:
                print(
                    f"[PPMI] Cleaned length stats (before padding/cropping): "
                    f"min={lengths_np.min()}  p25={np.percentile(lengths_np,25):.0f}  "
                    f"median={np.percentile(lengths_np,50):.0f}  p75={np.percentile(lengths_np,75):.0f}  "
                    f"max={lengths_np.max()}"
                )
            print(f"[PPMI] Output fixed length T_fix={self.T_fix}")

    def __len__(self):
        return len(self.data_paths)

    # ---------- PATNO extract from filename ----------
    @staticmethod
    def _extract_patno_from_filename(filename: str):
        """
        Try robust patterns to find PATNO as digits.
        Examples supported:
          sub-1234_...
          sub-0001234_...
          patno-1234_...
          ..._1234_...
        Returns: "1234" (leading zeros removed) or None
        """
        # 1) sub-<digits>
        m = re.search(r"sub-0*([0-9]+)", filename, flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1)))

        # 2) patno-<digits>
        m = re.search(r"patno-0*([0-9]+)", filename, flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1)))

        # 3) fallback: first long-ish digit token (>=3 digits)
        m = re.search(r"(^|[^0-9])0*([0-9]{3,})([^0-9]|$)", filename)
        if m:
            return str(int(m.group(2)))

        return None

    # ---------- load + clean BOLD ----------
    @staticmethod
    def _load_bold_array(path: str):
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None

        if raw.ndim < 2:
            return None

        data = raw[1:, 1:]  # remove first row/col

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        data = np.vectorize(to_float)(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # remove all-zero rows
        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0 or data.ndim != 2:
            return None

        return data

    # ---------- row-normalize SC ----------
    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    # ---------- SC load ----------
    def _load_sc_for_subject(self, patno_str: str) -> torch.Tensor:

        if patno_str is None or self.sc_root_dir is None:
            return self.fallback_sc.clone()

        root = self.sc_root_dir
        if not os.path.isdir(root):
            return self.fallback_sc.clone()

        # loose match: any mat containing PATNO
        mat_paths = glob.glob(os.path.join(root, f"*{patno_str}*.mat"))
        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        sc_data = scipy.io.loadmat(mat_paths[0])
        key = "aal116_sift_radius2_count_connectivity"
        if key not in sc_data:
            return self.fallback_sc.clone()

        return self._normalize_sc(sc_data[key])

    # ---------- fix length ----------
    def _fix_length_to_T(self, x_np: np.ndarray) -> np.ndarray:
        T, N = x_np.shape
        if T == self.T_fix:
            return x_np
        if T > self.T_fix:
            start = (T - self.T_fix) // 2
            end = start + self.T_fix
            return x_np[start:end, :]
        pad_len = self.T_fix - T
        pad = np.zeros((pad_len, N), dtype=x_np.dtype)
        return np.concatenate([x_np, pad], axis=0)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        name = os.path.basename(data_path)

        x_np = self._load_bold_array(data_path)
        if x_np is None:
            x_np = np.zeros((self.T_fix, self.N), dtype=np.float32)
        x_np = self._fix_length_to_T(x_np)
        x = torch.from_numpy(x_np.astype(np.float32))

        label = self.labels[idx].clone()

        patno_str = self._extract_patno_from_filename(name)

        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            sc = self._load_sc_for_subject(patno_str)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        return x, label, sc.clone()


import os, re, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io
import pandas as pd


class NIFD(Dataset):
    """
    NIFD BOLD dataset (subject-level classification) with labels from XLSX.

    Label mapping:
      DX in {PNFA, BV, CON, SV} -> 4 classes
      others/unknown -> OTHER

    Subject matching:
      - XLSX column LONI_ID: e.g. "1_S_0354" -> "1S0354"
      - BOLD filename contains: "sub-1S0354" (case-insensitive supported)

    Output:
      x     : [T_fix, N] float32
      label : scalar int64
      sc    : [N_sc, N_sc] float32 row-normalized
    """

    DX_MAIN = ["PNFA", "BV", "CON", "SV"]
    DX2ID = {dx: i for i, dx in enumerate(DX_MAIN)}
    OTHER_ID = len(DX_MAIN)  # 4

    def __init__(
        self,
        data_dir,
        label_xlsx,
        sc_root_dir=None,
        fallback_sc_path=None,
        data_ext="*.csv",
        T_fix=220,
        cache_sc=True,
        verbose=True,
        xlsx_sheet_name=None,   # optional
        xlsx_subject_col="LONI_ID",
        xlsx_label_col="DX",
        sc_key="aal116_sift_radius2_count_connectivity",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.label_xlsx = label_xlsx
        self.sc_root_dir = sc_root_dir
        self.fallback_sc_path = fallback_sc_path
        self.data_ext = data_ext
        self.T_fix = int(T_fix)
        self.cache_sc = bool(cache_sc)
        self.verbose = bool(verbose)
        self.sc_key = sc_key

        if self.fallback_sc_path is None:
            raise ValueError("fallback_sc_path must be provided.")

        # ---- preload fallback SC ----
        fb_mat = scipy.io.loadmat(self.fallback_sc_path)
        if self.sc_key not in fb_mat:
            raise KeyError(f"Key '{self.sc_key}' not found in fallback SC mat: {self.fallback_sc_path}")
        self.fallback_sc = self._normalize_sc(fb_mat[self.sc_key])  # torch.FloatTensor

        # ---- read XLSX ----
        # ---- read xlsx robustly: always get a DataFrame ----
        df = pd.read_excel(self.label_xlsx, sheet_name=xlsx_sheet_name, engine="openpyxl")

        # If sheet_name=None or list -> pandas returns dict(sheet->DataFrame)
        if isinstance(df, dict):
            if xlsx_sheet_name is None:
                # take the first sheet by default
                first_sheet = next(iter(df.keys()))
                print(f"[NIFD] sheet_name=None -> using first sheet: {first_sheet}")
                df = df[first_sheet]
            else:
                # if user passed a list, pick the first requested sheet that exists
                if isinstance(xlsx_sheet_name, (list, tuple)):
                    picked = None
                    for s in xlsx_sheet_name:
                        if s in df:
                            picked = s
                            break
                    if picked is None:
                        picked = next(iter(df.keys()))
                        print(f"[NIFD] requested sheets not found -> using first sheet: {picked}")
                    else:
                        print(f"[NIFD] using sheet: {picked}")
                    df = df[picked]
                else:
                    # string but still dict (rare) -> fallback to first
                    picked = xlsx_sheet_name if xlsx_sheet_name in df else next(iter(df.keys()))
                    print(f"[NIFD] using sheet: {picked}")
                    df = df[picked]

        # final sanity
        if not hasattr(df, "columns"):
            raise TypeError(f"[NIFD] read_excel did not return a DataFrame. Got type={type(df)}")

        if xlsx_subject_col not in df.columns:
            raise KeyError(f"XLSX missing subject column '{xlsx_subject_col}'. Columns={list(df.columns)}")
        if xlsx_label_col not in df.columns:
            raise KeyError(f"XLSX missing label column '{xlsx_label_col}'. Columns={list(df.columns)}")

        # build subject->label dict
        subj2label = {}
        for _, row in df.iterrows():
            lon = row[xlsx_subject_col]
            dx = row[xlsx_label_col]

            if pd.isna(lon):
                continue
            lon = str(lon).strip()
            # "1_S_0354" -> "1S0354"
            lon_norm = re.sub(r"_", "", lon)

            dx_str = "" if pd.isna(dx) else str(dx).strip().upper()
            if dx_str in self.DX2ID:
                y = self.DX2ID[dx_str]
            else:
                y = self.OTHER_ID

            subj2label[lon_norm.upper()] = int(y)

        # ---- scan BOLD files ----
        pattern = os.path.join(self.data_dir, self.data_ext)
        all_paths = sorted(glob.glob(pattern))

        data_paths = []
        labels = []
        lengths = []
        n_skipped_no_sub = 0
        n_skipped_no_label = 0

        for p in all_paths:
            name = os.path.basename(p)
            subj = self._extract_subj_id(name)  # returns like "1S0354" (no "sub-")
            if subj is None:
                n_skipped_no_sub += 1
                continue

            key = subj.upper()
            if key not in subj2label:
                n_skipped_no_label += 1
                continue
            y = subj2label[key]

            x_np = self._load_bold_array(p)
            if x_np is None:
                continue

            T, N = x_np.shape
            lengths.append(T)

            # keep all, we will crop/pad in __getitem__
            data_paths.append(p)
            labels.append(y)

        if len(data_paths) == 0:
            raise RuntimeError(
                f"No valid BOLD files found in {data_dir} with pattern {pattern}. "
                f"Matched subjects in xlsx={len(subj2label)}. "
                f"Skipped(no sub)= {n_skipped_no_sub}, skipped(no label)= {n_skipped_no_label}."
            )

        self.data_paths = data_paths
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.num_classes = len(self.DX_MAIN) + 1  # +OTHER

        # ---- determine N from first sample ----
        x0 = self._load_bold_array(self.data_paths[0])
        if x0 is None:
            raise RuntimeError("First valid BOLD sample failed to load.")
        self.N = int(x0.shape[1])
        self.T = self.T_fix

        # ---- SC cache ----
        self.sc_cache = [None] * len(self.data_paths) if self.cache_sc else None

        # ---- summary ----
        if self.verbose:
            id2name = {i: n for n, i in self.DX2ID.items()}
            id2name[self.OTHER_ID] = "OTHER"
            uniq, cnt = np.unique(np.asarray(labels), return_counts=True)
            counts = {id2name[int(k)]: int(v) for k, v in zip(uniq, cnt)}
            lengths_np = np.asarray(lengths, dtype=np.int32) if len(lengths) else None
            print(f"[NIFD] Dataset size: {len(self.data_paths)} samples")
            print(f"[NIFD] Class counts: {counts}")
            if lengths_np is not None and lengths_np.size > 0:
                print(
                    f"[NIFD] Cleaned length stats (before fixing): "
                    f"min={lengths_np.min()}  p25={np.percentile(lengths_np,25):.0f}  "
                    f"median={np.percentile(lengths_np,50):.0f}  p75={np.percentile(lengths_np,75):.0f}  "
                    f"max={lengths_np.max()}"
                )
            print(f"[NIFD] Output fixed length T_fix={self.T_fix}")

    def __len__(self):
        return len(self.data_paths)

    # ---------- parse subject id from filename ----------
    @staticmethod
    def _extract_subj_id(filename: str):
        """
        Find subject id in filename:
          sub-1S0354 -> returns "1S0354"
        """
        m = re.search(r"sub-([A-Za-z0-9]+)", filename, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    # ---------- load + clean BOLD CSV ----------
    @staticmethod
    def _load_bold_array(path: str):
        try:
            raw = np.genfromtxt(path, delimiter=",", dtype=str)
        except Exception as e:
            print(f"[ERR] genfromtxt failed on {path}: {e}")
            return None
        if raw.ndim < 2:
            return None

        data = raw[1:, 1:]  # drop header row/col

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        data = np.vectorize(to_float)(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # remove all-zero rows
        if data.shape[0] > 1:
            valid = ~(np.all(data == 0, axis=1))
            data = data[valid]

        if data.size == 0 or data.ndim != 2:
            return None
        return data

    # ---------- row-normalize SC ----------
    @staticmethod
    def _normalize_sc(sc_np: np.ndarray) -> torch.Tensor:
        sc = np.array(sc_np, dtype=np.float32)
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sc = sc / row_sums
        return torch.from_numpy(sc).float()

    # ---------- load SC (simple: directly under sc_root_dir) ----------
    def _load_sc_for_subject(self, subj_id: str) -> torch.Tensor:
        if subj_id is None or self.sc_root_dir is None:
            return self.fallback_sc.clone()

        root = self.sc_root_dir
        if not os.path.isdir(root):
            return self.fallback_sc.clone()

        # match any mat containing subject id (subj_id already like "1S0354")
        mat_paths = glob.glob(os.path.join(root, f"*{subj_id}*.mat"))
        if len(mat_paths) == 0:
            return self.fallback_sc.clone()

        try:
            sc_data = scipy.io.loadmat(mat_paths[0])
            if self.sc_key in sc_data:
                return self._normalize_sc(sc_data[self.sc_key])
        except Exception:
            pass

        return self.fallback_sc.clone()

    # ---------- fix length to T_fix ----------
    def _fix_length_to_T(self, x_np: np.ndarray) -> np.ndarray:
        T, N = x_np.shape
        if T == self.T_fix:
            return x_np
        if T > self.T_fix:
            start = (T - self.T_fix) // 2
            end = start + self.T_fix
            return x_np[start:end, :]
        # pad
        pad_len = self.T_fix - T
        pad = np.zeros((pad_len, N), dtype=x_np.dtype)
        return np.concatenate([x_np, pad], axis=0)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        name = os.path.basename(data_path)
        subj = self._extract_subj_id(name)

        x_np = self._load_bold_array(data_path)
        if x_np is None:
            x_np = np.zeros((self.T_fix, self.N), dtype=np.float32)
        else:
            x_np = self._fix_length_to_T(x_np)

        x = torch.from_numpy(x_np.astype(np.float32))  # [T_fix, N]
        label = self.labels[idx].clone()

        if self.sc_cache is not None and self.sc_cache[idx] is not None:
            sc = self.sc_cache[idx]
        else:
            sc = self._load_sc_for_subject(subj)
            if self.sc_cache is not None:
                self.sc_cache[idx] = sc

        return x, label, sc.clone()
