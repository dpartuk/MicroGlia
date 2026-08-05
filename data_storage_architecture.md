# High-Performance Data Storage Architecture
## Storing 1,000,000+ Microglial Crops, Silhouettes, Embeddings & Labels

---

## Executive Storage Architecture

### **Database Strategy: MongoDB + Image Shards Grouped by Original Image ID**

1. **Metadata & Active Labeling Database (MongoDB / Parquet)**:
   - JSON-like document structure in **MongoDB** (or Apache Parquet index) for real-time active labeling queries, cluster assignments, uncertainty scores, and filter queries (e.g. *"find all high-uncertainty cells in Image X"*).
2. **Binary Image Containers Sharded by Original Image ID (`.h5` per Tissue Slide)**:
   - Rather than arbitrary chunk sizes, store extracted cell crops, masks, and embeddings **sharded directly by their original whole-slide Image ID** (e.g., `VID2724_A3_4_00d07h00m_cells.h5`).
   - Keeps all single-cell crops extracted from a single tissue slide tightly coupled together!

---

## Why We MUST Store Bounding Box (`bbox`) & Spatial Coordinates

Even though single-cell crops are already extracted into standalone images, **storing `bbox = [x, y, w, h]` and centroid `(x_c, y_c)` is CRITICAL for 4 reasons**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SPATIAL GRAPH RECONSTRUCTION (Theme 2: G = (V, E))                       │
│    To build k-NN / Delaunay spatial graphs connecting somas and process     │
│    fragments within 35 µm, the GNN MUST know each cell's physical position  │
│    on the whole-slide tissue section. Without bbox, spatial context is lost!│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. WHOLE-SLIDE OVERLAY & VISUAL QC MAPS                                     │
│    After inference, researchers (Dr. Lilach Gavish's team) inspect the     │
│    original slide with predicted cell states highlighted (Resting=Green,   │
│    Activated=Red, Resolution=Yellow). Bbox projects predictions back to 2D.│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. RECONNECTING SHATTERED DYSTROPHIC PROCESS FRAGMENTS                      │
│    Dystrophic cells break into beaded fragments. Bbox coordinates allow     │
│    the GNN to measure physical distance and reconnect fragments to somas.   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. CROSS-TILE SEAM DEDUPLICATION (NMS)                                       │
│    During 1024x1024 tile inference, overlapping tile seams crop the same    │
│    cell twice. Bbox enables Non-Maximum Suppression (NMS) deduplication.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Database & Sharding Schema

### 1. MongoDB Document Schema (`microglia_metadata_collection`)
MongoDB stores fast-queried document metadata:

```json
{
  "_id": "VID2724_A3_4_cell_00421",
  "source_image_id": "VID2724_A3_4_00d07h00m.tif",
  "shard_filename": "VID2724_A3_4_00d07h00m_cells.h5",
  "shard_index": 421,
  "spatial_coordinates": {
    "bbox": [650, 1050, 64, 64],
    "centroid_x": 682.0,
    "centroid_y": 1082.0
  },
  "morphometrics": {
    "soma_area_px": 412.5,
    "circularity": 0.82,
    "fractal_dimension": 1.45
  },
  "cluster_id": 14,
  "active_learning": {
    "uncertainty_score": 0.12,
    "is_flagged_for_review": false
  },
  "label": {
    "class_id": 0,
    "class_name": "Resting",
    "annotated_by": "expert_1",
    "annotated_at": "2026-08-05T12:00:00Z"
  }
}
```

### 2. Binary Image Container Sharded by Original Image ID (`VID2724_A3_4_00d07h00m_cells.h5`)
Inside each per-image container file:

```python
with h5py.File("VID2724_A3_4_00d07h00m_cells.h5", "w") as f:
    # N = total cells extracted from this specific slide (e.g. N = 287)
    f.create_dataset("images", shape=(N, 128, 128, 3), dtype="uint8", compression="gzip")
    f.create_dataset("masks", shape=(N, 128, 128), dtype="uint8", compression="gzip")
    f.create_dataset("embeddings", shape=(N, 768), dtype="float32", compression="gzip")
    f.create_dataset("labels", shape=(N,), dtype="int32")
    f.create_dataset("bboxes", shape=(N, 4), dtype="int32")  # [x, y, w, h]
```

---

## PyTorch DataLoader Implementation with MongoDB + Image Shards

Fetching training mini-batches is ultra-fast:

```python
import h5py, torch
from pymongo import MongoClient

class MicrogliaLabDataset(torch.utils.data.Dataset):
    def __init__(self, mongo_uri, query_filter):
        self.db = MongoClient(mongo_uri)['microglia_db']
        # Query MongoDB for metadata records matching filter (e.g. labeled cells)
        self.records = list(self.db['cells'].find(query_filter))
        self.open_shards = {} # Cache open HDF5 file handles

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        shard_path = rec['shard_filename']
        shard_idx = rec['shard_index']

        if shard_path not in self.open_shards:
            self.open_shards[shard_path] = h5py.File(shard_path, 'r')

        h5_file = self.open_shards[shard_path]
        img = torch.from_numpy(h5_file['images'][shard_idx])      # uint8 [128, 128, 3]
        mask = torch.from_numpy(h5_file['masks'][shard_idx])     # uint8 [128, 128]
        emb = torch.from_numpy(h5_file['embeddings'][shard_idx])  # float32 [768]
        lbl = rec['label']['class_id']
        bbox = torch.tensor(rec['spatial_coordinates']['bbox'])   # [x, y, w, h]

        return img, mask, emb, lbl, bbox
```

---

## Summary

1. **MongoDB is Excellent for Metadata**: Keeps labels, cluster IDs, uncertainty scores, and spatial BBox coordinates searchable in JSON documents.
2. **Sharding Images by Original Image ID**: Storing cell crops inside `IMAGE_ID_cells.h5` keeps tissue-level spatial context intact.
3. **BBox is Mandatory**: Required to construct Spatial Graphs ($G=(V,E)$), map whole-slide classification overlays, reconnect shattered dystrophic fragments, and deduplicate tile seams.
