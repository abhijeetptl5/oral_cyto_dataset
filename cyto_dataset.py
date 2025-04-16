import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path

class CytologyDataset(Dataset):
    def __init__(self, geojson_dir, transform=None, mode='train', split_percent=0.8):
        self.geojson_dir = Path(geojson_dir)
        self.geojson_files = list(self.geojson_dir.rglob('*.geojson'))
        self.geojson_files = sorted([str(s) for s in list(self.geojson_dir.rglob('*.geojson'))])
        split_num = int(split_percent*len(self.geojson_files))
        if mode=='train': self.geojson_files = self.geojson_files[:split_num]
        if mode=='val': self.geojson_files = self.geojson_files[split_num:]
        self.transform = transform

    def __len__(self):
        return len(self.geojson_files)

    def _geojson_to_mask(self, geojson_path, image_shape):
        gdf = gpd.read_file(geojson_path)
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for geom in gdf.geometry:
            if geom is not None:
                coords = np.array(mapping(geom)['coordinates'][0], dtype=np.int32)
                cv2.fillPoly(mask, [coords], 1)
        return mask

    def __getitem__(self, idx):
        geojson_path = self.geojson_files[idx]
        image_path = geojson_path[:-8] + '.png'
        image_path = image_path.replace('geojsons', 'images')
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self._geojson_to_mask(geojson_path, image.shape)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.long()