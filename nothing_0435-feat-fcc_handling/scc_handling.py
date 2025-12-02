import numpy as np
from PIL import Image
from pathlib import Path

class IR2RGBModel:
    def __init__(self, npz_path):
        npz = np.load(npz_path, allow_pickle=False)
        # load rpcc weights for R and G (RPCC), and LUT for B
        self.rpcc = {
            "R": npz.get("rpcc_R").astype(np.float32) if "rpcc_R" in npz else None,
            "G": npz.get("rpcc_G").astype(np.float32) if "rpcc_G" in npz else None
        }
        self.lut = {
            "B": npz.get("lut_B").astype(np.float32) if "lut_B" in npz else None
        }
        # infer LUT grid size if present
        self.grid_size = None
        if self.lut["B"] is not None:
            self.grid_size = self.lut["B"].shape[0]

    # ---------------- utilities ----------------
    @staticmethod
    def _load_img(img):
        if isinstance(img, (str, Path)):
            im = Image.open(str(img)).convert("RGB")
            arr = np.array(im).astype(np.float32) / 255.0
        else:
            arr = np.array(img).astype(np.float32)
            if arr.max() > 2.0:  # probably 0-255
                arr = arr / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Input must be 3-channel image (H,W,3).")
        return arr

    @staticmethod
    def _map_channels(arr, channels):
        # channels: list-like length 3, order corresponds to arr[...,0], arr[...,1], arr[...,2]
        if len(channels) != 3:
            raise ValueError("channels must be length 3 (one label per image channel).")
        ch_map = {}
        for i, name in enumerate(channels):
            ch_map[name.strip().upper()] = arr[..., i]
        return ch_map

    @staticmethod
    def _features_root_poly2(X):
        # X shape (N,3) -> returns Psi (N,10)
        x1 = X[:, 0]; x2 = X[:, 1]; x3 = X[:, 2]
        eps = 1e-8
        m4 = x1 * x2; m5 = x1 * x3; m6 = x2 * x3
        m7 = x1 ** 2; m8 = x2 ** 2; m9 = x3 ** 2
        return np.stack([
            x1, x2, x3,
            np.sqrt(np.abs(m4) + eps),
            np.sqrt(np.abs(m5) + eps),
            np.sqrt(np.abs(m6) + eps),
            np.sqrt(np.abs(m7) + eps),
            np.sqrt(np.abs(m8) + eps),
            np.sqrt(np.abs(m9) + eps),
            np.ones_like(x1)
        ], axis=1).astype(np.float32)

    @staticmethod
    def _apply_lut_scalar(X, lut):
        # trilinear interpolation of scalar LUT
        gs = lut.shape[0]
        scale = gs - 1
        coords = np.clip(X * scale, 0, scale - 1e-6)
        x = coords[:, 0]; y = coords[:, 1]; z = coords[:, 2]
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); z0 = np.floor(z).astype(int)
        x1 = np.clip(x0 + 1, 0, scale); y1 = np.clip(y0 + 1, 0, scale); z1 = np.clip(z0 + 1, 0, scale)
        dx = (x - x0).astype(np.float32); dy = (y - y0).astype(np.float32); dz = (z - z0).astype(np.float32)

        c000 = lut[x0, y0, z0]; c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]; c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]; c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]; c111 = lut[x1, y1, z1]

        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w010 = (1 - dx) * dy       * (1 - dz)
        w011 = (1 - dx) * dy       * dz
        w100 = dx       * (1 - dy) * (1 - dz)
        w101 = dx       * (1 - dy) * dz
        w110 = dx       * dy       * (1 - dz)
        w111 = dx       * dy       * dz

        out = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011 +
               w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)
        return out.astype(np.float32)

    @staticmethod
    def _to_pil(rgb_arr):
        rgb = np.clip(rgb_arr, 0.0, 1.0)
        return Image.fromarray((rgb * 255.0).astype(np.uint8))

    # ---------------- synthesis routines ----------------
    def synthesize_R(self, img, channels):
        arr = self._load_img(img)
        ch = self._map_channels(arr, channels)
        # need NIR, G, B
        if not all(k in ch for k in ("NIR", "G", "B")):
            raise ValueError("synthesize_R requires channels to include NIR, G, B in some order.")
        nir = ch["NIR"]; G = ch["G"]; B = ch["B"]
        H, W = nir.shape
        X_flat = np.stack([nir.reshape(-1), G.reshape(-1), B.reshape(-1)], axis=1).astype(np.float32)
        w = self.rpcc.get("R")
        if w is None:
            raise RuntimeError("RPCC weights for R not loaded.")
        Psi = self._features_root_poly2(X_flat)
        y_hat = (Psi @ w).astype(np.float32)
        R_hat = y_hat.reshape(H, W)
        # merge into RGB: [R_hat, G, B] — assume G and B are already in correct scale (0..1)
        rgb_out = np.stack([R_hat, G, B], axis=-1)
        return self._to_pil(rgb_out)

    def synthesize_G(self, img, channels):
        arr = self._load_img(img)
        ch = self._map_channels(arr, channels)
        # need NIR, R, B
        if not all(k in ch for k in ("NIR", "R", "B")):
            raise ValueError("synthesize_G requires channels to include NIR, R, B in some order.")
        nir = ch["NIR"]; R = ch["R"]; B = ch["B"]
        H, W = nir.shape
        X_flat = np.stack([nir.reshape(-1), R.reshape(-1), B.reshape(-1)], axis=1).astype(np.float32)
        w = self.rpcc.get("G")
        if w is None:
            raise RuntimeError("RPCC weights for G not loaded.")
        Psi = self._features_root_poly2(X_flat)
        y_hat = (Psi @ w).astype(np.float32)
        G_hat = y_hat.reshape(H, W)
        rgb_out = np.stack([R, G_hat, B], axis=-1)
        return self._to_pil(rgb_out)

    def synthesize_B(self, img, channels):
        arr = self._load_img(img)
        ch = self._map_channels(arr, channels)
        # need NIR, R, G
        if not all(k in ch for k in ("NIR", "R", "G")):
            raise ValueError("synthesize_B requires channels to include NIR, R, G in some order.")
        nir = ch["NIR"]; R = ch["R"]; G = ch["G"]
        H, W = nir.shape
        X_flat = np.stack([nir.reshape(-1), R.reshape(-1), G.reshape(-1)], axis=1).astype(np.float32)
        lut = self.lut.get("B")
        if lut is None:
            raise RuntimeError("LUT for B not loaded.")
        y_hat = self._apply_lut_scalar(X_flat, lut)
        B_hat = y_hat.reshape(H, W)
        rgb_out = np.stack([R, G, B_hat], axis=-1)
        return self._to_pil(rgb_out)
    
    
if __name__ == "__main__":
    model = IR2RGBModel("model_weights/models_ir_rgb.npz")

    img_path = "assets/scc_img3.jpg"

    # example channel order (the channels present in the input FCC)
    channels = ["NIR", "R", "G"]   # change as needed

    # choose what to synthesize
    out = model.synthesize_B(img_path, channels)

    out.save("assets/output_rgb_scc3.png")
    print("Saved assets/output_rgb_scc3.png")