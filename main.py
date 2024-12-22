import os
import numpy as np
from PIL import Image
import pywt
from tqdm import tqdm

# === 新增的 import ===
from scipy.fftpack import dct, idct
from sklearn.cluster import MiniBatchKMeans
import huffman

# =========================
# 1. 建立演算法資料夾
# =========================
def create_folders(base_dir):
    """
    在 base_dir 下建立所有演算法的子資料夾
    """
    algorithms = ["Fast_DCT", "Vector_Quantization", "Huffman_Coding", "DWT"]
    for algo in algorithms:
        algo_dir = os.path.join(base_dir, algo)
        os.makedirs(algo_dir, exist_ok=True)

# =========================
# 2. 讀取圖片
# =========================
def read_images(folder):
    """
    從指定資料夾內讀取 .png 檔，轉為灰階並回傳 (檔名, 圖片物件) 的清單
    """
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith(".png"):
            img_path = os.path.join(folder, file)
            images.append((file, Image.open(img_path).convert("L")))
    return images

# =========================
# 3. 輸出圖片 (共用)
# =========================
def save_images(output_folder, algo, freq_domain_img, reconstructed_img, filtered_img):
    """
    將三張影像（或矩陣）以 PNG 檔案輸出：
    1. 頻域圖像 (freq_domain_img)
    2. 還原影像 (reconstructed_img)
    3. 濾波後還原影像 (filtered_img)
    """
    freq_domain_img = np.asarray(freq_domain_img)
    reconstructed_img = np.asarray(reconstructed_img)
    filtered_img = np.asarray(filtered_img)

    # freq_domain_img 可能值域很大(如 DCT 或 wavelet 係數)，需正規化到 [0,255]
    if freq_domain_img.dtype != np.uint8:
        denom = np.max(np.abs(freq_domain_img)) if np.max(np.abs(freq_domain_img)) != 0 else 1
        freq_domain_vis = (freq_domain_img / denom * 127.5) + 127.5
        freq_domain_vis = np.clip(freq_domain_vis, 0, 255).astype(np.uint8)
    else:
        freq_domain_vis = freq_domain_img

    # reconstructed_img 與 filtered_img 若在 [0,255] 外，需 clip 後轉 uint8
    recon_vis = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
    filtered_vis = np.clip(filtered_img, 0, 255).astype(np.uint8)

    # 寫檔
    Image.fromarray(freq_domain_vis).save(os.path.join(output_folder, f"{algo}_dct.png"))
    Image.fromarray(recon_vis).save(os.path.join(output_folder, f"{algo}_idct.png"))
    Image.fromarray(filtered_vis).save(os.path.join(output_folder, f"{algo}_filtered_idct.png"))

# =========================
# 4. Fast DCT (scipy)
# =========================
def fast_dct(image, output_folder):
    """
    1) 2D DCT (scipy.fftpack)
    2) 2D IDCT
    3) 頻域濾波：只保留左上 1/4 (示範)
    """
    img_array = np.array(image, dtype=np.float32)

    # 2D DCT: dct(dct(A, axis=0), axis=1)
    dct_2d = dct(dct(img_array, axis=0, norm='ortho'), axis=1, norm='ortho')

    # 2D IDCT
    idct_2d = idct(idct(dct_2d, axis=1, norm='ortho'), axis=0, norm='ortho')

    # 頻域濾波
    h, w = dct_2d.shape
    filtered_dct = np.copy(dct_2d)
    filtered_dct[h//4:, w//4:] = 0
    filtered_idct = idct(idct(filtered_dct, axis=1, norm='ortho'), axis=0, norm='ortho')

    # 輸出結果
    save_images(output_folder, "Fast_DCT", dct_2d, idct_2d, filtered_idct)

# =========================
# 5. Vector Quantization (sklearn)
# =========================
def vector_quantization(image, output_folder):
    """
    使用 sklearn 的 MiniBatchKMeans 做向量量化
    """
    # (H,W) -> (H*W, 1)
    img_array = np.array(image, dtype=np.float32).reshape(-1, 1)

    # 分成 n_clusters=16 群
    kmeans = MiniBatchKMeans(n_clusters=16, random_state=0)
    labels = kmeans.fit_predict(img_array)
    centers = kmeans.cluster_centers_  # shape=(16,1)

    # 解壓(反量化)：把每個像素的 label 換回對應 cluster center
    quantized = centers[labels].reshape(image.size[1], image.size[0])  # (H,W)

    # 簡單示範：把 < 60 的值全部歸 0 (視為濾波)
    filtered_q = np.copy(quantized)
    filtered_q[filtered_q < 60] = 0

    save_images(output_folder, "Vector_Quantization", quantized, quantized, filtered_q)

# =========================
# 6. Huffman Coding (huffman)
# =========================
def huffman_coding(image, output_folder):
    """
    使用 huffman 套件:
    1) 計算資料的 (symbol, freq)
    2) 生成 codebook
    3) 將原始資料轉 bitstring
    4) 解析 bitstring 還原原始資料
    """
    # 讀取影像並攤平成一維
    img_array = np.array(image, dtype=np.uint8)
    data = img_array.flatten()

    # 計算符號頻率
    freq_map = {}
    for val in data:
        freq_map[val] = freq_map.get(val, 0) + 1

    # 將 (symbol, freq) 集合做成 list
    pairs = [(symbol, freq) for symbol, freq in freq_map.items()]

    # 產生 codebook
    codebook = huffman.codebook(pairs)  # 回傳 {symbol: bitstring}

    # 壓縮：將每個像素值用對應的 bitstring 替換
    encoded_str = ''.join(codebook[val] for val in data)

    # 解壓：自己解析 bitstring
    rev_codebook = {v: k for k, v in codebook.items()}
    decoded_list = []
    current_bits = ""
    for bit in encoded_str:
        current_bits += bit
        if current_bits in rev_codebook:
            decoded_list.append(rev_codebook[current_bits])
            current_bits = ""

    # 轉回 2D
    decoded_arr = np.array(decoded_list, dtype=np.uint8).reshape(img_array.shape)

    # 濾波 => 這邊就示範直接複製還原結果
    filtered_arr = decoded_arr.copy()

    # 以 (原圖, 解壓圖, 濾波圖) 三張輸出
    save_images(output_folder, "Huffman_Coding", img_array, decoded_arr, filtered_arr)

# =========================
# 7. DWT (pywt)
# =========================
def discrete_wavelet_transform(image, output_folder):
    img_array = np.array(image, dtype=np.float32)
    coeffs2 = pywt.dwt2(img_array, 'haar')
    cA, (cH, cV, cD) = coeffs2

    # 逆變換
    reconstructed = pywt.idwt2(coeffs2, 'haar')

    # 濾波：將 cH, cV, cD 歸零
    filtered_coeffs = (cA, (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))
    filtered_reconstructed = pywt.idwt2(filtered_coeffs, 'haar')

    save_images(output_folder, "DWT", cA, reconstructed, filtered_reconstructed)

# =========================
# 8. 整合流程
# =========================
def process_images(input_folder, output_folder):
    create_folders(output_folder)
    images = read_images(input_folder)

    for file_name, image in tqdm(images, desc="Processing Images"):
        print(f"Processing {file_name}...")

        fast_dct(image, os.path.join(output_folder, "Fast_DCT"))
        vector_quantization(image, os.path.join(output_folder, "Vector_Quantization"))
        huffman_coding(image, os.path.join(output_folder, "Huffman_Coding"))
        discrete_wavelet_transform(image, os.path.join(output_folder, "DWT"))

# =========================
# 9. 主程式入口
# =========================
if __name__ == "__main__":
    # 請依需求修改資料夾
    input_folder = "images"
    output_folder = "output"
    process_images(input_folder, output_folder)
