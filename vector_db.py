import faiss
import numpy as np
from deepface import DeepFace
import os
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




# 提取特征向量的函数
def process_image(args):
    img_path, model_name, detector_backend = args
    try:
        # 重定向标准输出到/dev/null
        sys.stdout = open(os.devnull, 'w')
        embedding = DeepFace.represent(img_path, model_name=model_name, enforce_detection=False, detector_backend=detector_backend)[0]["embedding"]
        return img_path, embedding
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
    # finally:
    # 恢复标准输出
        sys.stdout = sys.__stdout__

# 设置参数
db_path = "../VGG-Face2/data/test"
model_name = "VGG-Face"
detector_backend = "mtcnn"
num_processes = 2
# 收集所有图片路径
image_paths = []
for person in os.listdir(db_path):
    person_dir = os.path.join(db_path, person)
    if os.path.isdir(person_dir):
        count = 0
        for img_name in os.listdir(person_dir):
            if count >= 20:  # 处理每个人的20张照片
                break
            img_path = os.path.join(person_dir, img_name)
            image_paths.append((img_path, model_name, detector_backend))
            count += 1

# 使用多进程处理图片并显示进度条
with Pool(num_processes) as pool:
    results = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths)))

# 过滤掉处理失败的图片
results = [result for result in results if result is not None]

# 分离路径和特征向量
paths, embeddings = zip(*results)

# 转换为 numpy 数组并确保数据类型为 float32
embeddings = np.array(embeddings).astype('float32')

# 创建Faiss索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 将路径和索引保存
with open("paths1.pkl", "wb") as f:
    pickle.dump(paths, f)
faiss.write_index(index, "faiss_index1.bin")

print("Indexing completed successfully.")
