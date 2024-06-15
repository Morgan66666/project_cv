import faiss
import numpy as np
from deepface import DeepFace
import os
import random
import pickle

# 加载Faiss索引
index = faiss.read_index("faiss_index1.bin")

# 加载路径
with open("paths1.pkl", "rb") as f:
    paths = pickle.load(f)

# 设置数据库路径
db_path_test = "../VGG-Face2/data/test"
db_path_train = "../VGG-Face2/data/train"

# 获取所有正例图像路径
all_image_paths_test = []
for person in os.listdir(db_path_test):
    person_dir = os.path.join(db_path_test, person)
    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            all_image_paths_test.append(img_path)

# 获取所有负例图像路径
all_image_paths_train = []
for person in os.listdir(db_path_train):
    person_dir = os.path.join(db_path_train, person)
    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            all_image_paths_train.append(img_path)


unique_train_images = list(set(all_image_paths_train) - set(all_image_paths_test))

# 随机选择一些正例进行评估
num_queries_test = 50  # 可以根据需要调整查询图像的数量
random.seed(3407)  # 设置随机种子以便结果可重复
query_image_paths_test = random.sample(all_image_paths_test, num_queries_test)

# 随机选择一些负例进行评估
num_queries_train = 50  # 可以根据需要调整查询图像的数量
query_image_paths_train = random.sample(unique_train_images, num_queries_train)

# 合并查询图像路径
query_image_paths = query_image_paths_test + query_image_paths_train

# 置信系数阈值
confidence_threshold = 0.98  # 这个值可以根据需要调整，就是欧氏距离的值

# 进行评估
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
total_queries = len(query_image_paths)

for query_img_path in query_image_paths:
    try:
        # 提取查询图像的特征向量
        query_embedding = DeepFace.represent(query_img_path, model_name="VGG-Face", enforce_detection=False, detector_backend="mtcnn")[0]["embedding"]
        query_embedding = np.array(query_embedding).astype('float32')

        # 查询Faiss索引
        D, I = index.search(np.array([query_embedding]), k=1) 

        # 获取查询图像的人的ID
        query_person_id = os.path.basename(os.path.dirname(query_img_path))
        
        avg_distance = np.mean(D[0])
        
        match_found = False
        for idx in I[0]:
            predicted_img_path = paths[idx]
            predicted_person_id = os.path.basename(os.path.dirname(predicted_img_path))
            if query_person_id == predicted_person_id:
                match_found = True
                break

        # 判断匹配是否正确
        if query_img_path in query_image_paths_test:
            if avg_distance < confidence_threshold and match_found:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if avg_distance >= confidence_threshold or not match_found:
                true_negatives += 1
            else:
                false_positives += 1

        # 打印结果
        print(f"Query image: {query_img_path}")
        print(f"Top 5 nearest neighbors: {[paths[idx] for idx in I[0]]}")
        print(f"Average distance: {avg_distance}")
        print(f"Match found: {match_found}")

    except Exception as e:
        print(f"Error processing {query_img_path}: {e}")
        false_positives += 1

# 计算准确性
accuracy = (true_positives + true_negatives) / total_queries

print(f"Accuracy: {accuracy * 100:.2f}%")
