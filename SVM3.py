import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 定義類別
classes = ['car', 'dog', 'chicken']

# 定義特徵提取函數
def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法加載圖像: {img_path}")
            return None
        # 轉換為 HSV 顏色空間
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 縮放圖像到固定大小 (64x64)
        img_resized = cv2.resize(img_hsv, (64, 64))
        # 將圖像展平為一維特徵向量
        feature = img_resized.flatten()
        return feature
    except Exception as e:
        print(f"錯誤發生在 {img_path}: {str(e)}")
        return None

# 准備數據集
def prepare_dataset(dataset_root):
    features = []
    labels = []
    failed_images = []

    for cls in classes:
        cls_path = os.path.join(dataset_root, cls)
        if not os.path.exists(cls_path):
            print(f"類別文件夾不存在: {cls_path}")
            continue
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            feature = extract_features(img_path)
            if feature is None:
                failed_images.append(img_path)
            else:
                features.append(feature)
                labels.append(cls)
    
    print(f"成功加載的圖像數量: {len(features)}")
    if failed_images:
        print("無法加載的圖像:")
        for img in failed_images:
            print(img)
    
    return np.array(features), np.array(labels)

# 訓練模型
def train_model(features, labels):
    # 分割數據集為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 創建 SVM 分類器
    clf = svm.SVC(kernel='rbf', gamma='scale')
    clf.fit(X_train, y_train)
    
    # 在測試集上評估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型準確率: {accuracy}")
    
    return clf

# 保存模型
def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {model_path}")

# 加載模型
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# 預測圖像類別
def predict_image(model, img_path):
    feature = extract_features(img_path)
    if feature is None:
        return None, None
    prediction = model.predict([feature])
    confidence = model.decision_function([feature])
    return prediction[0], confidence[0]

# 主程序
if __name__ == "__main__":
    # 設定數據集根目錄
    dataset_root = 'D:\AI\dataset'
    
    # 準備數據集
    features, labels = prepare_dataset(dataset_root)
    
    if len(features) == 0:
        print("沒有有效的特徵數據，無法訓練模型。")
        exit()
    
    # 訓練模型
    model = train_model(features, labels)
    
    # 保存模型
    model_path = 'model.pkl'
    save_model(model, model_path)
    
    # 加載模型
    loaded_model = load_model(model_path)
    
    # 測試模型
    test_image_path = 'test_image.jpg'
    if not os.path.exists(test_image_path):
        print(f"測試圖像不存在: {test_image_path}")
        exit()
    
    prediction, confidence = predict_image(loaded_model, test_image_path)
    if prediction is not None:
        print(f"預測結果: {prediction}")
        print(f"置信度: {confidence}")


