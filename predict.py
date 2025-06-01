import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

# 預測配置（直接定義，不使用命令行參數）
CONFIG = {
    'model_path': './output/best_model_large.pth',  # 默認模型路徑
    'model_size': 'large',                          # 模型大小 ('small' 或 'large')
}

#---------------------------------------------------------------------------------------
# 模型定義
#---------------------------------------------------------------------------------------

class AgeGenderMobileNetV3(nn.Module):
    """基於MobileNetV3的年齡和性別預測模型"""
    
    def __init__(self, model_size='large', pretrained=True):
        """
        初始化模型
        
        參數:
            model_size (str): 'small' 或 'large' MobileNetV3版本
            pretrained (bool): 是否使用預訓練權重
        """
        super(AgeGenderMobileNetV3, self).__init__()
        
        # 載入基礎MobileNetV3模型
        if model_size == 'small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            last_channel = self.backbone.classifier[0].in_features
        else:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            last_channel = self.backbone.classifier[0].in_features
        
        # 移除原始分類器
        self.backbone.classifier = nn.Identity()
        
        # 年齡回歸器
        self.age_regressor = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # 性別分類器
        self.gender_classifier = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        """前向傳播"""
        # 特徵提取
        features = self.backbone(x)
        
        # 年齡預測 (回歸)
        age = self.age_regressor(features).squeeze(1)
        
        # 性別預測 (分類)
        gender = self.gender_classifier(features)
        
        return age, gender

def get_model(model_size='large', pretrained=True):
    """
    獲取MobileNetV3模型
    
    參數:
        model_size (str): 'small' 或 'large' MobileNetV3版本
        pretrained (bool): 是否使用預訓練權重
        
    返回:
        model (nn.Module): 模型實例
    """
    return AgeGenderMobileNetV3(model_size=model_size, pretrained=pretrained)

class AgeGenderPredictor:
    """年齡和性別預測器"""
    
    def __init__(self, model_path, model_size='large', device=None):
        """
        初始化預測器
        
        參數:
            model_path (str): 模型路徑
            model_size (str): 模型大小 ('small' 或 'large')
            device (torch.device): 運行設備
        """
        # 設置設備
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"使用設備: {self.device}")
        
        # 載入模型
        self.model = get_model(model_size=model_size, pretrained=False)
        
        # 載入權重 - 使用更寬容的方式
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 嘗試直接載入
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("成功直接載入模型權重")
            except Exception as e:
                print(f"直接載入失敗: {str(e)}")
                print("嘗試強制匹配權重...")
                
                # 獲取模型的狀態字典和檢查點的狀態字典
                model_state_dict = self.model.state_dict()
                checkpoint_state_dict = checkpoint['model_state_dict']
                
                # 創建新的狀態字典，僅包含名稱匹配的參數
                new_state_dict = {}
                for k, v in model_state_dict.items():
                    # 尋找相同的層名
                    if k in checkpoint_state_dict and v.size() == checkpoint_state_dict[k].size():
                        new_state_dict[k] = checkpoint_state_dict[k]
                    else:
                        # 如果找不到匹配，保留初始化權重
                        print(f"警告: 無法加載層 {k}，保留初始化權重")
                
                # 加載新的狀態字典
                self.model.load_state_dict(new_state_dict, strict=False)
                print("已使用部分匹配加載權重")
                
            self.model.to(self.device)
            self.model.eval()
            
            print(f"成功載入模型: {model_path}")
        except Exception as e:
            print(f"載入模型失敗: {str(e)}")
            raise
        
        # 數據轉換
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 載入人臉檢測器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("警告: 無法載入人臉檢測器。請確保OpenCV正確安裝。")
    
    def detect_faces(self, img):
        """
        檢測圖片中的人臉
        
        參數:
            img (numpy.ndarray): 輸入圖片 (BGR格式)
            
        返回:
            list: 人臉坐標 [x, y, w, h]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def predict_image(self, img_path):
        """
        預測單張圖片
        
        參數:
            img_path (str): 圖片路徑
            
        返回:
            tuple: (年齡, 性別) 或 None (如果沒有檢測到人臉)
        """
        # 讀取圖片
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            return None
        
        # 檢測人臉
        faces = self.detect_faces(img)
        
        if len(faces) == 0:
            print("未檢測到人臉")
            return None
        
        results = []
        
        # 處理每個人臉
        for (x, y, w, h) in faces:
            # 裁剪人臉
            face_img = img[y:y+h, x:x+w]
            
            # 轉換為PIL圖片
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
            # 應用轉換
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # 預測
            with torch.no_grad():
                pred_age, pred_gender = self.model(face_tensor)
                
                # 處理預測結果
                age = pred_age.item()
                gender_prob = torch.softmax(pred_gender, dim=1)
                gender_idx = torch.argmax(gender_prob, dim=1).item()
                gender = "Male" if gender_idx == 0 else "Female"
                gender_confidence = gender_prob[0][gender_idx].item()
                
                results.append({
                    'box': (x, y, w, h),
                    'age': age,
                    'gender': gender,
                    'gender_confidence': gender_confidence
                })
        
        # 在圖片上繪製結果
        result_img = img.copy()
        for res in results:
            x, y, w, h = res['box']
            
            # 繪製人臉框
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 繪製標籤
            label = f"{res['gender']} {res['age']:.1f}yrs ({res['gender_confidence']:.2f})"
            cv2.putText(result_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存和顯示結果
        output_path = f"result_{os.path.basename(img_path)}"
        cv2.imwrite(output_path, result_img)
        print(f"結果已保存至: {output_path}")
        
        # 顯示結果
        cv2.imshow('Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return results
    
    def predict_camera(self):
        """從攝像頭預測年齡和性別"""
        # 開啟攝像頭
        cap = cv2.VideoCapture(0)
        
        # 檢查攝像頭是否成功開啟
        if not cap.isOpened():
            print("無法開啟攝像頭")
            return
        
        print("按 'q' 退出")
        
        while True:
            # 讀取一幀
            ret, frame = cap.read()
            
            if not ret:
                print("無法獲取影像幀")
                break
            
            # 檢測人臉
            faces = self.detect_faces(frame)
            
            # 處理每個人臉
            for (x, y, w, h) in faces:
                # 裁剪人臉
                face_img = frame[y:y+h, x:x+w]
                
                # 轉換為PIL圖片
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                
                # 應用轉換
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                
                # 預測
                with torch.no_grad():
                    pred_age, pred_gender = self.model(face_tensor)
                    
                    # 處理預測結果
                    age = pred_age.item()
                    gender_prob = torch.softmax(pred_gender, dim=1)
                    gender_idx = torch.argmax(gender_prob, dim=1).item()
                    gender = "Male" if gender_idx == 0 else "Female"
                    gender_confidence = gender_prob[0][gender_idx].item()
                
                # 繪製人臉框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 繪製標籤
                label = f"{gender} {age:.1f}yrs ({gender_confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示結果
            cv2.imshow('Age and Gender Prediction', frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 釋放資源
        cap.release()
        cv2.destroyAllWindows()

def select_model_path():
    """互動式選擇模型路徑"""
    print("\n=== 選擇模型檔案 ===")
    
    # 檢查默認模型路徑
    default_path = CONFIG['model_path']
    if os.path.exists(default_path):
        print(f"找到默認模型: {default_path}")
        use_default = input("使用此模型? (y/n): ").strip().lower()
        if use_default == 'y':
            return default_path
    
    # 檢查output目錄
    output_dir = './output'
    if os.path.exists(output_dir):
        models = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
        if models:
            print("\n可用模型:")
            for i, model in enumerate(models):
                print(f"{i+1}. {model}")
            
            try:
                choice = int(input("\n請選擇模型 (輸入編號): ").strip())
                if 1 <= choice <= len(models):
                    return os.path.join(output_dir, models[choice-1])
            except ValueError:
                pass
    
    # 手動輸入路徑
    while True:
        path = input("\n請輸入模型檔案的完整路徑: ").strip()
        if os.path.exists(path) and path.endswith('.pth'):
            return path
        print("無效的路徑，請重試")

def select_model_size():
    """互動式選擇模型大小"""
    print("\n=== 選擇模型大小 ===")
    print("1. small (更快)")
    print("2. large (更準確)")
    
    # 默認選擇large
    default_choice = 2
    print(f"默認選擇: {default_choice}. large")
    
    while True:
        try:
            choice_input = input("\n請選擇 (1/2，直接按Enter使用默認): ").strip()
            if not choice_input:  # 如果用戶直接按Enter
                return 'large'
                
            choice = int(choice_input)
            if choice == 1:
                return 'small'
            elif choice == 2:
                return 'large'
        except ValueError:
            pass
        print("無效的選擇，請重試")

def select_image_path():
    """互動式選擇圖片路徑"""
    print("\n=== 選擇圖片 ===")
    
    while True:
        path = input("請輸入圖片的完整路徑: ").strip()
        if os.path.exists(path) and path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            return path
        print("無效的圖片路徑，請重試")

def main():
    """主函數"""
    print("\n=== AFAD 年齡和性別預測 ===")
    print("1. 使用攝像頭進行實時預測")
    print("2. 預測單張圖片")
    print("3. 退出")
    
    while True:
        try:
            choice = int(input("\n請選擇功能 (1/2/3): ").strip())
            
            if choice == 3:
                print("再見!")
                break
                
            if choice not in [1, 2]:
                print("無效的選擇，請重試")
                continue
            
            # 選擇模型路徑和大小
            model_path = select_model_path()
            model_size = select_model_size()
            
            # 初始化預測器
            predictor = AgeGenderPredictor(model_path=model_path, model_size=model_size)
            
            # 執行選定功能
            if choice == 1:
                predictor.predict_camera()
            elif choice == 2:
                image_path = select_image_path()
                predictor.predict_image(image_path)
            
            # 詢問是否繼續
            cont = input("\n是否繼續使用其他功能? (y/n): ").strip().lower()
            if cont != 'y':
                print("再見!")
                break
                
        except ValueError:
            print("無效的輸入，請重試")
        except Exception as e:
            print(f"發生錯誤: {str(e)}")

if __name__ == '__main__':
    main() 