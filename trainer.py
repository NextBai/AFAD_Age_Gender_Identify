import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from sklearn.model_selection import train_test_split

# 訓練配置（直接定義，不使用命令行參數）
CONFIG = {
    'data_dir': './datasset',        # 數據目錄
    'output_dir': './output',        # 輸出目錄
    'model_size': 'large',           # MobileNetV3模型大小 ('small' 或 'large')
    'batch_size': 64,                # 批次大小
    'num_epochs': 100,               # 訓練輪數
    'learning_rate': 0.001,          # 學習率
}

#---------------------------------------------------------------------------------------
# 數據加載模塊
#---------------------------------------------------------------------------------------

class AFADDataset(Dataset):
    """
    AFAD (Asian Face Age Dataset) 數據集載入器
    """
    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.8):
        """
        參數:
            root_dir (string): AFAD 數據集根目錄
            transform (callable, optional): 可選的數據轉換
            split (string): 'train' 或 'val' 分割
            train_ratio (float): 訓練數據的比例
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # 獲取所有圖片路徑和標籤
        self.image_paths = []
        self.ages = []
        self.genders = []
        
        # 遍歷年齡目錄 (15-70)
        for age_dir in sorted(os.listdir(root_dir)):
            if not age_dir.isdigit():
                continue
                
            age = int(age_dir)
            age_path = os.path.join(root_dir, age_dir)
            
            # 遍歷性別目錄 (111=男性, 112=女性)
            for gender_dir in os.listdir(age_path):
                if gender_dir == '111':
                    gender = 0  # 男性
                elif gender_dir == '112':
                    gender = 1  # 女性
                else:
                    continue
                
                gender_path = os.path.join(age_path, gender_dir)
                
                # 獲取該年齡和性別下的所有圖片
                for img_name in os.listdir(gender_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(gender_path, img_name)
                        self.image_paths.append(img_path)
                        self.ages.append(age)
                        self.genders.append(gender)
        
        # 訓練/驗證分割
        indices = np.arange(len(self.image_paths))
        train_indices, val_indices = train_test_split(
            indices, train_size=train_ratio, random_state=42, stratify=self.ages
        )
        
        if split == 'train':
            self.indices = train_indices
        else:  # 'val'
            self.indices = val_indices
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """獲取一個樣本"""
        idx = self.indices[idx]
        
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        gender = self.genders[idx]
        
        # 讀取圖片
        image = Image.open(img_path).convert('RGB')
        
        # 應用轉換
        if self.transform:
            image = self.transform(image)
        
        return image, age, gender  # 返回圖片、年齡和性別

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    創建數據載入器
    
    參數:
        data_dir (string): 數據集路徑
        batch_size (int): 批次大小
        num_workers (int): 數據載入的工作進程數
        
    返回:
        train_loader, val_loader (DataLoader): 訓練和驗證數據載入器
    """
    # 數據增強和預處理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 創建數據集
    train_dataset = AFADDataset(root_dir=data_dir, transform=train_transform, split='train')
    val_dataset = AFADDataset(root_dir=data_dir, transform=val_transform, split='val')
    
    # 創建數據載入器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

#---------------------------------------------------------------------------------------
# 模型定義模塊
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

def get_loss_functions():
    """
    獲取損失函數
    
    返回:
        age_criterion, gender_criterion: 年齡和性別的損失函數
    """
    # 年齡使用MSE損失 (回歸問題)
    age_criterion = nn.MSELoss()
    
    # 性別使用交叉熵損失 (分類問題)
    gender_criterion = nn.CrossEntropyLoss()
    
    return age_criterion, gender_criterion

#---------------------------------------------------------------------------------------
# 訓練模塊
#---------------------------------------------------------------------------------------

def train_model(data_dir, output_dir, model_size='large', batch_size=64, 
                num_epochs=100, learning_rate=0.001, device=None):
    """
    訓練模型
    
    參數:
        data_dir (str): 數據目錄
        output_dir (str): 輸出目錄
        model_size (str): MobileNetV3模型大小 ('small' 或 'large')
        batch_size (int): 批次大小
        num_epochs (int): 訓練輪數
        learning_rate (float): 學習率
        device (torch.device): 訓練設備
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置設備
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 獲取數據載入器
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    print(f"訓練樣本數: {len(train_loader.dataset)}, 驗證樣本數: {len(val_loader.dataset)}")
    
    # 獲取模型
    model = get_model(model_size=model_size, pretrained=True)
    model = model.to(device)
    
    # 獲取損失函數
    age_criterion, gender_criterion = get_loss_functions()
    
    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 學習率調度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # 記錄最佳模型
    best_val_loss = float('inf')
    
    # 訓練循環
    for epoch in range(num_epochs):
        print(f"\n第 {epoch+1}/{num_epochs} 輪訓練開始")
        start_time = time.time()
        
        # 訓練階段
        model.train()
        train_age_loss = 0.0
        train_gender_loss = 0.0
        train_total_loss = 0.0
        
        for batch_idx, (images, ages, genders) in enumerate(train_loader):
            images = images.to(device)
            ages = ages.float().to(device)
            genders = genders.long().to(device)
            
            # 前向傳播
            pred_ages, pred_genders = model(images)
            
            # 計算損失
            age_loss = age_criterion(pred_ages, ages)
            gender_loss = gender_criterion(pred_genders, genders)
            
            # 總損失 (可以調整權重)
            total_loss = age_loss + 0.5 * gender_loss
            
            # 反向傳播和優化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 累積損失
            train_age_loss += age_loss.item()
            train_gender_loss += gender_loss.item()
            train_total_loss += total_loss.item()
            
            # 每10批次打印一次進度
            if (batch_idx + 1) % 10 == 0:
                print(f"批次 {batch_idx+1}/{len(train_loader)}, "
                      f"總損失: {total_loss.item():.4f}, "
                      f"年齡損失: {age_loss.item():.4f}, "
                      f"性別損失: {gender_loss.item():.4f}")
        
        # 計算平均訓練損失
        train_age_loss /= len(train_loader)
        train_gender_loss /= len(train_loader)
        train_total_loss /= len(train_loader)
        
        # 驗證階段
        model.eval()
        val_age_loss = 0.0
        val_gender_loss = 0.0
        val_total_loss = 0.0
        val_age_mae = 0.0
        val_gender_acc = 0.0
        
        with torch.no_grad():
            for images, ages, genders in val_loader:
                images = images.to(device)
                ages = ages.float().to(device)
                genders = genders.long().to(device)
                
                # 前向傳播
                pred_ages, pred_genders = model(images)
                
                # 計算損失
                age_loss = age_criterion(pred_ages, ages)
                gender_loss = gender_criterion(pred_genders, genders)
                total_loss = age_loss + 0.5 * gender_loss
                
                # 累積損失
                val_age_loss += age_loss.item()
                val_gender_loss += gender_loss.item()
                val_total_loss += total_loss.item()
                
                # 計算年齡MAE
                val_age_mae += torch.abs(pred_ages - ages).mean().item()
                
                # 計算性別準確率
                _, pred_gender = torch.max(pred_genders, 1)
                val_gender_acc += (pred_gender == genders).float().mean().item()
        
        # 計算平均驗證損失和指標
        val_age_loss /= len(val_loader)
        val_gender_loss /= len(val_loader)
        val_total_loss /= len(val_loader)
        val_age_mae /= len(val_loader)
        val_gender_acc /= len(val_loader)
        
        # 更新學習率
        scheduler.step(val_total_loss)
        
        # 記錄到TensorBoard
        writer.add_scalar('Loss/train_total', train_total_loss, epoch)
        writer.add_scalar('Loss/train_age', train_age_loss, epoch)
        writer.add_scalar('Loss/train_gender', train_gender_loss, epoch)
        writer.add_scalar('Loss/val_total', val_total_loss, epoch)
        writer.add_scalar('Loss/val_age', val_age_loss, epoch)
        writer.add_scalar('Loss/val_gender', val_gender_loss, epoch)
        writer.add_scalar('Metrics/val_age_mae', val_age_mae, epoch)
        writer.add_scalar('Metrics/val_gender_acc', val_gender_acc, epoch)
        
        # 打印訓練和驗證結果
        elapsed_time = time.time() - start_time
        print(f"第 {epoch+1}/{num_epochs} 輪完成，耗時 {elapsed_time:.2f} 秒")
        print(f"訓練損失: {train_total_loss:.4f} (年齡: {train_age_loss:.4f}, 性別: {train_gender_loss:.4f})")
        print(f"驗證損失: {val_total_loss:.4f} (年齡: {val_age_loss:.4f}, 性別: {val_gender_loss:.4f})")
        print(f"驗證指標: 年齡MAE: {val_age_mae:.2f}, 性別準確率: {val_gender_acc*100:.2f}%")
        
        # 保存最佳模型
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            model_path = os.path.join(output_dir, f'best_model_{model_size}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'val_age_mae': val_age_mae,
                'val_gender_acc': val_gender_acc
            }, model_path)
            print(f"保存最佳模型到 {model_path}")
        
        # 每10輪保存一次檢查點
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_ep{epoch+1}_{model_size}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss
            }, checkpoint_path)
            print(f"保存檢查點到 {checkpoint_path}")
    
    # 保存最終模型
    final_model_path = os.path.join(output_dir, f'final_model_{model_size}.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_total_loss,
        'val_age_mae': val_age_mae,
        'val_gender_acc': val_gender_acc
    }, final_model_path)
    print(f"保存最終模型到 {final_model_path}")
    
    writer.close()
    print("訓練完成！")

if __name__ == '__main__':
    # 使用 Kaggle P100 GPU 進行訓練
    print("啟動 AFAD 年齡和性別預測模型訓練...")
    print(f"使用模型: MobileNetV3 {CONFIG['model_size']}")
    
    # 訓練模型
    train_model(
        data_dir=CONFIG['data_dir'],
        output_dir=CONFIG['output_dir'],
        model_size=CONFIG['model_size'],
        batch_size=CONFIG['batch_size'],
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    ) 