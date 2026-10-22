深度学习YOLOv11-seg 无人机裸土覆盖实例分割数据集 航拍裸土覆盖实例分割数据集

航拍裸土覆盖实例分割数据集，1234张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb28cbc356c64c588581eb42ea5d7b39.png)

数据集统计信息
====================
图像尺寸:640*640
类别数量: 4 类
训练集图像:1080; 验证集图像:103; 测试集图像:51
总图像数: 1234

类别分布:![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0604492b2fe647ec9c5d8c59ae1ec779.png)

类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1089e4ee22a24adea4fe204eacd57be1.png)

laji - 垃圾，48,155
shangai - 裸土覆盖，815,3515
shunpoliuzha - 裸土滑落，310,1369
youtong - 油桶，30,41![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d1b0959dcfa14d629c4fbf482857b9e1.png)
# 航拍裸土覆盖实例分割数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eefd142bed96403595e340419e30254f.png)

## 一、数据集信息表
|项目|详情|
| ---- | ---- |
|数据集名称|航拍裸土覆盖实例分割数据集|
|总图像数量|1234张|
|图像尺寸|640×640|
|标注格式|YOLO、VOC、COCO三种|
|类别总数|4类|
|训练集|1080张|
|验证集|103张|
|测试集|51张|

### 类别标签统计表
|类别英文|类别中文|包含该类图像数量|标注目标总数|
| ---- | ---- | ---- | ---- |
|laji|垃圾|48|155|
|shangai|裸土覆盖（防尘网覆盖区域）|815|3515|
|shunpoliuzha|裸土滑落（滑坡渣土）|310|1369|
|youtong|油桶|30|41|

## 二、应用场景
1. **工地扬尘监管**：无人机航拍自动识别工地裸土区域，判断防尘网是否规范覆盖，及时发现未覆盖裸土，辅助环保扬尘巡检。
2. **边坡地质灾害监测**：识别山体边坡裸土滑落、滑坡渣土区域，监测边坡垮塌风险，预警地质隐患。
3. **施工现场环境巡检**：自动识别工地内垃圾、废弃油桶等杂物，辅助工地文明施工督查。
4. **国土空间遥感巡查**：大范围航拍遥感图像批量解析，统计裸土、渣土、废弃物分布面积，用于自然资源、生态环境部门常态化遥感监管。
5. **智慧工地平台接入**：模型部署到巡检平台，无人机回传图像自动分割、量化各类区域面积，生成巡检报告。

## 三、标签说明
1. `shangai`：裸土覆盖，指铺设绿色防尘网的裸土区域，实例分割轮廓圈出防尘网范围；
2. `shunpoliuzha`：裸土滑落，边坡、山体发生滑塌堆积的渣土区域；
3. `laji`：工地散落垃圾、废弃物堆；
4. `youtong`：施工现场废弃油桶。

## YOLOv11-seg 训练yaml配置
```yaml
path: ./bare_soil_dataset
train: train/images
val: val/images
test: test/images

names:
  0: laji
  1: shangai
  2: shunpoliuzha
  3: youtong
```

## 训练代码
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")
    results = model.train(
        data="bare_soil_dataset.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        save=True
    )
```

