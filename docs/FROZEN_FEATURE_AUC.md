# 冻结特征的独立结局分析入口

仅在用户授权读取临床结局后使用。结果保留本地，不能上传临床表、分析队列、预测或患者结果。此入口不修改追踪或冻结特征，不以结局选择ROI、频带、显著性阈值或算法参数。

```powershell
python -m experiments.frozen_feature_auc `
  --features <冻结流水线输出目录> `
  --cohort-map <视频与检查编号对应表.xlsx> `
  --clinical <人工复核临床表.xlsx> `
  --output <本地分析输出目录>
```

依赖：numpy、pandas、scikit-learn、openpyxl、matplotlib。沿用既有临床基线7变量、median插补、标准化和Logistic(C=1e6, liblinear)。需使用包含这些依赖的解释器。

按临床检查编号连接，并复核视频研究编号；不从视频路径获取结局。重复检查按患者聚类标识分组，禁止同一患者进入训练折和测试折。默认5折、5次固定种子重复；主结果为第一个预设种子的完整折外AUC，重复均值/范围单列，不选最高值。插补与标准化仅在训练折拟合。

主分析比较相同队列、相同折上的临床基线、双壁主幅度、临床加幅度；另列缺失指示对照、双壁幅度完整病例敏感性分析。缺失幅度保持缺失后在训练折插补，不能写成零活动。

ER、directional RMS和energy fraction尚无冻结的患者级聚合，不进入模型；不自行平均窗口。CC/MSE仅在恰好一个既有协调性block且二者有限的子集作探索分析，不平均或挑选多block结果，临床基线在同一子集重建。

95%区间来自1000次患者聚类bootstrap，对固定折外预测计算配对AUC差；不在bootstrap中重训，故未覆盖全部训练不确定性。已参与算法开发的数据只能作内部探索分析，不代表独立外部验证。临床AUC不代替运动测量准确性验证。

输出：预先记录的protocol、连接审计、分析队列、逐折审计、折外预测、重复指标、配对AUC差、Excel、ROC、报告、源文件SHA256。患者数据及这些输出均不提交仓库。

必要测试：`python -m unittest tests.test_frozen_auc -v`，验证患者分组互斥和插补不读取测试折。

方法依据：[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)、[避免预处理信息泄漏](https://scikit-learn.org/stable/common_pitfalls.html)。
