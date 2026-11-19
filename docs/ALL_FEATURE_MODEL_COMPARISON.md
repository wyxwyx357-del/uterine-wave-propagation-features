# 全部现有患者级特征的模型比较

用户授权把既有视频特征合并，测试LASSO与SVM。此步骤仅建模，不改追踪、ROI、频带、0.6、活动归属或任何冻结特征。患者数据、结局、预测和结果均保留本地。

```powershell
python -m experiments.all_features_auc `
  --previous <此前波次数AUC输出目录> `
  --output <本地新输出目录>
```

输入为同一已匹配检查队列的7个既有临床变量，以及6个已有视频标量：双壁幅度、显著传播活动次数、次数/有效分钟、既有单block的ER-CC和ER-MSE。多block协调性不自行平均；窗口ER、directional RMS、energy fraction尚无患者聚合定义，不输入模型。可分析性QC不当作新的生理变量。缺失保留到训练折内插补，不以0代表未观察。

预设比较三种模型，分别配对同算法的“临床”与“临床+全部视频”输入：

- 既有Logistic(C=1e6, liblinear)。
- L1惩罚Logistic，即分类任务中的LASSO；内层C为0.01/0.1/1/10。不是用连续结局Lasso回归拟合妊娠标签。
- RBF SVM；内层C为0.1/1/10，gamma为scale/0.01/0.1。

外层5折按患者分组，5个固定种子重复；内层3折同样按患者分组调参，以ROC AUC选参数。每次内层拟合均包括median插补和标准化；L1选择属于训练过程，不先在全数据筛变量。同一患者的重复检查不能跨任一训练/验证边界。未挑最高外层AUC的模型再当成独立验证结果。

主表为第一个预设种子的完整折外AUC与配对差值，重复均值另列。95%区间沿用固定折外预测的患者聚类bootstrap，不重训，未覆盖全部训练不确定性。SVM使用decision_function，不把分数称作校准妊娠概率。逐折AUC也保留以便检查合并折外分数的稳定性。

L1训练折非零系数比例仅作稳定性描述，不在同一数据上据此选变量重新报告AUC。次数与频率相关，L1可能只保留一个；不能因某一折系数为零就断言其无生理意义。

必要测试：`python -m unittest tests.test_all_features_auc -v`，检查嵌套分组隔离及预处理/选择属于模型pipeline。临床固定Logistic折外预测还必须逐例重现之前结果。

模型方法来源：[嵌套交叉验证](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)、[L1 Logistic](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)。
