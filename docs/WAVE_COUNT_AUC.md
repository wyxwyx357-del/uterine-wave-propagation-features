# 已有显著活动计数的批量应用与AUC对照

此入口仅调用现有计数代码，不改A0、A/epsilon、幅度、ER、CC/MSE、0.6或事件判据，不重启算法优化。用户授权读取结局后，AUC作为独立步骤运行。

```powershell
python -m experiments.batch_wave_counts `
  --features <已有冻结特征目录> `
  --noise-protocol <既有固定噪声参照protocol.json> `
  --output <本地计数结果目录> --workers 2

python -m experiments.wave_count_auc `
  --previous <此前冻结幅度AUC结果目录> `
  --counts <本地计数结果目录> `
  --output <本地波次数AUC结果目录>
```

计数链：已有2秒径向图→periodic+partial→候选内部受限拆分(既有3相邻section)→同壁ownership→全片两壁共同min/max显著性>0.6→跨壁无方向硬否决去重→补回phase-fit的原传播判据→显著传播活动次数。方向仅QC。所有已有native sections均保留，不按患者寻找区域，不强扩展至s=0/1。每例记录实际采样范围；计数范围不混同于幅度ROI。

频率分母复用既有`observation_union`：至少一壁有3相邻有效native sections的measurement interval真实并集。不能加两壁时长，不能用只有活动时段作分母。完全没有有限滤波连续支持的病例记null；有可分析支持但未检出活动记0。失败逐例保留原因，不能写成0。

这仍是“显著传播活动”的算法计数，存在已记录的local_once误计、相邻活动重复身份、固定0.6漏弱活动问题。批量运行与AUC并不解决这些测量有效性问题，不据AUC改变阈值。

AUC复用前次已匹配队列，沿用相同临床7变量、插补、标准化、Logistic参数、患者分组折和5个固定种子；临床及临床+幅度的主折外预测须逐例重现。主要输入是次数/有效分钟，总次数另列敏感性分析。6类预设模型包括临床、幅度、临床+幅度、活动频率、临床+频率、临床+幅度+频率；不根据最高AUC选模型。缺失在训练折插补，完整可用子集单列。95%区间使用患者聚类bootstrap固定折外预测，不重训。

代码测试：`python -m unittest tests.test_batch_wave_counts tests.test_frozen_auc -v`。新增测试只验证未观察不等于零、零活动在有效图上可表示为零及interval并集时长；计数器与既有12例保存输出逐例一致性另保存在本地结果中。

输出包括每例父候选、split、ownership、显著性和计数证据，CSV/Excel、失败状态、固定protocol、源文件哈希，以及AUC逐折审计、折外预测、配对差值、ROC与报告。患者数据、结局、计数和分析结果均只保存在本地outputs，不提交仓库。
