import numpy as np
import pandas as pd


def paired(time,anterior,posterior,lo=1,hi=5):
    return dict(window_start_s=time-10,window_end_s=time+10,window_center_s=time,
                first_section=lo,last_section=hi,anterior_ER=anterior,posterior_ER=posterior)


def clinical(n=60):
    x=np.arange(n,dtype=float)
    return pd.DataFrame(dict(age=25+x%15,bmi=19+x%9,infertility_years=1+x%7,
        transfer_embryo_count_component=1+x%2,blastocyst_transfer=x%2,
        endometrial_thickness_mm=7+x%5,endometrial_volume_ml=2+x%4,
        endometrial_pattern_pure_C=x%2,days_before_transfer=1+x%5,
        record_MSE_time_equal=.1+.01*(x%13),patient_id=['synthetic_'+str(i//2) for i in range(n)],
        exam_id=['fixture_'+str(i) for i in range(n)],outcome=(x//2%2).astype(int)))


def oof():
    rows=[]
    for repeat in range(10):
        for fold in range(1,6):
            for index in range(6):
                y=index%2
                p=.15*(fold-1)+.03+.02*y
                rows.append(dict(repeat=repeat,fold=fold,exam_id=f'fixture_{fold}_{index}',
                    patient_id=f'synthetic_{fold}_{index}',outcome=y,baseline=p,augmented=p+.001))
    return pd.DataFrame(rows)
