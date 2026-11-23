import pandas as pd
from src.data import prepare
from src.final_experiment import leakage_audit, run
import joblib

def test_huggingface_column_alias():
    df = pd.DataFrame({'resume_text':['Python'],'job_description_text':['Python engineer'],'label':['Good Fit']})
    clean, _ = prepare(df)
    assert clean.loc[0,'job'] == 'Python engineer'

def test_overlap_audit_distinguishes_resume_from_pair():
    a,_ = prepare(pd.DataFrame({'resume_text':['Python dev'], 'job_description_text':['Engineer'], 'label':['Good Fit']}))
    b,_ = prepare(pd.DataFrame({'resume_text':['Python dev'], 'job_description_text':['Analyst'], 'label':['Potential Fit']}))
    audit = leakage_audit(a,b)
    assert audit['resume_overlap_test_rows'] == 1
    assert audit['pair_id_overlap_test_rows'] == 0

def test_saved_model_scores_match_reloaded_model(tmp_path):
    labels=['No Fit','Potential Fit','Good Fit']*12
    df=pd.DataFrame({'resume_text':[f'python analyst {i}'+(' sql' if i%2 else '') for i in range(36)],
                     'job_description_text':[f'python developer {i} sql' for i in range(36)],'label':labels})
    train=tmp_path/'train.csv'; test=tmp_path/'test.csv'
    df.to_csv(train,index=False)
    pd.DataFrame({'resume_text':['python sql analyst'], 'job_description_text':['python sql developer'], 'label':['Good Fit']}).to_csv(test,index=False)
    out=tmp_path/'out'; run(train,test,out)
    m=joblib.load(out/'model.joblib')['matcher']
    assert m.score('python sql analyst','python sql developer') == joblib.load(out/'model.joblib')['matcher'].score('python sql analyst','python sql developer')
