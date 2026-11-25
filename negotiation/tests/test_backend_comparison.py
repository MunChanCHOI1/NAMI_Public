import copy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('backend_comparison',Path(__file__).parents[1]/'compare_backends.py')
C=importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def call(value=0.4, state='public'):
    request={'state':state,'questions':{'C1':{'type':'noul','question':'accept?'}}}
    h=C.digest(request)
    return {'status':'ok','request':request,'request_sha256':h,'predictions':{'C1':value},
            'response':{'metadata':{'request_sha256':h}},'latency_ms':100}


def test_duplicate_requests_and_unmatched_calls_are_not_hidden():
    a=call()
    result=C.compare_calls([a,a,call(state='other')],[call(0.401)])
    assert result['matched_request_occurrences']==1
    assert result['native_only_requests']==2
    assert not result['all_inputs_matched_and_within_tolerance']
    assert C.compare_calls([],[])['maximum_absolute_probability_difference'] is None
    assert not C.compare_calls([],[])['all_inputs_matched_and_within_tolerance']


def test_small_probability_difference_can_change_threshold_decision():
    report=C.compare_calls([call(0.499)],[call(0.501)])
    assert report['all_inputs_matched_and_within_tolerance']
    assert report['threshold_0_5_agreements']==0
    assert len(report['threshold_0_5_disagreements'])==1


@pytest.mark.parametrize('tamper',['request','response','probability','incomplete'])
def test_comparison_rejects_invalid_evidence(tamper):
    result={'complete':True,'calls':[call()]}
    if tamper=='request':result['calls'][0]['request']['state']='changed'
    elif tamper=='response':result['calls'][0]['response']['metadata']['request_sha256']='wrong'
    elif tamper=='probability':result['calls'][0]['predictions']['C1']=float('nan')
    else:result['complete']=False
    with pytest.raises(ValueError):C.checked_calls(result)


def test_same_outcome_does_not_hide_a_changed_action_path():
    a={'cases':[{'case_id':'dev-1','agent':{'status':'agreed','agreement':{'price':5},'realized_saving':5,
        'internal_record':[{'round':1,'action':'counter','chosen':{'kind':'price','terms':{'price':4}}}],
        'executed':[{'action':'counter','terms':{'price':4}}]}}]}
    b=copy.deepcopy(a)
    b['cases'][0]['agent']['internal_record'][0]['chosen']['terms']['price']=3
    report=C.compare_cases(a,b)
    assert report['same_outcomes']==1 and report['same_decision_paths']==0
