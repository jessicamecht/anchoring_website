import matplotlib as plt
from eval_utils import * 
from rl_anchoring.utils import * 
from eval_utils import *
from scipy.stats import pearsonr
import pandas as pd

from scipy import stats

def n_steps_since_last_dec(dec):
    none_steps = 0
    nSinceAccept = None
    res = []
    for elem in dec:
        accept = elem
        if accept:
            if nSinceAccept == None:
                # Skip the first one
                nSinceAccept = 0
                none_steps += 1
                continue
            res.append(nSinceAccept)
            nSinceAccept = 0
        else:
            if nSinceAccept != None:
                res.append(nSinceAccept)
                nSinceAccept += 1
            else:
                none_steps += 1

    return np.array(res), none_steps


def correlation_admissions(key):
    data = np.load('./review_data_mturk/admissions_without_reviewer.npy',allow_pickle='TRUE')

    all_prev = []
    all_dec = []

    for review_session in data:
        #timestamp, reviewer_score, final_decision, features, svm_predictions, svm_confidence 
        elem = review_session
        df = pd.DataFrame(elem[:,0], columns = ["timestamp"]) 
        df["reviewer_score"] = elem[:,1]
        df["final_decision"] = elem[:,2]
        df["features"] = elem[:,3]
        df["svm_decision"] = elem[:,4]
        df["svm_confidence"] = elem[:,5]

        dec = df[key].astype(float)
        prev_dec, none_steps = n_steps_since_last_dec(df['reviewer_score'].astype(int))   

        dec = dec[none_steps:]

        all_dec.extend(dec)
        all_prev.extend(prev_dec)
    corr_1, _ = np.corrcoef(all_prev, all_dec)[1]


    return corr_1

def correlation_admissions_to_prev(key, prev_nb):
    data = np.load('./review_data_mturk/admissions_without_reviewer.npy',allow_pickle='TRUE')

    all_prev = []
    all_dec = []

    for review_session in data:
        #timestamp, reviewer_score, final_decision, features, svm_predictions, svm_confidence 
        elem = review_session
        df = pd.DataFrame(elem[:,0], columns = ["timestamp"]) 
        df["reviewer_score"] = elem[:,1]
        df["final_decision"] = elem[:,2]
        df["features"] = elem[:,3]
        df["svm_decision"] = elem[:,4]
        df["svm_confidence"] = elem[:,5]

        dec = df[key].astype(float)[prev_nb:]
        prev_dec = df[key].astype(float)[:-prev_nb] 

        all_dec.extend(dec)
        all_prev.extend(prev_dec)
    corr_1, p = stats.pearsonr(all_prev, all_dec)
    print(corr_1, p, 'p')
    return corr_1

def correlation_items_to_prev(key, prev_nb):
    data = np.load('./review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')

    all_prev = []
    all_dec = []

    for review_session in data:
        elem = review_session
        df = pd.DataFrame(elem[:,0], columns = ["svm_decision"]) 
        df["svm_confidence"] = elem[:,1]
        df["features"] = elem[:,2]
        df["item_number"] = elem[:,3]
        df["original_review"] = elem[:,4]
        df["item_number"] = elem[:,5]
        df["turker_review"] = elem[:,6]

        dec = df[key].astype(float)[prev_nb:]
        prev_dec = df[key].astype(float)[:-prev_nb] 

        all_dec.extend(dec)
        all_prev.extend(prev_dec)
    corr_1, p = stats.pearsonr(all_prev, all_dec)
    print(corr_1, p, 'p')
    return corr_1


def correlation_items(key):
    data = np.load('./review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')


    all_prev = []
    all_dec = []

    for review_session in data:
        elem = review_session
        df = pd.DataFrame(elem[:,0], columns = ["svm_decision"]) 
        df["svm_confidence"] = elem[:,1]
        df["features"] = elem[:,2]
        df["item_number"] = elem[:,3]
        df["original_review"] = elem[:,4]
        df["item_number"] = elem[:,5]
        df["turker_review"] = elem[:,6]

        dec = df[key].astype(float)
        prev_dec, none_steps = n_steps_since_last_dec(df[key])   
        dec = dec[none_steps:]
        all_dec.extend(dec)
        all_prev.extend(prev_dec)
    corr_1, _ = np.corrcoef(all_prev, all_dec)[1]

    return corr_1

def correlation_resampled(key="turker_decision", ai_path ="./data/data_ai/"):
    data = load_data(ai_path)

    all_prev = []
    all_dec = []

    for review_session in data:
        dec = review_session[key].astype(float)
        prev_dec, none_steps = n_steps_since_last_dec(review_session[key])   
        dec = dec[none_steps:]
        all_dec.extend(dec)
        all_prev.extend(prev_dec)
    corr_1, _ = np.corrcoef(all_prev, all_dec)[1]

    return corr_1

def correlation_to_prev_resampled(key, i):
    ai_path ="./data/data_ai/"
    data = load_data(ai_path)

    all_prev = []
    all_dec = []

    for review_session in data:
        dec = review_session[key].astype(float)[i:]
        prev_dec = review_session[key].astype(float)[:-i] 
        all_dec.extend(dec)
        all_prev.extend(prev_dec)
    corr_1, _ = np.corrcoef(all_prev, all_dec)[1]

    return corr_1

def agreement_to_orig_resampled(key="turker_decision", ai_path="./data/data_ai/" ):
    data = load_data(ai_path)

    all_orig = []
    all_dec = []

    for review_session in data:
        dec = review_session[key].astype(float)
        original_rating = review_session["originalRating"] > 3
        all_dec.extend(dec)
        all_orig.extend(original_rating)
    return np.array((np.array(all_dec) == np.array(all_orig))).sum()/len(all_dec)

def agreement_to_original_items(key = "turker_review"):
    data = np.load('./review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')

    all_orig = []
    all_dec = []

    for review_session in data:
        elem = review_session
        df = pd.DataFrame(elem[:,0], columns = ["svm_decision"]) 
        df["svm_confidence"] = elem[:,1]
        df["features"] = elem[:,2]
        df["item_number"] = elem[:,3]
        df["original_review"] = elem[:,4] > 3
        df["item_number"] = elem[:,5]
        df["turker_review"] = elem[:,6]

        orig_dec = df["original_review"].astype(float)
        dec = df[key] 
        all_dec.extend(dec)
        all_orig.extend(orig_dec)
    
    return np.array((np.array(all_dec) == np.array(all_orig))).sum()/len(all_dec)

def plot_admissions_corr():
    all_corrs = []
    all_corrs_svm = []
    for i in [1,2,3,4,5,10,15]:
        all_corrs.append(correlation_admissions_to_prev('reviewer_score', i))
        all_corrs_svm.append(correlation_admissions_to_prev('svm_decision', i))
    x = [0,1,2,3,4,5,6]
    plt.plot(x,all_corrs, linestyle='--', marker='o', color="tab:blue")
    plt.plot(x,all_corrs_svm, linestyle='--', marker='o', color="black")
    plt.xticks(x, ['1','2','3','4','5','10','15'])
    plt.legend(["Human Decisions","SVM Decisions"])
    plt.savefig('corrs.png')

def plot_items_corr():
    all_corrs = []
    all_corrs_svm = []
    all_corrs_resampled = []
    mean_3_4_svm = 0
    mean_3_4_human = 0
    mean_3_4_resampled = 0

    mean_5_15_svm = 0
    mean_5_15_human = 0
    mean_5_15_resampled = 0

    mean_15_svm = 0
    mean_15_human = 0
    mean_15_resampled = 0
    for i in range(1,21):
        print(i)
        if i >=3 and i <=4:
            mean_3_4_human+=correlation_items_to_prev('turker_review', i)
            mean_3_4_svm+=correlation_items_to_prev('svm_decision', i)
            mean_3_4_resampled+=correlation_to_prev_resampled('turker_decision', i)
            if i == 4:
                all_corrs.append(mean_3_4_human/2)
                all_corrs_svm.append(mean_3_4_svm/2)
                all_corrs_resampled.append(mean_3_4_resampled/2)
        elif i >=5 and i <=15:
            mean_5_15_human+=correlation_items_to_prev('turker_review', i)
            mean_5_15_svm+=correlation_items_to_prev('svm_decision', i)
            mean_5_15_resampled+=correlation_to_prev_resampled('turker_decision', i)
            if i == 15:
                all_corrs.append(mean_5_15_human/10)
                all_corrs_svm.append(mean_5_15_svm/10)
                all_corrs_resampled.append(mean_5_15_resampled/10)
        elif i>15:
            mean_15_human+=correlation_items_to_prev('turker_review', i)
            mean_15_svm+=correlation_items_to_prev('svm_decision', i)
            mean_15_resampled+=correlation_to_prev_resampled('turker_decision', i)
            if i == 20:
                all_corrs.append(mean_15_human/5)
                all_corrs_svm.append(mean_15_svm/5)
                all_corrs_resampled.append(mean_15_resampled/5)
        else:
            all_corrs.append(correlation_items_to_prev('turker_review', i))
            all_corrs_svm.append(correlation_items_to_prev('svm_decision', i))
            all_corrs_resampled.append(correlation_to_prev_resampled('turker_decision', i))
        
    x = range(len(all_corrs))
    plt.plot(x,all_corrs, linestyle='--', marker='o', color="tab:blue")
    plt.plot(x,all_corrs_svm, linestyle='--', marker='o', color="black")
    plt.plot(x,all_corrs_resampled, linestyle='--', marker='o', color="grey")
    #plt.xticks(x, ['1','2','3-4','5-15','>15'])

    plt.xlabel("Previous Decision")
    plt.ylabel("Pearson Correlation Coefficient")
    
    plt.legend(["Human","SVM", "Resampled Human Decisions"])
    plt.savefig('corrs_items.png')

def plot_graph(bias, agreement):
    #plt.rcParams.update({'font.size': 15})
    print(bias, 'bias')
    print(agreement, 'agreement')
    annotations = ['SVM', 'Human', "Probabilistic", 'Human LSTM+AC', 'Human LSTM+DQN']

    colors = ["lightgrey", 'grey', "tab:blue", "black", "lightskyblue"]

    for i, txt in enumerate(annotations):
        plt.plot(bias[i], agreement[i], color=colors[i], marker="D")
        plt.annotate(txt, (bias[i]+0.002, agreement[i]+0.002), size = 12)
    plt.xlabel("Bias")
    plt.grid(b=True, linewidth=0.25)
    plt.ylim([0.75, 0.87])
    plt.xlim([-0.23, -0.07])
    plt.ylabel("Agreement")
    plt.savefig('auc.png')
    plt.close()

if __name__ == "__main__":
    '''rev_corr_adm = correlation_admissions("reviewer_score")
    svm_corr_adm = correlation_admissions("svm_decision")
    svm_corr =  correlation_items('svm_decision')
    rev_corr = correlation_items('turker_review')
    res_corr = correlation_resampled()
    res_corr_dqn = correlation_resampled(ai_path="./data/data_dqn/")

    orig_agreement_turker = agreement_to_original_items('turker_review')

    orig_agreement_svm = agreement_to_original_items('svm_decision')
    agreement_resampled = agreement_to_orig_resampled()

    agreement_resampled_dqn = agreement_to_orig_resampled(ai_path="./data/data_dqn/")
    '''
    agreement = [0.8044817927170869, 0.764985994397759, 0.8322, 0.8621863037005529, 0.85]#[orig_agreement_svm, orig_agreement_turker, 0.8322, agreement_resampled, agreement_resampled_dqn]
    bias = [-0.09216545720463447, -0.22257749848973915, -0.2237, -0.16320275663315698, -0.19283043533522523]#[svm_corr, rev_corr, -0.2237, res_corr, res_corr_dqn]

    '''bias = [-0.09216545720463447, -0.22257749848973915, -0.2237, -0.16320275663315698, -0.1881513956188553]
    agreement = [0.8044817927170869, 0.764985994397759, 0.8322, 0.8621863037005529, 0.8545454545454545]'''
    plot_graph(bias, agreement)

    '''print(rev_corr_adm, 'rev_corr_adm')
    print(svm_corr_adm, 'svm_corr_adm')
    print(svm_corr, 'svm_corr')
    print(rev_corr, 'rev_corr')
    print(res_corr, 'res_corr')
    print(orig_agreement_turker, 'orig_agreement_turker')
    print(orig_agreement_svm, 'orig_agreement_svm')
    print(agreement_resampled, "agreement_resampled")'''
    #plot_items_corr()
    

    
    
