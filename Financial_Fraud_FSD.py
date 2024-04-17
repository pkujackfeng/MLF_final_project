import math
from random import random
import pandas as pd
import numpy as np
from datetime import datetime
import csv
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import warnings
# Ignore all userWarning
warnings.filterwarnings('ignore', category=UserWarning)
from WindPy import w
w.start()

## 1. Get stock list of CSI_500 excluding finance stocks, and the companies should go public before 2015 and not be delisted before 2022
## It takes time to access 'ipo' and 'sec_status' from WIND API, so we export the stock list to csv 
# CSI_500 = (w.wset("indexconstituent","date=2020-01-01;windcode=000905.SH;field=wind_code,sec_name")).Data[0]
# finance = (w.wset("indexconstituent","date=2020-01-01;windcode=882007.WI;field=wind_code,sec_name")).Data[0]
# stock_list_1 = list(set(CSI_500)-set(finance))
# stock_list = []
# for stock in stock_list_1:
#     ipo = w.wss(stock,"ipo_date").Data[0][0]
#     status = w.wss(stock,"sec_status").Data[0][0]
#     if ipo < datetime.strptime("2015-01-01 00:00:00", '%Y-%m-%d %H:%M:%S') and status == 'L':
#         stock_list.append(stock)
# stock_list = pd.DataFrame(stock_list)
# stock_list.to_csv('CSI_500_exclude_finance.csv',index = False)

## 2. Get distribution of first digit of the 3 financial statements
## It takes about 2 hours to access all the numbers, so we export the reult into csv
# def get_frequency(stock,year):
#     # stock = '300251.SZ'
#     # year = 2010
#     frequency = [0,0,0,0,0,0,0,0,0]
#     financials = w.wss(stock, "monetary_cap,tradable_fin_assets,notes_rcv,acct_rcv,financing_a/r,prepay,dvd_rcv,int_rcv,oth_rcv,inventories,consumptive_bio_assets,cont_assets,deferred_exp,hfs_assets,non_cur_assets_due_within_1y,oth_cur_assets,fin_assets_chg_compreh_inc,fin_assets_amortizedcost,debt_invest,oth_debt_invest,fin_assets_avail_for_sale,oth_eqy_instruments_invest,held_to_mty_invest,oth_non_cur_fina_asset,invest_real_estate,long_term_eqy_invest,long_term_rec,fix_assets,const_in_prog,proj_matl,productive_bio_assets,oil_and_natural_gas_assets,prop_right_use,intang_assets,r_and_d_costs,goodwill,long_term_deferred_exp,deferred_tax_assets,loans_and_adv_granted,oth_non_cur_assets,st_borrow,tradable_fin_liab,notes_payable,acct_payable,adv_from_cust,cont_liab,empl_ben_payable,taxes_surcharges_payable,tot_acct_payable,int_payable,dvd_payable,oth_payable,acc_exp,deferred_inc_cur_liab,hfs_liab,non_cur_liab_due_within_1y,st_bonds_payable,oth_cur_liab,lt_borrow,bonds_payable,lease_obligation,lt_payable,lt_empl_ben_payable,specific_item_payable,provisions,deferred_tax_liab,deferred_inc_non_cur_liab,oth_non_cur_liab,cap_stk,other_equity_instruments_PRE,perpetual_debt,cap_rsrv,surplus_rsrv,undistributed_profit,tsy_stk,other_compreh_inc_bs,special_rsrv,prov_nom_risks,cnvd_diff_foreign_curr_stat,unconfirmed_invest_loss_bs,eqy_belongto_parcomsh,minority_int,oper_rev,other_oper_inc,oper_cost,taxes_surcharges_ops,selling_dist_exp,gerl_admin_exp,rd_exp,fin_exp_is,impair_loss_assets,credit_impair_loss,other_oper_exp,net_inc_other_ops,net_gain_chg_fv,net_invest_inc,inc_invest_assoc_jv_entp,ter_fin_ass_income,net_exposure_hedge_ben,net_gain_fx_trans,gain_asset_dispositions,other_grants_inc,opprofit,non_oper_rev,non_oper_exp,net_loss_disp_noncur_asset,tot_profit,tax,unconfirmed_invest_loss_is,net_profit_is,np_belongto_parcomsh,cash_recp_sg_and_rs,recp_tax_rends,other_cash_recp_ral_oper_act,stot_cash_inflows_oper_act,net_fina_instruments_measured_at_fmv,cash_pay_goods_purch_serv_rec,cash_pay_beh_empl,pay_all_typ_tax,other_cash_pay_ral_oper_act,stot_cash_outflows_oper_act,net_cash_flows_oper_act,cash_recp_disp_withdrwl_invest,cash_recp_return_invest,net_cash_recp_disp_fiolta,net_cash_recp_disp_sobu,other_cash_recp_ral_inv_act,stot_cash_inflows_inv_act,cash_pay_acq_const_fiolta,cash_paid_invest,net_incr_pledge_loan,net_cash_pay_aquis_sobu,other_cash_pay_ral_inv_act,stot_cash_outflows_inv_act,cash_recp_cap_contrib,cash_rec_saims,cash_recp_borrow,other_cash_recp_ral_fnc_act,proc_issue_bonds,stot_cash_inflows_fnc_act,cash_prepay_amt_borr,cash_pay_dist_dpcp_int_exp,dvd_profit_paid_sc_ms,other_cash_pay_ral_fnc_act,stot_cash_outflows_fnc_act,net_cash_flows_fnc_act,eff_fx_flu_cash,net_incr_cash_cash_equ_dm,cash_cash_equ_beg_period,cash_cash_equ_end_period","unit=1;rptDate=" + str(year) + "1231;rptType=1").Data
#     for item in financials:
#         number = item[0]
#         if number != np.nan and number > 0:
#             for i in range(len(str(number))):
#                 first_digit = str(number)[i]
#                 if first_digit in ['1','2','3','4','5','6','7','8','9']:
#                     frequency[int(first_digit)-1] += 1
#                     break
#                 i = i+1
#     total_sum = sum(frequency)
#     for i in range(len(frequency)):
#         frequency[i] = frequency[i]/total_sum
#     return frequency
# with open('CSI_500_exclude_finance.csv','r') as file:
#     csv_reader =csv.reader(file)
#     stock_list =[]
#     for row in csv_reader:
#         stock_list.append(row)
# for i in range(len(stock_list)):
#     stock_list[i] = stock_list[i][0]
# First_Digit_Frequency = pd.DataFrame(
#     index = stock_list,
#     columns = pd.MultiIndex.from_product([
#         [2015,2016,2017,2018,2019],
#         [1,2,3,4,5,6,7,8,9]]))
# for stock in stock_list:
#     for year in [2015,2016,2017,2018,2019]:
#         for i in range(9):
#             First_Digit_Frequency.loc[stock,(year,i+1)] = get_frequency(stock,year)[i]
# First_Digit_Frequency.to_csv('First_Digit_Frequency.csv',index = True)

## 3. Get X data
First_Digit_Frequency = pd.read_csv('First_Digit_Frequency.csv')
First_Digit_Frequency.set_index('stock',inplace = True)
First_Digit_Frequency.columns = pd.MultiIndex.from_product([
    [2015,2016,2017,2018,2019],
    [1,2,3,4,5,6,7,8,9]])
# print(First_Digit_Frequency.loc['000975.SZ',(2015,1)])
## Get the difference between the first digit distribution with the Benford distribution
Benford_Frequency = []
for i in range(1,10):
    Benford_Frequency.append(math.log10(1+1/i))
First_Digit_Frequency = First_Digit_Frequency.stack(level = 0)
for i in range(9):
    First_Digit_Frequency.iloc[:,i] = abs(First_Digit_Frequency.iloc[:,i]/Benford_Frequency[i])
First_Digit_Frequency = First_Digit_Frequency.unstack()
X = pd.DataFrame(index = First_Digit_Frequency.index)
for i in range(1,10):
    X[i] = First_Digit_Frequency.loc[:,i].max(axis=1)
X.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9']

## 4. Get y data
## We can get the auditor comments through WIND Excel Plugger
y = pd.read_csv('y.csv')
y = y.set_index('stock')

## 5. Put X and y together
X_y = pd.concat([X, y], axis=1)
print(X_y)
y = y.values.ravel()

## 6. Split into train set and test set
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = random_seed, stratify=y)

## 7.1. Use SMOTE, because the dataset is very imbalanced
# print("Before SMOTE, %s/%s=%s%% samples are y=1." % (np.sum(y_train==1),len(y_train),int(np.mean(y_train)*100)))
# smote = SMOTE(sampling_strategy=1,random_state=random_seed)
# x_new, y_new = smote.fit_resample(X_train, y_train)
# X_train = x_new
# y_train = y_new
# print("After SMOTE, %s/%s=%s%% samples are y=1." % (np.sum(y_train==1),len(y_train),int(np.mean(y_train)*100)))

## 7.2. Use RandomUnderSampler, because the dataset is very imbalanced
print("Before RandomUnderSampler, %s/%s=%s%% samples are y=1." % (np.sum(y_train==1),len(y_train),int(np.mean(y_train)*100)))
rus = RandomUnderSampler(sampling_strategy=1,random_state=random_seed)
x_new, y_new = rus.fit_resample(X_train, y_train)
X_train = x_new
y_train = y_new
print("After RandomUnderSampler, %s/%s=%s%% samples are y=1." % (np.sum(y_train==1),len(y_train),int(np.mean(y_train)*100)))

## 8. PCA feature engineering
# pca = PCA()
# X_train_pca = pca.fit_transform(X_train)
# pca.explained_variance_ratio_
# print(pca.explained_variance_ratio_)
# plt.bar(range(1, 10), pca.explained_variance_ratio_, alpha=0.5, align='center')
# plt.step(range(1, 10), np.cumsum(pca.explained_variance_ratio_), where='mid')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.xticks(ticks=list(range(1,10)))
# for i in range(1,10):
#     plt.text(i, pca.explained_variance_ratio_[i-1], "{:.0%}".format(pca.explained_variance_ratio_[i-1]),fontsize=10,verticalalignment='bottom',horizontalalignment='center')
# plt.show()
# Select features
X_train = X_train.iloc[:,:9]
X_test = X_test.iloc[:,:9]

## 9. For each model, we define a function to evaluate the result for test set
def show_result(y_true, y_predict):
    # Confusion matrix
    matrix_test = confusion_matrix(y_true, y_predict)
    print("Confusion matrix:")
    print(matrix_test)
    # Classification report
    report_test = classification_report(y_true, y_predict)
    print("Classification report:")
    print(report_test)
    fpr,tpr,thresholds1 = roc_curve(y_true,y_predict,drop_intermediate = False)
    # ROC curve
    plt.plot(fpr,tpr,'r*-')
    plt.plot([0,1],[0,1])
    plt.plot("FPR")
    plt.plot("TPR")
    plt.title("ROC Curve")
    plt.plot(fpr,tpr,'r*-')
    plt.plot([0,1],[0,1])
    plt.plot("FPR")
    plt.plot("TPR")
    plt.title("ROC Curve")
    plt.show()
    # AUC score
    print("AUC score:")
    print(roc_auc_score(y_true,y_predict))

# 10.1 LogisticRegression
LR = LogisticRegression(solver='liblinear')
LR_param_grid = {
    'C': [10,20,50,75,100,150,200,250,500,600,700,800,900,1000,1100,1200,1300,1400,1500],
    'penalty': ['l1', 'l2'] }
LR_grid_search = GridSearchCV(LR, LR_param_grid, cv=5, return_train_score=True)
LR_grid_search.fit(X_train, y_train)
# Grid search result for hyperparameters
print("Grid Search for best model", LR_grid_search.best_params_)
LR_best_model = LR_grid_search.best_estimator_
y_LR_predict = LR_best_model.predict(X_test)
show_result(y_test, y_LR_predict)

# # 10.2 MLP Classifier
MLP_grid = {
    'solver': ['adam'],
    'learning_rate_init': [0.001],
    'hidden_layer_sizes': [(10,10),(30,30),(80,80),(100,100),(120,120),(150,150)],
    'activation': ['tanh'], # ['logistic', 'tanh', 'relu'],
    'alpha': [0.0001],
    }
MLP_grid_search = GridSearchCV(MLPClassifier(max_iter = 300,random_state = random_seed), MLP_grid, cv=5, scoring='accuracy')
MLP_grid_search.fit(X_train, y_train)
# Grid search result for hyperparameters
print("Grid Search for best model", MLP_grid_search.best_params_)
MLP_best_model = MLP_grid_search.best_estimator_
y_MLP_predict = MLP_best_model.predict(X_test)
show_result(y_test, y_MLP_predict)

# 10.3 SVM
SVM_grid = {'C': [500,750,1000,1200,1500,1800,2000], 
	'gamma': [10, 0.1, 0.03, 0.01, 0.008, 0.005], 
	'kernel': ['rbf']} 
SVM_grid_search = GridSearchCV(SVC(), SVM_grid, refit = True, verbose = 3) 
SVM_grid_search.fit(X_train, y_train)
# Grid search result for hyperparameters
print("Grid Search for best model", SVM_grid_search.best_params_)
SVM_best_model = SVM_grid_search.best_estimator_
y_SVM_predict = SVM_best_model.predict(X_test)
show_result(y_test, y_SVM_predict)

# 10.4 Decision tree
DT_grid = {'max_depth': [2,4,6,8,10],
    'max_leaf_nodes': list(range(2, 10)),
    'min_samples_split': list(range(2, 10)),
    'min_samples_leaf': [2,3,4]
    }
DT_grid_search = GridSearchCV(DecisionTreeClassifier(random_state = random_seed), DT_grid, verbose=1, cv=3)
DT_grid_search.fit(X_train, y_train)
# Grid search result for hyperparameters
print("Grid Search for best model", DT_grid_search.best_params_)
DT_best_model = DT_grid_search.best_estimator_
y_DT_predict = DT_best_model.predict(X_test)
show_result(y_test, y_DT_predict)

# 10.5 Random forest
RF_grid = {"n_estimators":[800,1000,1200],
    'max_depth':[4,5,6],
    "min_samples_split":[25,30,35]
    }
RF_grid_search = GridSearchCV(RandomForestClassifier(),param_grid = RF_grid,cv=3,scoring='accuracy')
RF_grid_search.fit(X_train, y_train)
# Grid search result for hyperparameters
print("Grid Search for best model", RF_grid_search.best_params_)
RF_best_model = RF_grid_search.best_estimator_
y_RF_predict = RF_best_model.predict(X_test)
show_result(y_test, y_RF_predict)
