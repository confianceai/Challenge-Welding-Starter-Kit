"""
This module contains all intermediate functions involved in uncertainty metrics computations (metrics u_gain, and u_calib)
"""

from abc import ABC
import numpy as np
import sklearn as sk
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,average_precision_score
from sklearn.calibration import calibration_curve

def reduce_mask_and_round(metric,mask=None,reduce=True,round=3):
    if(mask):
        metric = metric[:,mask]
    if(reduce):
        if(reduce == 'sum'):
            metric = metric.sum()
        else:
            metric = metric.mean()
    if(round):
        metric = np.round(metric,round)
    return(metric)


def compute_cost_matrix(CIA=0.41,COP=10,CFN=150,CFP=1000,P_op_FN=0.003,P_op_FP=0.007):
    """_summary_

    Args:
        CIA (int, optional): Cout inference IA. Defaults to 0.1.
        COP (int, optional): Cout operator. Defaults to 10.
        CFN (int, optional): Cout faux negatif. Defaults to 150.
        CFP (int, optional): Cout faux positif. Defaults to 1000.
        P_op_FN (float, optional): Prob error FN operator. Defaults to 0.003.
        P_op_FP (float, optional): Prob error FP operator Defaults to 0.007.
    """
    cost_matrix = np.array([[CIA+COP+P_op_FN*CFN,CIA+CFP,CIA+COP+P_op_FN*CFN],
                            [CIA+COP+P_op_FP*CFP,CIA,CIA+COP+P_op_FP*CFP],
                            [CIA+COP+P_op_FN*CFN,CIA+CFP,CIA+COP+P_op_FN*CFN]])
    return(cost_matrix)


def reformat_prob(y_pred,y_prob,n_class,with_unknown):
    if ((n_class==2) & (len(y_prob.shape)==1)):
        y_prob = np.concatenate([1-y_prob[:,None],y_prob[:,None]],axis=1)
    elif ((n_class==2) & (y_prob.shape[1])==1):
        y_prob = np.concatenate([1-y_prob,y_prob],axis=1)
    elif (n_class==y_prob.shape[1]):
        pass
    elif (n_class==(y_prob.shape[1]-1)) & with_unknown:
        pass # Unkown probability already include
    else:
        raise(TypeError('dimension of prob does not mach with n_class'))

    if  ((n_class==y_prob.shape[1]) & (with_unknown)):
        y_prob = np.concatenate([y_prob,y_prob*0],axis=1)
        mask_unknown = (y_pred == n_class)
        # Divide actual mass by 2 and affect half of the mass probabilty to unknown class.
        y_prob[mask_unknown,-1]=1
        y_prob[mask_unknown]= (y_prob[mask_unknown]/2)
    return(y_prob)

def guess_class_number(y_true,y_prob):
    """Guess_class_number from ground_truth and y_prob tensor

    Args:
        y_true (_type_): ground truth
        y_prob (_type_): y_prob
    """
    n_class = y_true.max() + 1
    if(len(y_prob.shape)>1):
        n_class= max(n_class,y_prob.shape[1])
    return(n_class)

def U_output_management(y_true,output,n_class='auto',set_=None,epsilon=1e-10,with_unknown='False'):
    """Interpret Output and perform subselection

    Args:
        output (_type_): Model ouput
        set_ (_type_): subselection index or None
    """
    if(set_ is not None):
        y_true = y_true[set_]
        y_pred = output[0][set_]
        y_prob = output[1][set_]
    else:
        y_pred,y_prob = output
    
    y_prob = np.clip(y_prob,epsilon,1-epsilon)
    if(n_class=='auto'):
        n_class = guess_class_number(y_true,y_prob)
    y_prob = reformat_prob(y_pred,y_prob,n_class,with_unknown)
    return(y_true,y_pred,y_prob,n_class)


def U_metrics(y_true, output, n_class='auto',with_unknown=False,cost_matrix=None,mode='confusion',epsilon=1e-10,set_=None,mask=None,reduce=True,round=3, **kwargs):
    y_true,y_pred,y_prob,n_class = U_output_management(y_true,output,n_class=n_class,set_=set_,epsilon=epsilon,with_unknown=with_unknown)
    # Selection of sample
    flag_diff_fluzzy = False
    if(mode == 'diff_fluzzy'):
        flag_diff_fluzzy = True
        mode = "fluzzy_confusion"
    
    flag_diff_gap = False
    if(mode == 'diff_gap_prob_quad'):
        flag_diff_gap = True
        mode = "gap_prob_quad"

    conf_matrix_perfect = confusion_matrix(y_true,y_true,labels=np.arange(n_class))
    if(conf_matrix_perfect.shape[1]!=(n_class+with_unknown)):
        conf_matrix_perfect = np.concatenate([conf_matrix_perfect,np.zeros(n_class)[:,None]],axis=1)
        conf_matrix_perfect = np.concatenate([conf_matrix_perfect,np.zeros(n_class+with_unknown)[None,:]],axis=0)

    if(mode=='confusion'):
        conf_matrix = confusion_matrix(y_true,y_pred,labels=np.arange(n_class))

    else:
        conf_matrix=np.zeros((n_class+with_unknown,n_class+with_unknown))
        for label in range(n_class):
            for class_predicted in range(n_class+with_unknown):
                mask_class = (y_true==label)
                true_prob = float(label == class_predicted)
                if(mode=='fluzzy_confusion'):
                    conf_matrix[label,class_predicted] += y_prob[mask_class,class_predicted].sum()
                
                elif(mode=='gap_prob_linear'):
                    conf_matrix[label,class_predicted] += np.abs((true_prob-y_prob[mask_class,class_predicted])).sum()

                elif(mode=='gap_prob_quad'):
                    conf_matrix[label,class_predicted] += np.power((true_prob-y_prob[mask_class,class_predicted]),2).sum()

                elif(mode=='gap_prob_cross_entropy'):
                    conf_matrix[label,class_predicted] -= true_prob*np.log(y_prob[mask_class,class_predicted]).sum()
                
                elif(mode=='gap_prob_self_entropy'):
                    conf_matrix[label,class_predicted] -= y_prob*np.log(y_prob[mask_class,class_predicted]).sum()
                
                else:
                    raise(TypeError("Mode "+str(mode)+" have to be within list : ['confusion','fluzzy_confusion','gap_prob_linear','gap_prob_quad','gap_prob_entropy']"))
    
    if(cost_matrix is None):
        cost_matrix =  np.ones(n_class+with_unknown,n_class+with_unknown) - np.identity(n_class+with_unknown)

    # Hold case where there is no unknown
    if(conf_matrix.shape[1]!=(n_class+with_unknown)):
        conf_matrix = np.concatenate([conf_matrix,np.zeros(n_class)[:,None]],axis=1)
        conf_matrix = np.concatenate([conf_matrix,np.zeros(n_class+with_unknown)[None,:]],axis=0)
    
    cost = np.multiply(conf_matrix,cost_matrix)

    # Ajout des couts obligatoire (vrai_positif & faux négatifs)
    if(mode in ['gap_prob_linear','gap_prob_quad','gap_prob_entropy']):
        cost+= np.multiply(conf_matrix_perfect,cost_matrix)
    cost = cost/len(y_true)

    if(flag_diff_fluzzy or flag_diff_gap):
        base_conf_matrix = confusion_matrix(y_true,y_pred,labels=np.arange(n_class))
        if(base_conf_matrix.shape[1]!=(n_class+with_unknown)):
            base_conf_matrix = np.concatenate([base_conf_matrix,np.zeros(n_class)[:,None]],axis=1)
            base_conf_matrix = np.concatenate([base_conf_matrix,np.zeros(n_class+with_unknown)[None,:]],axis=0)
        base_cost = np.multiply(base_conf_matrix,cost_matrix)
        base_cost = base_cost/len(y_true)
        cost = cost - base_cost

    cost=cost.T
    metric = reduce_mask_and_round(cost,mask=mask,reduce=reduce,round=round)
    return(metric)
    
###


def accuracy_score_metrics(y_true,output,n_class='auto',with_unknown=False,epsilon=1e-10,set_=None,mask=None,reduce=True,round=3,**kwargs):
    y_true,y_pred,y_prob,n_class = U_output_management(y_true,output,n_class=n_class,set_=set_,epsilon=epsilon,with_unknown=with_unknown) 
    metric = accuracy_score(y_true,y_pred)
    metric = reduce_mask_and_round(metric,mask=mask,reduce=reduce,round=round)
    return(metric)

def roc_auc_metrics(y_true,output,n_class='auto',with_unknown=False,epsilon=1e-10,set_=None,mask=None,reduce=True,round=3,**kwargs):
    if(len(list(set(y_true)))!=1):
        y_true,y_pred,y_prob,n_class = U_output_management(y_true,output,n_class=n_class,set_=set_,epsilon=epsilon,with_unknown=with_unknown) 
        metric = roc_auc_score(y_true,y_prob[:,1])
        metric = reduce_mask_and_round(metric,mask=mask,reduce=reduce,round=round)
    else: 
        metric = np.nan
    return(metric)

def average_precision_metrics(y_true,output,n_class='auto',with_unknown=False,epsilon=1e-10,set_=None,mask=None,reduce=True,round=3,**kwargs):
    if(len(list(set(y_true)))!=1):
        y_true,y_pred,y_prob,n_class = U_output_management(y_true,output,n_class=n_class,set_=set_,epsilon=epsilon,with_unknown=with_unknown) 
        metric = average_precision_score(y_true,y_prob[:,0],pos_label=0)
        metric = reduce_mask_and_round(metric,mask=mask,reduce=reduce,round=round)
    else: 
        metric = np.nan
    return(metric)



def curves_calibration_cost(y_true,output,n_class='auto',with_unknown=False,n_bins=10,mode='quantile',epsilon=1e-10,set_=None,mask=None,reduce=True,round=3,**kwargs):
    y_true,y_pred,y_prob,n_class = U_output_management(y_true,output,n_class=n_class,set_=set_,epsilon=epsilon,with_unknown=with_unknown) 
    calibration_diff = []
    
    for class_ in range(n_class):
        y_true_class_,y_prob_class_ = (y_true==class_), y_prob[:,class_]

        if mode == "quantile":  # Determine bin edges by distribution of data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.percentile(y_prob_class_, quantiles * 100)
        elif mode == "uniform":
            bins = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            raise ValueError(
                "Invalid entry to 'strategy' input. Strategy "
                "must be either 'quantile' or 'uniform'.")

        binids = np.searchsorted(bins[1:-1], y_prob_class_)
        bin_sums = np.bincount(binids, weights=y_prob_class_, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true_class_, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        bin_sums[nonzero] = ((bin_true[nonzero] - bin_sums[nonzero]) / bin_total[nonzero]) 
        bin_sums = np.abs(bin_sums) * (bin_total/bin_total.sum())
        bin_sums=bin_sums.sum()
        calibration_diff.append(bin_sums)

    metric= np.array(calibration_diff)
    metric = reduce_mask_and_round(metric,mask=mask,reduce=reduce,round=round)
    return(metric)

    
class Generic_metric():
    def __init__(
        self,
        ABmetric,
        name="Metric",
        mask=None,
        list_ctx_constraint=None,
        reduce=True,
        round=3,
        **kwarg
    ):
        """Metrics wrapper from function

        Args:
            ABmetric (function): function to wrap
            name (str, optional): name of the metric. Defaults to "Metric".
            mask (_type_, optional): mask for specify a focused dimension on multidimensional task. Defaults to None.
            list_ctx_constraint (_type_, optional): list of ctx_constraint link to context information.
                Defaults to None.
            reduce (bool, optional): if reduce multidimensional using mean. Defaults to True.
        """
        self.ABmetric = ABmetric
        self.mask = mask
        self.name = name
        self.reduce = reduce
        self.list_ctx_constraint = list_ctx_constraint
        self.kwarg = kwarg
        self.round = 3

    def compute(self, y, output, sets=None, context=None, **kwarg):
        perf_res = []
        if self.kwarg != dict():
            kwarg = self.kwarg
        if self.list_ctx_constraint is not None:
            ctx_mask = build_ctx_mask(context, self.list_ctx_constraint)
        
        if sets is not None :
            for set_ in sets:
                if self.list_ctx_constraint is not None:
                    set_ = set_ & ctx_mask

                metric = self.ABmetric(y, output, set_=set_, mask=None,reduce=False,round=None, **kwarg)
                metric = reduce_mask_and_round(metric,mask=self.mask,reduce=self.reduce,round=self.round)
                perf_res.append(metric)
        else:
            metric = self.ABmetric(y, output, set_=None, mask=None,reduce=False,round=None, **kwarg)
            metric = reduce_mask_and_round(metric,mask=self.mask,reduce=self.reduce,round=self.round)

        return metric
