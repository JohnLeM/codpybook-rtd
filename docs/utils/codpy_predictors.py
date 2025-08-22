import os 
import abc 

import numpy as np 
from codpy import core 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class data_predictor(abc.ABC):
    score,elapsed_predict_time,norm_function,discrepancy_error = np.NaN,np.NaN,np.NaN,np.NaN
    set_kernel,generator = None,None

    def get_params(**kwargs) :
        return kwargs.get('data_predictor',None)

    def __init__(self): pass
    def __init__(self,**kwargs):
        self.set_kernel = kwargs.get(
            'set_kernel',core.kernel_setter("tensornorm", None, polynomial_order=3)
            )
        self.set_kernel()

        #TODO which form should it have????
        # self.accuracy_score_function = kwargs.get(
        #     'accuracy_score_function',
        #     get_relative_mean_squared_error
        #     )
        self.name = kwargs.get('name','data_predictor')
    def get_index(self):
        if (self.generator):return self.generator.get_index()
        else:return []
    def get_index_z(self):
        if (self.generator):return self.generator.get_index_z()
        else:return []
    def get_input_data(self):
        return self.x,self.y,self.z,self.fx,self.fy,self.fz, self.dfx, self.dfz
    def copy_data(self,out):
        out.generator, out.set_kernel = self.generator,self.set_kernel
        out.x,out.y,out.z,out.fx,out.fy, out.fz, out.dfx, out.dfz = self.x.copy(),self.y.copy(),self.z.copy(),self.fx.copy(),self.fy.copy(),self.fz.copy(),self.dfx.copy(),self.dfz.copy()
        out.f_z = self.f_z.copy()
        # out.df_z= self.df_z.copy()
        out.D,out.Nx,out.Ny,out.Nz,out.Df = self.D,self.Nx,self.Ny,self.Nz,self.Df
        out.elapsed_predict_time,out.norm_function,out.discrepancy_error,out.accuracy_score= self.elapsed_predict_time,self.norm_function,self.discrepancy_error,self.accuracy_score
        return out
    def set_data(self,generator,**kwargs):
        import time
        self.generator = generator
        self.D,self.Nx,self.Ny,self.Nz,self.Df = generator.get_input_data()
        self.x,self.y,self.z,self.fx, self.fy, self.fz, self.dfx, self.dfz = generator.get_output_data()
        self.f_z,self.df_z = [],[]
        self.elapsed_predict_time,self.norm_function,self.discrepancy_error,self.accuracy_score = np.NaN,np.NaN,np.NaN,np.NaN
        if (self.D*self.Nx*self.Ny ):
            self.preamble(**kwargs)
            start = time.time()
            self.predictor(**kwargs)
            self.elapsed_predict_time = time.time()-start
            self.validator(**kwargs)

    def column_selector(self,**kwargs):
        variables_cols_drop = kwargs.get('variables_cols_drop',[])
        variables_cols_keep = kwargs.get('variables_cols_keep',[])
        values_cols_drop = kwargs.get('values_cols_drop',[])
        values_cols_keep = kwargs.get('values_cols_keep',[])

        if len(variables_cols_drop) or len(variables_cols_keep):
            self.x = column_selector(self.x,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.y = column_selector(self.y,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
            self.z = column_selector(self.z,cols_drop = variables_cols_drop, cols_keep = variables_cols_keep)
        if len(values_cols_drop) or len(values_cols_keep):
            self.fx = column_selector(self.fx,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            self.fy = column_selector(self.fy,cols_drop = values_cols_drop, cols_keep = values_cols_keep)
            # self.fz = column_selector(self.fz,cols_drop = values_cols_drop, cols_keep = values_cols_keep)


    def get_map_cluster_indices(self,cluster_indices=[],element_indices=[],**kwargs):
        if not len(element_indices): element_indices = self.f_z
        if not len(cluster_indices):
            test = type(self.z)
            switchDict = {np.ndarray: self.get_index_z, pd.DataFrame: lambda : list(self.z.index)}
            if test in switchDict.keys(): cluster_indices = switchDict[test]()
            else:
                raise TypeError("unknown type "+ str(test) + " in standard_scikit_cluster_predictor.get_map_cluster_indices")

        if not len(cluster_indices): return {}
        if len(cluster_indices) == len(element_indices):
            return pd.DataFrame({'key': element_indices,'values':cluster_indices}).groupby("key")["values"].apply(list)
        else: return {}

    def preamble(self,**kwargs):
      return
    @abc.abstractmethod
    def predictor(self,**kwargs):
      pass

    def is_validator_compute(self,field,**kwargs):
        if 'validator_compute' in kwargs:
            debug = field in kwargs.get('validator_compute')
            return debug
        return False

    def validator(self,**kwargs):
        kwargs['set_codpy_kernel'] = kwargs.get("set_codpy_kernel",self.set_kernel)
        kwargs['rescale'] = kwargs.get("rescale",True)
        if len(self.fx) and self.set_kernel:
            if self.is_validator_compute(field ='norm_function',**kwargs): self.norm_function = op.norm(x= self.x,y= self.y,z= self.z,fx = self.fx,**kwargs)
        if len(self.fz)*len(self.f_z):
            if self.is_validator_compute(field = 'accuracy_score',**kwargs): self.accuracy_score = self.accuracy_score_function(self.fz, self.f_z)
        if len(self.x)*len(self.z) and self.set_kernel:
            if ( self.is_validator_compute(field ='discrepancy_error',**kwargs)): self.discrepancy_error = op.discrepancy(x=self.x, y = self.y, z= self.z, **kwargs)


    def get_numbers(self):
        return self.D,self.Nx,self.Ny,self.Nz,self.Df

    def get_output_data(self):
        return self.f_z,self.df_z

    def get_params(**kwargs) :
        return kwargs.get('data_predictor',None)

    def get_new_params(self,**kwargs) :
        return kwargs

    def save_cd_data(self,**kwargs):
        params = data_predictor.get_params(**kwargs)
        if params is not None:
            x_csv,y_csv,z_csv,fx_csv,fz_csv,f_z_csv = params.get('x_csv',None),params.get('y_csv',None),params.get('z_csv',None),params.get('fx_csv',None),params.get('fz_csv',None),params.get('f_z_csv',None)
            if x_csv is not None :  self.x.to_csv(x_csv,sep = ';', index=False)
            if y_csv is not None :  self.y.to_csv(y_csv,sep = ';', index=False)
            if z_csv is not None :  self.z.to_csv(z_csv,sep = ';', index=False)
            if fx_csv is not None : self.fx.to_csv(fx_csv,sep = ';', index=False)
            if fz_csv is not None : self.fz.to_csv(fz_csv,sep = ';', index=False)
            if f_z_csv is not None: self.f_z.to_csv(f_z_csv,sep = ';', index=False)


    def id(self,name = ""):
        return self.name
  

class add_confusion_matrix:
    def confusion_matrix(self):
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.metrics import confusion_matrix
        out = []
        if len(self.fz)*len(self.f_z):out = confusion_matrix(self.fz, self.f_z)
        return out
    def plot_confusion_matrix(predictor ,ax, **kwargs):
        import seaborn as sns
        sns.heatmap(predictor.confusion_matrix(), ax=ax, annot=True, fmt="d", cmap=plt.cm.copper)
        labels = kwargs.get('labels',[str(s) for s in np.unique(predictor.fz)])
        title = kwargs.get('title',"Conf. Mat.:")
        ax.set_title(title, fontsize=14)
        ax.set_xticklabels(labels, fontsize=14, rotation=90)
        ax.set_yticklabels(labels, fontsize=14, rotation=360)


####A standard run

def standard_supervised_run(scenario_generator,scenarios_list,generator,predictor,accumulator,**kwargs):
    scenario_generator.run_scenarios(scenarios_list,generator,predictor,accumulator,**kwargs)
    if bool(kwargs.get("Show_results",True)):
        results = accumulator.get_output_datas().dropna(axis=1)
        print(results)
    if bool(kwargs.get("Show_confusion",True)):accumulator.plot_confusion_matrices(**kwargs,mp_title = "confusion matrices for "+predictor.id())
    if bool(kwargs.get("Show_maps",True)):print(accumulator.get_maps_cluster_indices())

######################### regressors ######################################""
class codpyprRegressor(data_predictor):
    
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            self.f_z = op.projection(x = self.x,y = self.y,z = self.z, fx = self.fx,set_codpy_kernel=self.set_kernel,rescale = True,**kwargs)
            pass
    def id(self,name = ""):
        return "codpy pred"

class codpyexRegressor(data_predictor):
    
    def predictor(self,**kwargs):
        kwargs['set_codpy_kernel'] = kwargs.get('set_codpy_kernel',self.set_kernel)
        kwargs['rescale'] = kwargs.get('rescale',False)
        self.column_selector(**kwargs)
        if (self.D*self.Nx*self.Ny*self.Nz ):
            self.f_z = op.projection(x = self.x,y = self.x,z = self.z, fx = self.fx,**kwargs)
    def id(self,name = ""):
        return "codpy extra"


def codpy_Classifier(**kwargs): #label_codpy_predictor
    f_z = op.projection(**kwargs)
    out= np.zeros(f_z.shape)
    softmaxindice_ = softmaxindice(f_z)
    def helper(n): out[n,softmaxindice_[n]] = 1.
    [helper(n) for n in range(f_z.shape[0])]
    if isinstance(f_z,pd.DataFrame): out= pd.DataFrame(out,columns=f_z.columns)
    return out

def classifier_score_fun(**kwargs) :
    from sklearn.metrics import confusion_matrix
    fz,f_z = softmaxindice(mat = kwargs['fz']),softmaxindice(mat = kwargs['f_z'])
    out = confusion_matrix(fz,f_z)
    print("confusion matrix:",out)
    score = np.trace(out)/np.sum(out) 
    print("overall score :", (score * 100),"%")
    return 1.-score

def softmax_predictor(f_z): return softmaxindice(f_z)
    # out= np.zeros(f_z.shape)
    # softmaxindice_ = softmaxindice(f_z)
    # def helper(n): out[n,softmaxindice_[n]] = 1.
    # [helper(n) for n in range(f_z.shape[0])]
    # if isinstance(f_z,pd.DataFrame): out= pd.DataFrame(out,columns=f_z.columns)
    # return out

def proba_predictor(**kwargs): 
    out = op.projection(**kwargs)
    return out

def proba_classifier(**kwargs):return softmax_predictor(proba_predictor(**kwargs))

def weighted_predictor(**kwargs): 
    fx = kwargs['fx']
    weights = np.zeros(fx.shape[0])
    for col in fx.columns:
        index = list(fx[fx[col]==1.].index)
        nb = len(index)
        if nb: weights[index] = nb
    return(op.weighted_projection(weights=weights,**kwargs))

def weighted_classifier(**kwargs): return softmax_predictor(weighted_predictor(**kwargs))


def get_occurence_count(x):
    cols_ordinal = {}
    for col in x.columns:
        my_value_count = x[col].value_counts()
        my_value_count = my_value_count.loc[my_value_count.index.isin([1.0])]
        if my_value_count.empty : cols_ordinal[col] = 0
        else : cols_ordinal[col] = int(my_value_count)
    cols_ordinal = dict(sorted(cols_ordinal.items(),key=lambda x:x[1]))
    return cols_ordinal

def codpy_rl_classifier(kwargs):
    import random

    random_state=kwargs.get("random_state",42)
    max_number = kwargs.get('max_number',5000)
    # predictor = kwargs.get('predictor',proba_predictor)
    proba_predictor_ = kwargs.get('proba_predictor',proba_predictor)
    batch_size = kwargs.get('batch_size',100)
    if 'y' in kwargs:
        if kwargs['y'].shape[0] >= max_number: return kwargs

    fz= kwargs['fz']
    repartition_rem_index,repartition_rem_count = {},{}
    columns = list(kwargs.get("columns",kwargs['fz'].columns))
    for col in columns : repartition_rem_index[col] = fz[fz[col]==1.].index
    for col in columns : repartition_rem_count[col] = len(repartition_rem_index[col])
    keep_indices = set()
    probas_erreurs = {}

    def error_fun(kwargs): 
        out= pd.DataFrame.abs(kwargs['fz']-kwargs['f_z']).sort_values(ascending = False)
        return out

    def add_indices_fun(**kwargs) :
        erreurs_values = kwargs.get('error_fun',error_fun)(kwargs)
        erreurs = list(erreurs_values.index)[:batch_size]
        test = erreurs_values.loc[erreurs]
        cols_ordinal = get_occurence_count(kwargs["fy"])
        print("repartition training set:",cols_ordinal)
        print("repartition remaining set :",repartition_rem_count)
        add_indices = list()
        added = {}
        for col in cols_ordinal:
            erreurs_list = [item for item in erreurs if item in repartition_rem_index[col]]
            if len(erreurs_list) : probas_erreurs[col] = float(erreurs_values.loc[erreurs_list[0]])
            else: 
                probas_erreurs[col] = 1.
            added[col] = erreurs_list

        print("probas_erreurs:",probas_erreurs)

        for col in added:
            samples_nb = batch_size
            samples_nb= int(min(batch_size,samples_nb))
            repartition_rem_index[col] = repartition_rem_index[col].difference(added[col])
            repartition_rem_count[col] = len(repartition_rem_index[col])
            add_indices = add_indices + list(added[col][:samples_nb])
            cols_ordinal[col] += len(added[col])
        return add_indices

    # def add_indices_fun(**kwargs) :
    #     rl_weights = kwargs.get('rl_weights',{})
    #     erreurs = kwargs.get('error_fun',error_fun)(kwargs)

    #     cols_ordinal = get_occurence_count(kwargs["fy"])
    #     print("repartition training set:",cols_ordinal)
    #     print("repartition remaining set :",repartition_rem_count)
    #     add_indices = list()
    #     added = {}
    #     for col in cols_ordinal:
    #         if col in rl_weights:
    #             weight_ = float(rl_weights[col])
    #             sum_ = np.sum(list(cols_ordinal.values()))
    #             tol = int(weight_*sum_)-cols_ordinal[col]
    #             samples_nb= max(min(batch_size,tol),0)
    #         else:
    #             samples_nb = batch_size
    #         erreurs_list = erreurs[erreurs.index.isin(repartition_rem_index[col])].sort_values(by = col, ascending = False)
    #         if erreurs_list.shape[0] : probas_erreurs[col] = erreurs_list.iloc[0][col]
    #         else: 
    #             erreurs_list[col] = 0.
    #             probas_erreurs[col] = 0.
    #         added[col] = list(erreurs_list.index[:samples_nb])

    #     print("probas_erreurs:",probas_erreurs)

    #     for col in added:
    #         samples_nb = batch_size
    #         samples_nb= int(min(batch_size,samples_nb))
    #         repartition_rem_index[col] = repartition_rem_index[col].difference(added[col][:samples_nb])
    #         repartition_rem_count[col] = len(repartition_rem_index[col])
    #         add_indices = add_indices + list(added[col][:samples_nb])
    #         cols_ordinal[col] += len(added[col])
    #     return add_indices


    score_fun_ = kwargs.get('score_fun',classifier_score_fun)
    add_indices_fun_ = kwargs.get('add_indices_fun',add_indices_fun)

    y,fy = pd.DataFrame(),pd.DataFrame()
    if 'y' not in kwargs: kwargs['y']=y
    cols_ordinal = get_occurence_count(kwargs["fz"])
    if not kwargs['y'].shape[0]:
        for col in cols_ordinal:
            new_set = list(kwargs["fz"].loc[kwargs["fz"][col] == 1.].index)
            random.Random(random_state).shuffle(new_set)
            keep_indices.update(set(new_set[:min(batch_size,2)]))
    else:
        y,fy = kwargs['y'],kwargs['fy']


    def update(keep_indices,kwargs):
        kwargs['y'] = pd.concat([y,kwargs['z'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
        kwargs['fy'] = pd.concat([fy,kwargs['fz'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
        kwargs['x'],kwargs['fx']=kwargs['y'],kwargs['fy']
        kwargs['f_z'] = proba_predictor_(**kwargs)
        return kwargs
    # def update(keep_indices,kwargs):
    #     kwargs['x'],kwargs['fx']=kwargs['z'],kwargs['fz']
    #     kwargs['y'] = pd.concat([y,kwargs['z'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
    #     kwargs['fy'] = pd.concat([fy,kwargs['fz'].iloc[list(keep_indices)]],axis=0).reset_index(drop = True)
    #     kwargs['f_z'] = proba_predictor_(**kwargs)
    #     return kwargs
    kwargs = update(keep_indices,kwargs)

    iteration = 0
    add_indices = [0]
    best_score = float("inf")
    best_indices = {}
    while kwargs['y'].shape[0] < max_number and len(add_indices) > 0:
        add_indices = add_indices_fun_(**kwargs)
        keep_indices = keep_indices | set(add_indices)
        kwargs = update(keep_indices,kwargs)
        score_ = score_fun_(**kwargs)
        iteration = iteration+1
        print("iteration: ", iteration, "training set size:", kwargs['y'].shape[0]," - score: ",score_," - best_score: ",best_score)
       
    kwargs = update(keep_indices,kwargs)
    # kwargs = update(best_indices,kwargs)
    score_ = score_fun_(**kwargs)
    print("final: training set size:", kwargs['x'].shape[0]," - score: ",score_)
    return kwargs


############################ classifiers ###############################################
class codpyprClassifier(codpyprRegressor,add_confusion_matrix): #label_codpy_predictor
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'accuracy_score_function' not in kwargs: 
            from sklearn import metrics
            self.accuracy_score_function = metrics.accuracy_score

    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            get_proba = kwargs.get('get_proba',False)
            kwargs['set_codpy_kernel'] = kwargs.get('set_codpy_kernel',self.set_kernel)
            kwargs['rescale'] = kwargs.get('rescale',False)
            fx = unity_partition(self.fx)
            f_z = op.projection(x = self.x,y = self.y,z = self.z, fx = fx,**kwargs)
            if get_proba:
                self.f_z = f_z
            else:
                self.f_z = softmaxindice(f_z)
    def id(self,name = ""):
        return "codpy lab pred"
    def copy(self):
        return self.copy_data(codpyprClassifier())


class codpyexClassifier(codpyprClassifier):
    def copy(self):
        return self.copy_data(codpyexClassifier())
    def predictor(self,**kwargs):
        if (self.D*self.Nx*self.Ny*self.Nz ):
            get_proba = kwargs.get('get_proba',False)
            fx = unity_partition(self.fx)
            f_z = op.projection(x = self.x,y = self.x,z = self.z, fx = fx,set_codpy_kernel=self.set_kernel,rescale = True)
            if get_proba:
                self.f_z = f_z
            else:
                self.f_z = softmaxindice(f_z)
    def id(self,name = ""):
        return "codpy"
        # return "codpy lab extra"

################### Semi_supervised ######################################""
class graphical_cluster_utilities:
    def plot_clusters(predictor ,ax, **kwargs):
        import seaborn as sns
        fun = get_representation_function(**kwargs)

        xlabel = kwargs.get('xlabel',"pca1")
        ylabel = kwargs.get('ylabel',"pca2")
        cluster_label = kwargs.get('cluster_label',"cluster:")


        x = np.asarray(predictor.z)
        fx = np.asarray(predictor.f_z)
        centers = np.asarray(predictor.y)
        ny = len(centers)
        if (len(x)*len(fx)*len(centers)):
            colors = plt.cm.Spectral(fx / ny)
            x,y = fun(x)
            num = len(x)
            df = pd.DataFrame({'x': x, 'y':y, 'label':fx})
            groups = df.groupby(fx)
            for name, group in groups:
                ax.plot(group.x, group.y, marker='o', linestyle='', ms=50/np.sqrt(num), mec='none')
                ax.set_aspect('auto')
                ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
                ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
            if len(centers):
                c1,c2 = fun(centers)
                ax.scatter(c1, c2,marker='o', c="black", alpha=1, s=200)
                for n in range(0,len(c1)):
                    a1,a2 = c1[n],c2[n]
                    ax.scatter(a1, a2, marker='$%d$' % n, alpha=1, s=50)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.title.set_text(cluster_label + str(ny))


class standard_cluster_predictor(data_predictor,graphical_cluster_utilities):
    import time
    score_silhouette, score_calinski_harabasz, homogeneity_test, inertia,  discrepancy = np.NaN,np.NaN,np.NaN,np.NaN,np.NaN
    estimator = None

    def copy_data(self,out):
        super().copy_data(out)
        out.score_silhouette,out.score_calinski_harabasz,out.homogeneity_test,out.inertia= self.score_silhouette,self.score_calinski_harabasz,self.homogeneity_test,self.inertia
        return out

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'accuracy_score_function' not in kwargs:
            from sklearn import metrics
            self.accuracy_score_function = metrics.accuracy_score

    def validator(self,**kwargs):
        super().validator(**kwargs)
        from sklearn import metrics
        from sklearn.metrics import silhouette_samples, silhouette_score
        if len(self.z)*len(self.f_z):
            try:
                if self.is_validator_compute(field ='score_silhouette',**kwargs): self.score_silhouette = silhouette_score(self.z, self.f_z)
                if self.is_validator_compute(field ='score_calinski_harabasz',**kwargs): self.score_calinski_harabasz = metrics.calinski_harabasz_score(self.z, self.f_z)
            except:
                pass
        if len(self.fz)*len(self.f_z):
            if self.is_validator_compute(field ='homogeneity_test',**kwargs): self.homogeneity_test = metrics.homogeneity_score(self.fz, self.f_z)
        if (self.estimator):
            if self.is_validator_compute(field ='inertia',**kwargs): self.inertia = self.estimator.inertia_
        else:
            if self.is_validator_compute(field ='inertia',**kwargs):
                from sklearn.cluster import KMeans
                self.inertia = KMeans(n_clusters=self.Ny).fit(self.x).inertia_
        pass


class codpyClusterClassifier(standard_cluster_predictor,add_confusion_matrix):
    def copy(self):
        return self.copy_data(codpyClusterClassifier())
    def predictor(self,**kwargs):
        kwargs['set_codpy_kernel'] = kwargs.get("set_codpy_kernel",self.set_kernel)
        kwargs['rescale'] = kwargs.get("rescale",True)
        kwargs['x'] = kwargs.get("x",self.x)
        kwargs['z'] = kwargs.get("z",self.z)
        self.y = alg.sharp_discrepancy( **kwargs)
        kwargs['y'] = self.y
        fx = alg.distance_labelling(**kwargs)
        if (self.x is self.z):
            self.f_z = fx
        else: 
            up = unity_partition(fx = fx)
            debug = op.projection(fx = up,**kwargs)
            self.f_z = softmaxindice(debug,axis=1)
        if len(self.fx) : self.f_z = remap(self.f_z,get_surjective_dictionnary(fx,self.fx))
        pass
    def id(self,name = ""):
        return "codpy"

class codpyClusterPredictor(standard_cluster_predictor,add_confusion_matrix):
    def copy(self):
        return self.copy_data(codpyClusterPredictor())
    def predictor(self,**kwargs):
        self.y = alg.sharp_discrepancy(x = self.x, y = self.y,**kwargs)
        fx = alg.distance_labelling(**{**kwargs,**{'x':self.x,'y':self.y}})
        if (self.x is self.z):
            self.f_z = fx
        else: 
            up = unity_partition(fx = fx)
            debug = op.projection(x = self.x,y = self.y,z = self.z,fx = up,**kwargs)
            self.f_z = softmaxindice(debug,axis=1)
        if len(self.fx) : self.f_z = remap(self.f_z,get_surjective_dictionnary(fx,self.fx))
        pass

    def id(self,name = ""):
        return "codpy"        


def test_predictor(my_fun):

    D,Nx,Ny,Nz=2,2000,2000,2000
    data_random_generator_ = data_random_generator(fun = my_fun,types=["cart","sto","cart"])
    x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
    fz_extrapolated = op.extrapolation(x,fx,x,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
    multi_plot([(x,fx),(x,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
    fz_extrapolated = op.extrapolation(x,fx,y,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
    multi_plot([(y,fy),(y,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")
    fz_extrapolated = op.extrapolation(x,fx,z,set_codpy_kernel = kernel_setters.set_gaussian_kernel(0,1e-8,map_setters.set_standard_min_map),rescale = True)    
    multi_plot([(x,fx),(z,fz_extrapolated)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")

def test_nablaT_nabla(my_fun,nabla_my_fun,set_kernel):
    D,Nx,Ny,Nz=2,2000,2000,2000
    data_random_generator_ = data_random_generator(fun = my_fun,nabla_fun = nabla_my_fun, types=["cart","cart","cart"])
    x,y,z,fx,fy,fz,nabla_fx,nabla_fz,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    f1 = op.nablaT(x,y,z,op.nabla(x,y,z,fx,set_codpy_kernel = set_kernel,rescale = True))
    f2 = op.nablaT_nabla(x,y,fx)
    multi_plot([(x,f1),(x,f2)],plot_trisurf,projection='3d')


def test_withgenerator(my_fun):
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel,0,1e-8,map_setters.set_standard_min_map)
    D,Nx,Ny,Nz=2,1000,1000,1000
    scenarios_list = [ (D, 100*i, 100*i ,100*i ) for i in np.arange(1,5,1)]
    if D!=1: projection="3d"
    else: projection=""

    data_random_generator_ = data_random_generator(fun = my_fun,types=["cart","sto","cart"])
    x,y,z,Nx,Ny,Nz =  data_random_generator_.get_raw_data(D=1,Nx=5,Ny=1000,Nz=0)

    x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    
    multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection=projection)

    scenario_generator_ = scenario_generator()
    scenario_generator_.run_scenarios(scenarios_list,data_random_generator_,codpyexRegressor(set_kernel = set_kernel),
data_accumulator())
    list_results = [(s.z,s.f_z) for s in scenario_generator_.accumulator.predictors]
    multi_plot(list_results,plot1D,mp_max_items = 2)





if __name__ == "__main__":

    set_kernel = kernel_setters.kernel_helper(
    kernel_setters.set_tensornorm_kernel, 0,1e-8 ,map_setters.set_unitcube_map)
    test_predictor(my_fun)
    test_nablaT_nabla(my_fun,nabla_my_fun,set_kernel)
