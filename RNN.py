import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import pickle
import sys
from lasagne.layers import *
import DataUtility as du

class RNN_GRU:
    f_predictions = T.fvector('func_predictions')
    f_labels = T.ivector('func_labels')
    ACE_cost = T.nnet.categorical_crossentropy(f_predictions, f_labels).mean()
    AverageCrossEntropy = theano.function([f_predictions, f_labels], ACE_cost,
                                          allow_input_downcast=True)

    def __init__(self):
        self.training = []
        self.cov_mean = []
        self.cov_stdev = []

        self.num_units = 200
        self.num_input = 5
        self.num_output = 3
        self.step_size = 0.01
        self.batch_size = 10
        self.num_folds = 10
        self.num_epochs = 20
        self.dropout1 = 0.6
        self.dropout2 = 0.6

        self.eval_metrics = ['NA', 'NA', 'NA', 'NA', 'NA']

        self.l_in = None
        self.l_drop1 = None
        self.l_GRU = None
        self.l_reshape_GRU = None
        self.l_relu = None
        self.l_drop2 = None
        self.l_output_GRU = None

        self.target_values = T.matrix('target_output')
        self.cost_vector = T.dvector('cost_list')
        self.num_elements = T.dscalar('batch_size')

        self.network_output_GRU = None
        self.network_reshape_GRU = None
        self.cost_GRU = None
        self.all_params_GRU = None
        self.updates_adhoc = None
        self.compute_cost_GRU = None
        self.pred_GRU = None
        self.rshp_GRU = None
        self.train_GRU_no_update = None
        self.train_GRU_update = None

        self.train_validation_GRU = [['GRU Training Error'], ['GRU Validation Error']]

        self.isBuilt = False
        self.isInitialized = False

    def set_hyperparams(self,num_recurrent, step_size=.01, dropout1=0.0,
                        dropout2=0.0, batch_size=10,num_epochs=20,num_folds=10):
        self.num_units = num_recurrent
        self.step_size = step_size
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.dropout1=dropout1
        self.dropout2=dropout2
        self.isBuilt = False

    def set_training_params(self, batch_size, num_epochs, num_folds=10):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_folds = num_folds

    def save_parameters(self, filename_no_ext):
        all_params = lasagne.layers.get_all_params(self.l_output_GRU)
        all_param_values = [p.get_value() for p in all_params]
        np.save(filename_no_ext+'.npy', np.array(all_param_values))

    def load_from_file(self, filename_no_ext):
        self.build_network()
        all_param_values = np.load(filename_no_ext+'.npy')
        all_params = lasagne.layers.get_all_params(self.l_output_GRU)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    def build_network(self):
        print("\nBuilding Network...")

        if not self.isInitialized:
            # Recurrent network structure
            self.l_in = lasagne.layers.InputLayer(shape=(None, None, self.num_input))
            self.l_drop1 = lasagne.layers.DropoutLayer(self.l_in, self.dropout1)
            self.l_GRU = lasagne.layers.GRULayer(self.l_drop1, self.num_units, precompute_input=True,grad_clipping=100.)
            self.l_reshape_GRU = lasagne.layers.ReshapeLayer(self.l_GRU, shape=(-1, self.num_units))
            self.l_relu = lasagne.layers.RandomizedRectifierLayer(self.l_reshape_GRU)
            self.l_drop2 = lasagne.layers.DropoutLayer(self.l_relu,self.dropout2)
            self.l_output_GRU = lasagne.layers.DenseLayer(self.l_drop2, num_units=self.num_output,
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=lasagne.nonlinearities.identity)
            self.isInitialized = True

        # theano variables for output
        self.network_output_GRU = lasagne.layers.get_output(self.l_output_GRU,deterministic=False)
        self.network_output_GRU_test = lasagne.layers.get_output(self.l_output_GRU, deterministic=True)
        self.network_reshape_GRU = lasagne.layers.get_output(self.l_reshape_GRU)

        # use cross-entropy for cost - average across the batch
        self.cost_GRU = T.nnet.categorical_crossentropy(T.exp(self.network_output_GRU) /
                                                        (T.exp(self.network_output_GRU).sum(1, keepdims=True)),
                                                        self.target_values).mean()

        # theano variable for network parameters for updating
        self.all_params_GRU = lasagne.layers.get_all_params(self.l_output_GRU, trainable=True)

        #print("Computing updates...")
        # update the network given a list of batch costs (for batches of sequences)
        self.updates_adhoc = lasagne.updates.adagrad((T.sum(self.cost_vector) + self.cost_GRU) / self.num_elements,
                                                     self.all_params_GRU,
                                                     self.step_size)
        #print("Compiling functions...")
        # get the GRU cost given inputs and labels
        self.compute_cost_GRU = theano.function([self.l_in.input_var, self.target_values],
                                                self.cost_GRU, allow_input_downcast=True)
        # get the prediction vector of the network given some inputs
        self.pred_GRU_train = theano.function([self.l_in.input_var], T.exp(self.network_output_GRU) /
                                        (T.exp(self.network_output_GRU).sum(1, keepdims=True)),
                                        allow_input_downcast=True)

        self.pred_GRU = theano.function([self.l_in.input_var], T.exp(self.network_output_GRU_test) /
                                        (T.exp(self.network_output_GRU_test).sum(1, keepdims=True)),
                                        allow_input_downcast=True)

        self.rshp_GRU = theano.function([self.l_in.input_var], self.network_reshape_GRU, allow_input_downcast=True)

        # get the cost of the network without updating parameters (for batch updating)
        self.train_GRU_no_update = theano.function([self.l_in.input_var, self.target_values],
                                                   self.cost_GRU, allow_input_downcast=True)
        # get the cost of the network and update parameters based on previous costs (for batch updating)
        self.train_GRU_update = theano.function([self.l_in.input_var, self.target_values,
                                                 self.cost_vector, self.num_elements],
                                                (T.sum(self.cost_vector) + self.cost_GRU) / self.num_elements,
                                                updates=self.updates_adhoc, allow_input_downcast=True)
        self.isBuilt = True
        print "Network Params:", count_params(self.l_output_GRU)

    def train(self, training, training_labels):

        self.num_input = du.len_deepest(training)
        self.num_output = du.len_deepest(training_labels)

        if not self.isBuilt:
            self.build_network()

        t_tr = du.transpose(flatten_sequence(training))
        self.cov_mean = []
        self.cov_stdev = []

        for a in range(0,len(t_tr)):
            mn = np.nanmean(t_tr[a])
            sd = np.nanstd(t_tr[a])
            self.cov_mean.append(mn)
            self.cov_stdev.append(sd)

        training_samples = []

        import math
        for a in range(0,len(training)):
            sample = []
            for e in range(0,len(training[a])):
                covariates = []
                for i in range(0,len(training[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (training[a][e][i]-self.cov_mean[i])/self.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covariates.append(cov)
                sample.append(covariates)
            training_samples.append(sample)

        label_train = training_labels
        # introduce cross-validation
        from sklearn.cross_validation import StratifiedKFold

        strat_label = []
        for i in range(0, len(training_samples)):
            if label_train[i][len(label_train[i]) - 1][1] == 1:
                strat_label.append(0)
            else:
                strat_label.append(1)

        skf = StratifiedKFold(strat_label, n_folds=self.num_folds)

        print"Number of Folds:", len(skf)

        print "Training Samples (Sequences):", len(training_samples)

        print("\nTraining GRU...")
        print "{:<9}".format("  Epoch"), \
            "{:<9}".format("  Train"), \
            "{:<9}".format("  Valid"), \
            "{:<9}".format("  Time"), \
            "\n======================================"
        start_time = time.clock()
        gru_train_err = []
        gru_val_err = []
        previous = 0
        # for each epoch...
        for e in range(0, self.num_epochs):
            epoch_time = time.clock()
            epoch = 0
            eval = 0
            n_train = 0
            n_test = 0

            # train and validate
            for ktrain, ktest in skf:
                for i in range(0, len(ktrain), self.batch_size):
                    batch_cost = []
                    # get the cost of each sequence in the batch
                    for j in range(i, min(len(ktrain) - 1, i + self.batch_size - 1)):
                        # rx = rshp_GRU([training_samples[ktrain[j]]])
                        # print rx
                        # print np.array(rx).shape
                        cost_val = self.train_GRU_no_update([training_samples[ktrain[j]]], label_train[ktrain[j]])
                        if math.isnan(cost_val):
                            print "nan Value found: Rebuilding Network..."
                            print batch_cost
                            for ktr in training_samples[ktrain[j]]:
                                print ktr
                            print self.pred_GRU([training_samples[ktrain[j]]])
                            print "======="
                            par = get_all_param_values(self.l_output_GRU)

                            for p in par:
                                if p is list:
                                    for pe in p:
                                        if pe is list:
                                            for pr in pe:
                                                print pr
                                        else:
                                            print pe

                                else:
                                    print p
                            exit()

                        batch_cost.append(self.train_GRU_no_update([training_samples[ktrain[j]]], label_train[ktrain[j]]))

                    j = min(len(ktrain) - 1, i + self.batch_size - 1)

                    # the last sample of the batch updates the network with the cost values
                    cost_val = self.train_GRU_update([training_samples[ktrain[j]]], label_train[ktrain[j]],
                                                   batch_cost, self.batch_size)

                    if math.isnan(cost_val):
                        print "NaN Value found: Rebuilding Network..."
                        print len(batch_cost)
                        print batch_cost
                        for ktr in training_samples[ktrain[j]]:
                            print ktr
                        print self.pred_GRU([training_samples[ktrain[j]]])
                        print "======="
                        par = get_all_param_values(self.l_output_GRU)
                        for p in par:
                            print p
                        exit()

                    epoch += self.train_GRU_update([training_samples[ktrain[j]]], label_train[ktrain[j]],
                                                   batch_cost, self.batch_size)

                    n_train += 1

                for i in range(0, len(ktest)):
                    # get the validation error
                    eval += self.compute_cost_GRU([training_samples[ktest[i]]], label_train[ktest[i]])
                    n_test += 1

            gru_train_err.append(epoch / n_train)
            gru_val_err.append(eval / n_test)
            print "{:<9}".format("Epoch " + str(e + 1) + ":"), \
                "  {0:.4f}".format(epoch / n_train), \
                "   {0:.4f}".format(eval / n_test), \
                "   {0:.1f}s".format(time.clock() - epoch_time)

            if e == 0:
                previous = eval/n_test
            else:
                if eval/n_test - previous > 0.005:
                    break
                previous = eval/n_test
            if math.isnan(epoch / n_train):
                print "NaN Value found: Rebuilding Network..."
                self.isBuilt = False
                self.isInitialized = False

        print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

        self.train_validation_GRU = [['GRU Training Error'], ['GRU Validation Error']]
        for i in range(0, len(gru_train_err)):
            self.train_validation_GRU[0].append(str(gru_train_err[i]))
        for i in range(0, len(gru_val_err)):
            self.train_validation_GRU[1].append(str(gru_val_err[i]))

    def predict(self, test):

        if len(self.cov_mean) == 0 or len(self.cov_stdev) == 0:
            print "Scaling factors have not been generated: calculating using test sample"
            t_tr = du.transpose(flatten_sequence(test))
            self.cov_mean = []
            self.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                self.cov_mean.append(mn)
                self.cov_stdev.append(sd)

        test_samples = []

        import math
        for a in range(0, len(test)):
            sample = []
            for e in range(0, len(test[a])):
                covariates = []
                for i in range(0, len(test[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (test[a][e][i] - self.cov_mean[i]) / self.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covariates.append(cov)
                sample.append(covariates)
            test_samples.append(sample)

        predictions_GRU = []
        for i in range(0, len(test_samples)):
            # get the prediction and calculate cost
            prediction_GRU = self.pred_GRU([test_samples[i]])

            for j in range(0, len(prediction_GRU)):
                predictions_GRU.append(prediction_GRU[j].tolist())

        predictions_GRU = np.round(predictions_GRU, 3).tolist()

        return predictions_GRU

    def test(self, test, test_labels=None, label_names=None):
        if test_labels is None:
            return self.predict(test)

        if len(self.cov_mean) == 0 or len(self.cov_stdev) == 0:
            print "Scaling factors have not been generated: calculating using test sample"
            t_tr = du.transpose(flatten_sequence(test))
            self.cov_mean = []
            self.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                self.cov_mean.append(mn)
                self.cov_stdev.append(sd)

        test_samples = []

        import math
        for a in range(0, len(test)):
            sample = []
            for e in range(0, len(test[a])):
                covariates = []
                for i in range(0, len(test[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (test[a][e][i] - self.cov_mean[i]) / self.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covariates.append(cov)
                sample.append(covariates)
            test_samples.append(sample)

        label_test = test_labels
        print("\nTesting...")
        print "Test Samples:", len(test_samples)

        classes = []
        p_count = 0

        avg_class_err = []
        avg_err_GRU = []

        predictions_GRU = []
        for i in range(0, len(test_samples)):
            # get the prediction and calculate cost
            prediction_GRU = self.pred_GRU([test_samples[i]])
            avg_err_GRU.append(self.compute_cost_GRU([test_samples[i]], label_test[i]))

            for j in range(0, len(label_test[i])):
                p_count += 1

                classes.append(label_test[i][j].tolist())
                predictions_GRU.append(prediction_GRU[j].tolist())

        predictions_GRU = np.round(predictions_GRU, 3).tolist()

        actual = []
        pred_GRU = []
        cor_GRU = []

        # get the percent correct for the predictions
        # how often the prediction is right when it is made
        for i in range(0, len(predictions_GRU)):
            c = classes[i].index(max(classes[i]))
            actual.append(c)

            p_GRU = predictions_GRU[i].index(max(predictions_GRU[i]))
            pred_GRU.append(p_GRU)
            cor_GRU.append(int(c == p_GRU))

        # calculate a naive baseline using averages
        flattened_label = []
        for i in range(0, len(label_test)):
            for j in range(0, len(label_test[i])):
                flattened_label.append(label_test[i][j])
        flattened_label = np.array(flattened_label)
        avg_class_pred = np.mean(flattened_label,0)

        print "Predicting:", avg_class_pred, "for baseline*"
        for i in range(0, len(flattened_label)):
            res = RNN_GRU.AverageCrossEntropy(np.array(avg_class_pred), np.array(classes[i]))
            avg_class_err.append(res)
            # res = RNN_GRU.AverageCrossEntropy(np.array(predictions_GRU[i]), np.array(classes[i]))
            # avg_err_GRU.append(res)
        print "*This is calculated from the TEST labels"

        from sklearn.metrics import roc_auc_score,f1_score
        from skll.metrics import kappa

        kpa = []
        auc = []
        f1s = []
        apr = []
        t_pred = du.transpose(predictions_GRU)
        t_lab = du.transpose(flattened_label)

        for i in range(0,len(t_lab)):
            #if i == 0 or i == 3:
            #    t_pred[i] = du.normalize(t_pred[i],method='max')
            kpa.append(kappa(t_lab[i],t_pred[i]))
            apr.append(du.Aprime(t_lab[i],t_pred[i]))
            auc.append(roc_auc_score(t_lab[i],t_pred[i]))
            temp_p = [round(j) for j in t_pred[i]]
            if np.nanmax(temp_p)==0:
                f1s.append(0)
            else:
                f1s.append(f1_score(t_lab[i],temp_p))

        if label_names is None or len(label_names) != len(t_lab):
            label_names = []
            for i in range(0, len(t_lab)):
                label_names.append("Label " + str(i + 1))

        print_label_distribution(flattened_label, label_names)

        self.eval_metrics = [np.nanmean(avg_err_GRU),np.nanmean(auc),np.nanmean(kpa),
                             np.nanmean(f1s),np.nanmean(cor_GRU) * 100]

        print "\nBaseline Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_class_err))
        print "\nNetwork Performance:"
        print "Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_err_GRU))
        print "AUC:", "{0:.4f}".format(np.nanmean(auc))
        print "A':", "{0:.4f}".format(np.nanmean(apr))
        print "Kappa:", "{0:.4f}".format(np.nanmean(kpa))
        print "F1 Score:", "{0:.4f}".format(np.nanmean(f1s))
        print "Percent Correct:", "{0:.2f}%".format(np.nanmean(cor_GRU) * 100)

        print "\n{:<15}".format("  Label"), \
            "{:<9}".format("  AUC"), \
            "{:<9}".format("  A'"), \
            "{:<9}".format("  Kappa"), \
            "{:<9}".format("  F Stat"), \
            "\n=============================================="

        for i in range(0,len(t_lab)):
            print "{:<15}".format(label_names[i]), \
                "{:<9}".format("  {0:.4f}".format(auc[i])), \
                "{:<9}".format("  {0:.4f}".format(apr[i])), \
                "{:<9}".format("  {0:.4f}".format(kpa[i])), \
                "{:<9}".format("  {0:.4f}".format(f1s[i]))
        print "\n=============================================="

        print "Confusion Matrix:"
        actual = []
        predicted = []
        flattened_label = flattened_label.tolist()
        for i in range(0, len(predictions_GRU)):
            actual.append(flattened_label[i].index(max(flattened_label[i])))
            predicted.append(predictions_GRU[i].index(max(predictions_GRU[i])))

        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(actual, predicted)
        for cm in conf_mat:
            cm_row = "\t"
            for element in cm:
                cm_row += "{:<6}".format(element)
            print cm_row
        print "\n=============================================="

        return predictions_GRU

    def get_performance(self):
        return self.eval_metrics

#################################################################################################


def build_sequences(data,primary_column,secondary_column, covariate_columns, label_columns):
    pdata = []
    labels = []

    # if data is not null, build the dataset
    if data is not None:
        data = du.transpose(data)
        for i in range(0,len(covariate_columns)):
            for j in range(0,len(data[i])):
                data[covariate_columns[i]][j] = float(data[covariate_columns[i]][j])
            #data[covariate_columns[i]] = du.normalize(data[covariate_columns[i]],method="zscore")
        data = du.transpose(data)
        assignments = du.unique(du.transpose(data)[primary_column])

        for i in range(0,len(assignments)):
            a_set = du.select(data, assignments[i], '==', primary_column)
            users = du.unique(du.transpose(a_set)[secondary_column])

            for j in range(0,len(users)):
                u_set = du.select(a_set,users[j],'==',secondary_column)
                timesteps = []
                ts_labels = []

                for k in range(0,len(u_set)):
                    step = []
                    for m in range(0,len(covariate_columns)):
                        step.append(float(u_set[k][covariate_columns[m]]))

                    timesteps.append(np.array(step))

                    lbl = []
                    for m in range(0,len(label_columns)):
                        lbl.append(float(u_set[k][label_columns[m]]))

                    ts_labels.append(np.array(lbl))

                pdata.append(np.array(timesteps))
                labels.append(np.array(ts_labels))

        pdata = np.array(pdata)
        labels = np.array(labels)

        # save the dataset for easy loading
        np.save('timeseries_data.npy',pdata)
        np.save('timeseries_labels.npy',labels)
    else:
        pdata = np.load('timeseries_data.npy')
        labels = np.load('timeseries_labels.npy')

    data = []
    label = []
    for i in range(0,len(pdata)):
        if (len(pdata[i]) == 0):
            continue
        data.append(pdata[i])
        label.append(labels[i])

    return np.array(data),np.array(label)


def load_unlabeled_data(filename,primary_column,secondary_column,covariate_columns):
    # load from file or rebuild dataset
    load = 0

    data = None
    if load == 0:
        data, headers = du.loadCSVwithHeaders(filename)

        for i in range(0, len(headers)):
            print '{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i])
    else:
        print 'Skipping dataset loading - using cached data instead'

    print '\ntransforming data to time series...'
    pdata, labels = build_sequences(data, primary_column, secondary_column, covariate_columns, [1,2])

    print '\nDataset Info:'
    print 'number of samples:', len(pdata)
    print 'sequence length of first sample:', len(pdata[0])
    print 'input nodes: ', len(pdata[0][0])

    return pdata, labels


# loads the dataset
def load_data(filename,primary_column,secondary_column,covariate_columns,label_columns):
    # load from file or rebuild dataset
    load = 0

    data = None
    if load == 0:
        data,headers = du.loadCSVwithHeaders(filename)

        for i in range(0,len(headers)):
            print '{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i])
    else:
        print 'Skipping dataset loading - using cached data instead'

    print '\ntransforming data to time series...'
    pdata,labels = build_sequences(data,primary_column,secondary_column,covariate_columns,label_columns)

    print '\nDataset Info:'
    print 'number of samples:', len(pdata)
    print 'sequence length of first sample:', len(pdata[0])
    print 'input nodes: ', len(pdata[0][0])

    #print [i[0] for i in pdata[0]]
    #print [i[1] for i in labels[0]]

    pdata,labels = du.shuffle(pdata,labels)

    return pdata,labels


# adds redundancy for time-series labels
def add_representation(data,labels,label_column,duplicate=10,threshold=0.0):
    assert len(data) == len(labels)
    print "Adding Representation to label:",label_column
    ndata = []
    nlabel = []
    for i in range(0,len(data)):
        represent = 1

        if np.nanmean(labels[i],0)[label_column] > threshold:
            represent = duplicate

        for j in range(0,represent):
            ndata.append(data[i])
            nlabel.append(labels[i])

    ndata,nlabel = du.shuffle(ndata,nlabel)
    return np.array(ndata),np.array(nlabel)


# adds redundancy for labels
def add_representation_flattened(data, labels=None, label_column=0, duplicate=10):
    if labels is not None:
        assert len(data) == len(labels)

    print "Adding Representation to label:", label_column
    ndata = []
    nlabel = []
    for i in range(0,len(data)):
        represent = 1

        if labels is None:
            if data[i][label_column] == 1:
                represent = duplicate
        else:
            if labels[i][label_column] == 1:
                represent = duplicate

        for j in range(0,represent):
            ndata.append(data[i])
            if labels is not None:
                nlabel.append(labels[i])

    if labels is None:
        ndata = du.shuffle(ndata)
        return np.array(ndata)

    ndata,nlabel = du.shuffle(ndata,nlabel)
    return np.array(ndata),np.array(nlabel)


def flatten_sequence(sequence_data):
    # print "flattening sequence..."
    flattened = []

    for i in range(0,len(sequence_data)):
        for j in range(0,len(sequence_data[i])):
            row = []
            for k in range(0,len(sequence_data[i][j])):
                row.append(sequence_data[i][j][k])
            flattened.append(row)

    return flattened


def print_label_distribution(flat_labels,label_names=None):
    print "\nLabel Distribution:"

    labels = du.transpose(flat_labels)

    if label_names is not None:
        assert len(label_names) == len(labels)
    else:
        label_names = []
        for i in range(0, len(labels)):
            label_names[i] = "Label_" + str(i)

    for i in range(0,len(labels)):
        print "   " + label_names[i] + ":", "{:<6}".format(np.nansum(np.array(labels[i]))), \
            "({0:.0f}%)".format((float(np.nansum(np.array(labels[i])))/len(labels[i]))*100)


def saveInstance(model,filename):
    pickle.dump(model, open(filename,"wb"))

def loadInstance(filename):
    model = pickle.load(open(filename, "rb" ))
    return model

def train_and_test(data_filename):
    training = []
    test = []
    label_train = []
    label_test = []

    data, labels = load_data(data_filename, 0, 1, range(3, 95), range(95, 99))

    training, test, label_train, label_test = du.split_training_test(data, labels, training_size=.8)

    t_labels = du.transpose(flatten_sequence(labels))

    import math
    rep_confused = int(math.floor((len(t_labels[0]) / np.nansum(t_labels[0])) + 1))
    rep_bored = int(math.floor((len(t_labels[2]) / np.nansum(t_labels[2])) + 1))
    rep_frustrated = int(math.floor((len(t_labels[3]) / np.nansum(t_labels[3])) + 1))

    # add redundancy
    rep_training, rep_label_train = add_representation(training, label_train, 0, rep_confused, 0.1)
    rep_training, rep_label_train = add_representation(rep_training, rep_label_train, 2, rep_bored, 0.1)
    rep_training, rep_label_train = add_representation(rep_training, rep_label_train, 3, rep_frustrated, 0.1)

    rep_training, rep_label_train = du.sample(rep_training, rep_label_train, p=1, n=len(training))


    nodes = 250
    batches = 1
    re_epoch = 5
    epoch = 10

    GNET = RNN_GRU()
    GNET.set_hyperparams(nodes, batch_size=2, num_folds=10, num_epochs=20, step_size=0.001,
                         dropout1=0.2, dropout2=0.0)

    if not re_epoch == 0:
        print_label_distribution(flatten_sequence(rep_label_train),
                                 ["Confused", "Concentrating", "Bored", "Frustrated"])
        GNET.set_training_params(batches, re_epoch)
        GNET.train(rep_training, rep_label_train)
        # pred = GNET.test(test, label_test, ["Confused", "Concentrating", "Bored", "Frustrated"])

    if not epoch == 0:
        print_label_distribution(flatten_sequence(labels), ["Confused", "Concentrating", "Bored", "Frustrated"])
        GNET.set_training_params(batches, epoch)
        GNET.train(training, label_train)
    sys.setrecursionlimit(10000)
    saveInstance(GNET,"GNET.pickle")
    GNET.test(test, label_test, ["Confused", "Concentrating", "Bored", "Frustrated"])
    GNET2 = loadInstance("GNET.pickle")
    GNET2.test(test, label_test, ["Confused", "Concentrating", "Bored", "Frustrated"])

##########################################################################################################


if __name__ == "__main__":

    train_and_test("resources/affect_ground_truth.csv")