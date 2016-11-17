import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import sys
from lasagne.layers import *
import DataUtility as du
import pickleUtility as pu


class RNN:
    f_predictions = T.fvector('func_predictions')
    f_labels = T.ivector('func_labels')
    ACE_cost = T.nnet.categorical_crossentropy(f_predictions, f_labels).mean()
    AverageCrossEntropy = theano.function([f_predictions, f_labels], ACE_cost,
                                          allow_input_downcast=True)

    @staticmethod
    def build_sequences(data, primary_column, secondary_column, covariate_columns, label_columns):
        pdata = []
        labels = []
        primary_group = []

        # if data is not null, build the dataset
        if data is not None:
            data = du.transpose(data)
            for i in range(0, len(covariate_columns)):
                for j in range(0, len(data[i])):
                    data[covariate_columns[i]][j] = float(data[covariate_columns[i]][j])
                    # data[covariate_columns[i]] = du.normalize(data[covariate_columns[i]],method="zscore")
            data = du.transpose(data)
            primary = du.unique(du.transpose(data)[primary_column])

            for i in range(0, len(primary)):
                p_set = du.select(data, primary[i], '==', primary_column)
                secondary = du.unique(du.transpose(p_set)[secondary_column])

                for j in range(0, len(secondary)):
                    s_set = du.select(p_set, secondary[j], '==', secondary_column)
                    timesteps = []
                    ts_labels = []

                    for k in range(0, len(s_set)):
                        step = []
                        for m in range(0, len(covariate_columns)):
                            step.append(float(s_set[k][covariate_columns[m]]))

                        timesteps.append(np.array(step))

                        lbl = []
                        for m in range(0, len(label_columns)):
                            lbl.append(float(s_set[k][label_columns[m]]))

                        ts_labels.append(np.array(lbl))

                    pdata.append(np.array(timesteps))
                    labels.append(np.array(ts_labels))
                    primary_group.append(primary[i])

            pdata = np.array(pdata)
            labels = np.array(labels)

            # save the dataset for easy loading
            np.save('timeseries_data.npy', pdata)
            np.save('timeseries_labels.npy', labels)
            np.save('timeseries_groups.npy', primary_group)
        else:
            pdata = np.load('timeseries_data.npy')
            labels = np.load('timeseries_labels.npy')
            primary_group = np.load('timeseries_groups.npy')

        data = []
        label = []
        for i in range(0, len(pdata)):
            if len(pdata[i]) == 0:
                continue
            data.append(pdata[i])
            label.append(labels[i])

        return np.array(data), np.array(label), primary_group

    @staticmethod
    def load_unlabeled_data(filename, primary_column, secondary_column, covariate_columns, load_from_file=False):
        # load from file or rebuild dataset
        load = load_from_file

        data = None
        if not load:
            data, headers = du.loadCSVwithHeaders(filename)

            for i in range(0, len(headers)):
                print '{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i])
        else:
            print 'Skipping dataset loading - using cached data instead'

        print '\ntransforming data to time series...'
        pdata, labels, grouping = RNN.build_sequences(data, primary_column, secondary_column, covariate_columns, [1, 2])

        print '\nDataset Info:'
        print 'number of samples:', len(pdata)
        print 'sequence length of first sample:', len(pdata[0])
        print 'input nodes: ', len(pdata[0][0])

        return pdata, labels, grouping

    @staticmethod
    def load_data(filename, primary_column, secondary_column, covariate_columns, label_columns, load_from_file=False):
        # load from file or rebuild dataset
        load = load_from_file

        data = None
        if not load:
            data, headers = du.loadCSVwithHeaders(filename)

            for i in range(0, len(headers)):
                print '{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i])
        else:
            print 'Skipping dataset loading - using cached data instead'

        print '\ntransforming data to time series...'
        pdata, labels, grouping = RNN.build_sequences(data, primary_column, secondary_column, covariate_columns,
                                                      label_columns)

        print '\nDataset Info:'
        print 'number of samples:', len(pdata)
        print 'sequence length of first sample:', len(pdata[0])
        print 'input nodes: ', len(pdata[0][0])

        return pdata, labels, grouping

    @staticmethod
    def add_representation(data, labels, label_column, duplicate=10, threshold=0.0):
        assert len(data) == len(labels)
        # print "Adding Representation to label:",label_column
        ndata = []
        nlabel = []
        for i in range(0, len(data)):
            represent = 1

            if np.nanmean(labels[i], 0)[label_column] > threshold:
                represent = duplicate

            for j in range(0, represent):
                ndata.append(data[i])
                nlabel.append(labels[i])

        ndata, nlabel = du.shuffle(ndata, nlabel)
        return np.array(ndata), np.array(nlabel)

    @staticmethod
    def flatten_sequence(sequence_data):
        # print "flattening sequence..."
        flattened = []

        for i in range(0, len(sequence_data)):
            for j in range(0, len(sequence_data[i])):
                row = []
                for k in range(0, len(sequence_data[i][j])):
                    row.append(sequence_data[i][j][k])
                flattened.append(row)

        return flattened

    @staticmethod
    def get_label_distribution(labels):
        flat_labels = RNN.flatten_sequence(labels)

        labels = du.transpose(flat_labels)

        dist = []
        for i in range(0, len(labels)):
            dist.append((float(np.nansum(np.array(labels[i]))) / len(labels[i])))
        return dist

    @staticmethod
    def print_label_distribution(labels, label_names=None):
        print "\nLabel Distribution:"

        flat_labels = RNN.flatten_sequence(labels)
        labels = du.transpose(flat_labels)

        if label_names is not None:
            assert len(label_names) == len(labels)
        else:
            label_names = []
            for i in range(0, len(labels)):
                label_names[i] = "Label_" + str(i)

        for i in range(0, len(labels)):
            print "   " + label_names[i] + ":", "{:<6}".format(np.nansum(np.array(labels[i]))), \
                "({0:.0f}%)".format((float(np.nansum(np.array(labels[i]))) / len(labels[i])) * 100)

    def __init__(self, variant="RNN"):
        self.training = []
        self.cov_mean = []
        self.cov_stdev = []

        self.variant = variant
        if self.variant not in ["GRU","gru","LSTM","lstm","RNN","rnn"]:
            print "Invalid variant \"" + variant + "\" - defaulting to traditional RNN"
            self.variant = "RNN"

        self.num_units = 200
        self.num_input = 5
        self.num_output = 3
        self.step_size = 0.01
        self.batch_size = 10
        self.num_folds = 2
        self.num_epochs = 20
        self.dropout1 = 0.6
        self.dropout2 = 0.6

        self.min_preds = [1,1,1,1]
        self.min_preds = [0,0,0,0]
        self.avg_preds = [0.5,0.5,0.5,0.5]

        self.eval_metrics = ['NA', 'NA', 'NA', 'NA', 'NA']

        self.l_in = None
        self.l_drop1 = None
        self.l_Recurrent = None
        self.l_reshape_RNN = None
        self.l_relu = None
        self.l_drop2 = None
        self.l_output_RNN = None

        self.target_values = T.matrix('target_output')
        self.cost_vector = T.dvector('cost_list')
        self.num_elements = T.dscalar('batch_size')

        self.network_output_RNN = None
        self.network_reshape_RNN = None
        self.cost_RNN = None
        self.all_params_RNN = None
        self.updates_adhoc = None
        self.compute_cost_RNN = None
        self.pred_RNN = None
        self.rshp_RNN = None
        self.train_RNN_no_update = None
        self.train_RNN_update = None

        self.train_validation_RNN = [['RNN Training Error'], ['RNN Validation Error']]

        self.isBuilt = False
        self.isInitialized = False
        self.balance = False
        self.scale_output = False
        self.majorityclass = 0

    def set_hyperparams(self,num_recurrent, step_size=.01, dropout1=0.0,
                        dropout2=0.0, batch_size=10,num_epochs=20,num_folds=2):
        self.num_units = num_recurrent
        self.step_size = step_size
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.dropout1=dropout1
        self.dropout2=dropout2
        self.isBuilt = False

    def set_training_params(self, batch_size, num_epochs, balance=False, scale_output=False, num_folds=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if num_folds is not None:
            self.num_folds = num_folds
        self.balance = balance
        self.scale_output = scale_output

    def save_parameters(self, filename_no_ext):
        all_params = lasagne.layers.get_all_params(self.l_output_RNN)
        all_param_values = [p.get_value() for p in all_params]
        np.save(filename_no_ext+'.npy', np.array(all_param_values))

    def load_from_file(self, filename_no_ext):
        self.build_network()
        all_param_values = np.load(filename_no_ext+'.npy')
        all_params = lasagne.layers.get_all_params(self.l_output_RNN)
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    def build_network(self):
        print("\nBuilding Network...")

        if not self.isInitialized:
            # Recurrent network structure
            self.l_in = lasagne.layers.InputLayer(shape=(None, None, self.num_input))
            self.l_drop1 = lasagne.layers.DropoutLayer(self.l_in, self.dropout1)

            if self.variant == "GRU" or self.variant == "gru":
                self.l_Recurrent = lasagne.layers.GRULayer(self.l_drop1, self.num_units, precompute_input=True,
                                                           grad_clipping=100.)
            elif self.variant == "LSTM" or self.variant == "lstm":
                self.l_Recurrent = lasagne.layers.LSTMLayer(self.l_drop1, self.num_units, precompute_input=True,
                                                           grad_clipping=100.)
            else:
                self.l_Recurrent = lasagne.layers.RecurrentLayer(self.l_drop1, self.num_units, precompute_input=True,
                                                           grad_clipping=100.)
            self.l_reshape_RNN = lasagne.layers.ReshapeLayer(self.l_Recurrent, shape=(-1, self.num_units))
            self.l_relu = lasagne.layers.RandomizedRectifierLayer(self.l_reshape_RNN)
            self.l_drop2 = lasagne.layers.DropoutLayer(self.l_relu,self.dropout2)
            self.l_output_RNN = lasagne.layers.DenseLayer(self.l_drop2, num_units=self.num_output,
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=lasagne.nonlinearities.identity)
            self.isInitialized = True

        # theano variables for output
        self.network_output_RNN = lasagne.layers.get_output(self.l_output_RNN,deterministic=False)
        self.network_output_RNN_test = lasagne.layers.get_output(self.l_output_RNN, deterministic=True)
        self.network_reshape_RNN = lasagne.layers.get_output(self.l_reshape_RNN)

        # use cross-entropy for cost - average across the batch
        self.cost_RNN = T.nnet.categorical_crossentropy(T.exp(self.network_output_RNN) /
                                                        (T.exp(self.network_output_RNN).sum(1, keepdims=True)),
                                                        self.target_values).mean()

        # theano variable for network parameters for updating
        self.all_params_RNN = lasagne.layers.get_all_params(self.l_output_RNN, trainable=True)

        #print("Computing updates...")
        # update the network given a list of batch costs (for batches of sequences)
        self.updates_adhoc = lasagne.updates.adagrad((T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                     self.all_params_RNN,
                                                     self.step_size)
        #print("Compiling functions...")
        # get the RNN cost given inputs and labels
        self.compute_cost_RNN = theano.function([self.l_in.input_var, self.target_values],
                                                self.cost_RNN, allow_input_downcast=True)
        # get the prediction vector of the network given some inputs
        self.pred_RNN_train = theano.function([self.l_in.input_var], T.exp(self.network_output_RNN) /
                                        (T.exp(self.network_output_RNN).sum(1, keepdims=True)),
                                        allow_input_downcast=True)

        self.pred_RNN = theano.function([self.l_in.input_var], T.exp(self.network_output_RNN_test) /
                                        (T.exp(self.network_output_RNN_test).sum(1, keepdims=True)),
                                        allow_input_downcast=True)

        self.rshp_RNN = theano.function([self.l_in.input_var], self.network_reshape_RNN, allow_input_downcast=True)

        # get the cost of the network without updating parameters (for batch updating)
        self.train_RNN_no_update = theano.function([self.l_in.input_var, self.target_values],
                                                   self.cost_RNN, allow_input_downcast=True)
        # get the cost of the network and update parameters based on previous costs (for batch updating)
        self.train_RNN_update = theano.function([self.l_in.input_var, self.target_values,
                                                 self.cost_vector, self.num_elements],
                                                (T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                updates=self.updates_adhoc, allow_input_downcast=True)
        self.isBuilt = True
        print "Network Params:", count_params(self.l_output_RNN)

    def train(self, training, training_labels):

        self.num_input = du.len_deepest(training)
        self.num_output = du.len_deepest(training_labels)

        if not self.isBuilt:
            self.build_network()

        t_tr = du.transpose(RNN.flatten_sequence(training))
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

        label_distribution = RNN.get_label_distribution(label_train)
        self.majorityclass = label_distribution.index(np.nanmax(label_distribution))

        # introduce cross-validation
        from sklearn.cross_validation import KFold
        skf = KFold(len(training_samples), n_folds=self.num_folds)

        print"Number of Folds:", len(skf)

        print "Training Samples (Sequences):", len(training_samples)

        if self.balance:
            print "\nTraining " + self.variant + "with Balanced Labels..."
        else:
            print "\nTraining " + self.variant + "..."

        print "{:<9}".format("  Epoch"), \
            "{:<9}".format("  Train"), \
            "{:<9}".format("  Valid"), \
            "{:<9}".format("  Time"), \
            "\n======================================"
        start_time = time.clock()
        RNN_train_err = []
        RNN_val_err = []
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

                fold_training = [training_samples[ktrain[i]] for i in range(0,len(ktrain))]
                fold_train_labels = [label_train[ktrain[i]] for i in range(0,len(ktrain))]
                rep_training = np.array(fold_training)
                rep_label_train = np.array(fold_train_labels)

                if self.balance:
                    t_label_train = du.transpose(RNN.flatten_sequence(label_train))
                    rep = []
                    for r in range(0, self.num_output):
                        rep.append(int(math.floor((len(t_label_train[r]) / np.nansum(t_label_train[r])) + 1)))
                        rep_training, rep_label_train = RNN.add_representation(rep_training, rep_label_train, r, rep[r],
                                                                           0.2)
                    rep_training, rep_label_train = du.sample(rep_training, rep_label_train, p=1, n=len(fold_training))

                for i in range(0, len(rep_training), self.batch_size):
                    batch_cost = []
                    # get the cost of each sequence in the batch
                    for j in range(i, min(len(rep_training) - 1, i + self.batch_size - 1)):
                        batch_cost.append(self.train_RNN_no_update([rep_training[j]], rep_label_train[j]))

                    j = min(len(rep_training) - 1, i + self.batch_size - 1)

                    epoch += self.train_RNN_update([rep_training[j]], rep_label_train[j],
                                                   batch_cost, self.batch_size)

                    n_train += 1

                for i in range(0, len(ktest)):
                    # get the validation error
                    eval += self.compute_cost_RNN([training_samples[ktest[i]]], label_train[ktest[i]])
                    n_test += 1

            RNN_train_err.append(epoch / n_train)
            RNN_val_err.append(eval / n_test)
            print "{:<9}".format("Epoch " + str(e + 1) + ":"), \
                "  {0:.4f}".format(epoch / n_train), \
                "   {0:.4f}".format(eval / n_test), \
                "   {0:.1f}s".format(time.clock() - epoch_time)

            # if e == 0:
            #     previous = eval/n_test
            # else:
            #     if eval/n_test - previous > 0.005:
            #         break
            #     previous = eval/n_test
            if math.isnan(epoch / n_train):
                print "NaN Value found: Rebuilding Network..."
                self.isBuilt = False
                self.isInitialized = False

        tmp_scaling = self.scale_output
        self.scale_output = False
        pred = self.predict(training)
        self.scale_output = tmp_scaling

        self.max_preds = np.max(pred,axis=0)
        self.min_preds = np.min(pred, axis=0)

        print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

        self.train_validation_RNN = [['RNN Training Error'], ['RNN Validation Error']]
        for i in range(0, len(RNN_train_err)):
            self.train_validation_RNN[0].append(str(RNN_train_err[i]))
        for i in range(0, len(RNN_val_err)):
            self.train_validation_RNN[1].append(str(RNN_val_err[i]))

    def predict(self, test):

        if len(self.cov_mean) == 0 or len(self.cov_stdev) == 0:
            print "Scaling factors have not been generated: calculating using test sample"
            t_tr = du.transpose(RNN.flatten_sequence(test))
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

        if self.scale_output:
            print "Scaling output..."

        predictions_RNN = []
        for i in range(0, len(test_samples)):
            # get the prediction and calculate cost
            prediction_RNN = self.pred_RNN([test_samples[i]])
            if self.scale_output:
                prediction_RNN -= self.min_preds
                prediction_RNN /= (self.max_preds - self.min_preds)
                prediction_RNN = np.clip(prediction_RNN, 0, 1)

                prediction_RNN = [(x * [1 if c == self.majorityclass else 0.9999 for c in range(0,self.num_output)])
                                  if np.sum(x) == 4 else x for x in prediction_RNN]

            for j in range(0, len(prediction_RNN)):
                predictions_RNN.append(prediction_RNN[j].tolist())

        predictions_RNN = np.round(predictions_RNN, 3).tolist()

        return predictions_RNN

    def test(self, test, test_labels=None, label_names=None):
        if test_labels is None:
            return self.predict(test)

        if len(self.cov_mean) == 0 or len(self.cov_stdev) == 0:
            print "Scaling factors have not been generated: calculating using test sample"
            t_tr = du.transpose(RNN.flatten_sequence(test))
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
        avg_err_RNN = []

        if self.scale_output:
            print "Scaling output..."

        predictions_RNN = []
        for i in range(0, len(test_samples)):
            # get the prediction and calculate cost
            prediction_RNN = self.pred_RNN([test_samples[i]])
            #prediction_RNN += .5-self.avg_preds
            if self.scale_output:
                prediction_RNN -= self.min_preds
                prediction_RNN /= (self.max_preds - self.min_preds)
                prediction_RNN = np.clip(prediction_RNN,0,1)
                prediction_RNN = [(x * [1 if c == self.majorityclass else 0.9999 for c in range(0, self.num_output)])
                                  if np.sum(x) == 4 else x for x in prediction_RNN]
            avg_err_RNN.append(self.compute_cost_RNN([test_samples[i]], label_test[i]))

            for j in range(0, len(label_test[i])):
                p_count += 1

                classes.append(label_test[i][j].tolist())
                predictions_RNN.append(prediction_RNN[j].tolist())

        predictions_RNN = np.round(predictions_RNN, 3).tolist()

        actual = []
        pred_RNN = []
        cor_RNN = []

        # get the percent correct for the predictions
        # how often the prediction is right when it is made
        for i in range(0, len(predictions_RNN)):
            c = classes[i].index(max(classes[i]))
            actual.append(c)

            p_RNN = predictions_RNN[i].index(max(predictions_RNN[i]))
            pred_RNN.append(p_RNN)
            cor_RNN.append(int(c == p_RNN))

        # calculate a naive baseline using averages
        flattened_label = []
        for i in range(0, len(label_test)):
            for j in range(0, len(label_test[i])):
                flattened_label.append(label_test[i][j])
        flattened_label = np.array(flattened_label)
        avg_class_pred = np.mean(flattened_label,0)

        print "Predicting:", avg_class_pred, "for baseline*"
        for i in range(0, len(flattened_label)):
            res = RNN.AverageCrossEntropy(np.array(avg_class_pred), np.array(classes[i]))
            avg_class_err.append(res)
            # res = RNN.AverageCrossEntropy(np.array(predictions_RNN[i]), np.array(classes[i]))
            # avg_err_RNN.append(res)
        print "*This is calculated from the TEST labels"

        from sklearn.metrics import roc_auc_score,f1_score
        from skll.metrics import kappa

        kpa = []
        auc = []
        f1s = []
        apr = []
        t_pred = du.transpose(predictions_RNN)
        t_lab = du.transpose(flattened_label)

        for i in range(0,len(t_lab)):
            #if i == 0 or i == 3:
            #    t_pred[i] = du.normalize(t_pred[i],method='max')
            temp_p = [round(j) for j in t_pred[i]]

            kpa.append(kappa(t_lab[i], t_pred[i]))
            apr.append(du.Aprime(t_lab[i],t_pred[i]))
            auc.append(roc_auc_score(t_lab[i],t_pred[i]))

            if np.nanmax(temp_p)==0:
                f1s.append(0)
            else:
                f1s.append(f1_score(t_lab[i],temp_p))

        if label_names is None or len(label_names) != len(t_lab):
            label_names = []
            for i in range(0, len(t_lab)):
                label_names.append("Label " + str(i + 1))

        RNN.print_label_distribution(label_test, label_names)

        self.eval_metrics = [np.nanmean(avg_err_RNN),np.nanmean(auc),np.nanmean(kpa),
                             np.nanmean(f1s),np.nanmean(cor_RNN) * 100]

        print "\nBaseline Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_class_err))
        print "\nNetwork Performance:"
        print "Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_err_RNN))
        print "AUC:", "{0:.4f}".format(np.nanmean(auc))
        print "A':", "{0:.4f}".format(np.nanmean(apr))
        print "Kappa:", "{0:.4f}".format(np.nanmean(kpa))
        print "F1 Score:", "{0:.4f}".format(np.nanmean(f1s))
        print "Percent Correct:", "{0:.2f}%".format(np.nanmean(cor_RNN) * 100)

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
        for i in range(0, len(predictions_RNN)):
            actual.append(flattened_label[i].index(max(flattened_label[i])))
            predicted.append(predictions_RNN[i].index(max(predictions_RNN[i])))

        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(actual, predicted)
        for cm in conf_mat:
            cm_row = "\t"
            for element in cm:
                cm_row += "{:<6}".format(element)
            print cm_row
        print "\n=============================================="

        return predictions_RNN

    def get_performance(self):
        return self.eval_metrics

#################################################################################################


def train_and_evaluate_model(recurrent_nodes, batches, epochs, dropout1=0.3, dropout2=0.3, step_size=0.01,
                             balance_model=False, scale_output=True, variant="GRU", folds=5):
    confidence_table = []
    uid = 1

    filename = "Dataset/ground_truth_no_resample.csv"

    data, labels, student = RNN.load_data(filename, 0, 1, range(3, 207), range(207, 211))

    RNN.print_label_distribution(labels, ["Confused", "Concentrating", "Bored", "Frustrated"])
    eval_metrics = []
    unique_st = du.unique(student)
    st_fold = folds

    fold = []
    for f in range(0, st_fold):
        fold.append(0)

    import random as rand
    for f in unique_st:
        r = rand.randrange(0, st_fold)
        for s in range(0, len(student)):
            if student[s] == f:
                student[s] = r
                fold[r] += 1

    print '\nStudent Fold Distribution:'
    print fold
    GNET = ""
    for f in range(0, st_fold):
        training = []
        test = []
        label_train = []
        label_test = []
        for i in range(0, len(data)):
            if student[i] == f:
                test.append(data[i])
                label_test.append(labels[i])
            else:
                training.append(data[i])
                label_train.append(labels[i])

        training, label_train = du.shuffle(training, label_train)
        test, label_test = du.shuffle(test, label_test)

        GNET = RNN(variant)
        GNET.set_hyperparams(recurrent_nodes, batch_size=batches, num_folds=2, num_epochs=epochs, step_size=step_size,
                             dropout1=dropout1, dropout2=dropout2)

        RNN.print_label_distribution(label_train, ["Confused", "Concentrating", "Bored", "Frustrated"])
        GNET.set_training_params(batches, epochs, balance=balance_model, scale_output=scale_output)
        GNET.train(training, label_train)

        pred = GNET.test(test, label_test, ["Confused", "Concentrating", "Bored", "Frustrated"])
        label_name = ["Confused", "Concentrating", "Bored", "Frustrated"]
        lab = RNN.flatten_sequence(label_test)
        for k in range(0, len(pred)):
            confidence_table.append([uid, label_name[lab[k].index(max(lab[k]))],
                                     pred[k][0], pred[k][1], pred[k][2], pred[k][3]])
            uid += 1

        eval_metrics.append(GNET.get_performance())

    du.writetoCSV(eval_metrics, 'folds')
    ##################
    sys.setrecursionlimit(10000)
    print "Saving Instance"
    pu.saveInstance(GNET,"RNN.pickle")
    print "Instance Saved"
    ##################
    for m in eval_metrics:
        print m

    print '=============================================='
    print np.mean(eval_metrics, axis=0).tolist()
    du.writetoCSV(confidence_table, 'label_confidence', ['Unique ID', 'Human Label', 'Confused', 'Concentrating',
                                                         'Bored', 'Frustrated'])

def evaluate_model(data_filename, pickle_file, recurrent_nodes, batches, epochs,
							 dropout1=0.3, dropout2=0.3, step_size=0.01, balance_model=False,
							 scale_output=True, variant="GRU", folds=5):
    confidence_table = []
    uid = 1

    #filename = "Dataset/ground_truth_no_resample.csv"
    filename = data_filename
    data, labels, student = RNN.load_data(filename, 0, 1, range(3, 207), range(207, 211))

    RNN.print_label_distribution(labels, ["Confused", "Concentrating", "Bored", "Frustrated"])
    eval_metrics = []
    unique_st = du.unique(student)
    st_fold = folds

    fold = []
    for f in range(0, st_fold):
        fold.append(0)

    import random as rand
    for f in unique_st:
        r = rand.randrange(0, st_fold)
        for s in range(0, len(student)):
            if student[s] == f:
                student[s] = r
                fold[r] += 1

    print '\nStudent Fold Distribution:'
    print fold

    print "Loading GNET from Pickle File"
    GNET = pu.loadInstance(pickle_file)
    print "Pickle File: %s  -- Loaded" % pickle_file
    print "Evaluating Model"

    for f in range(0, st_fold):
        training = []
        test = []
        label_train = []
        label_test = []
        for i in range(0, len(data)):
            if student[i] == f:
                test.append(data[i])
                label_test.append(labels[i])
            else:
                training.append(data[i])
                label_train.append(labels[i])

        training, label_train = du.shuffle(training, label_train)
        test, label_test = du.shuffle(test, label_test)
        
        pred = GNET.test(test, label_test, ["Confused", "Concentrating", "Bored", "Frustrated"])
        label_name = ["Confused", "Concentrating", "Bored", "Frustrated"]
        lab = RNN.flatten_sequence(label_test)
        for k in range(0, len(pred)):
            confidence_table.append([uid, label_name[lab[k].index(max(lab[k]))],
                                     pred[k][0], pred[k][1], pred[k][2], pred[k][3]])
            uid += 1

        eval_metrics.append(GNET.get_performance())

    du.writetoCSV(eval_metrics, 'folds')

    for m in eval_metrics:
        print m

    print '=============================================='
    print np.mean(eval_metrics, axis=0).tolist()
    du.writetoCSV(confidence_table, 'label_confidence', ['Unique ID', 'Human Label', 'Confused', 'Concentrating',
                                                         'Bored', 'Frustrated'])


if __name__ == "__main__":
    run_start = time.clock()
    train_and_evaluate_model(200, 1, 20, step_size=0.005, balance_model=False, scale_output=True, variant="GRU")
    evaluate_model("Dataset/ground_truth_no_resample.csv", "RNN.pickle", 200, 1, 20, step_size=0.005, balance_model=False, scale_output=True, variant="GRU")
    print 'Total Runtime:', "{0:.1f}s".format(time.clock() - run_start)

