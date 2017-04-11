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
    def build_sequences(data, primary_column, secondary_column, covariate_columns, label_columns,
                        one_hot_labels=False, exclude_unlabeled=True):
        pdata = []
        labels = []
        primary_group = []

        # if data is not null, build the dataset
        if data is not None:
            data = du.transpose(data)

            if one_hot_labels:
                assert len(label_columns) == 1

                numerated = du.numerate(data[label_columns[0]], ignore=[''])
                one_hot = np.zeros([int(np.nanmax(numerated)) + 1, len(numerated)])

                for i in range(0, len(numerated)):
                    if np.isnan(numerated[i]):
                        for j in range(0, len(one_hot)):
                            one_hot[j][i] = float('nan')
                    else:
                        one_hot[numerated[i]][i] = 1

                        label_columns = []
                cols = len(data)
                for i in range(0, len(one_hot)):
                    data.append(one_hot[i].tolist())
                    label_columns.append(cols + i)

            for i in range(0, len(covariate_columns)):
                for j in range(0, len(data[i])):
                    data[covariate_columns[i]][j] = float(data[covariate_columns[i]][j])

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
        group = []
        for i in range(0, len(pdata)):
            if len(pdata[i]) == 0 or (exclude_unlabeled and np.sum(labels[i]) == 0):
                continue
            data.append(pdata[i])
            label.append(labels[i])
            group.append(primary_group[i])

        return np.array(data), np.array(label), group

    @staticmethod
    def build_sequences_with_next_problem_label(data, primary_column, secondary_column, covariate_columns,
                                                label_columns,one_hot_labels=False):
        pdata = []
        labels = []
        primary_group = []

        # if data is not null, build the dataset
        if data is not None:
            data = du.convert_to_floats(data)

            if one_hot_labels:
                assert len(label_columns) == 1
                data = du.transpose(data)

                numerated = du.numerate(data[label_columns[0]],ignore=[''])
                one_hot = np.zeros([int(np.nanmax(numerated))+1,len(numerated)])

                for i in range(0,len(numerated)):
                    if np.isnan(numerated[i]):
                        for j in range(0,len(one_hot)):
                            one_hot[j][i] = float('nan')
                    else:
                        one_hot[numerated[i]][i] = 1

                        label_columns = []
                cols = len(data)
                for i in range(0,len(one_hot)):
                    data.append(one_hot[i].tolist())
                    label_columns.append(cols + i)

                data = du.transpose(data)

            primary = du.unique(du.transpose(data)[primary_column])
            # for each user...
            for i in range(0, len(primary)):
                p_set = du.select(data, primary[i], '==', primary_column)
                secondary = du.unique(du.transpose(p_set)[secondary_column])

                for j in range(0, len(secondary)):
                    s_set = du.select(p_set, secondary[j], '==', secondary_column)
                    timesteps = []
                    ts_labels = []

                    con = 0
                    for k in range(0, len(s_set) - 1):
                        valid = True
                        for m in range(0, len(covariate_columns)):
                            try:
                                cov = float(s_set[k][covariate_columns[m]])
                            except ValueError:
                                valid = False
                                break
                        for m in range(0, len(label_columns)):
                            if np.isnan(s_set[k + 1][label_columns[m]]):
                                valid = False
                                break
                            try:
                                cov = float(s_set[k + 1][label_columns[m]])
                            except ValueError:
                                valid = False
                                break
                        if not valid:
                            break
                        else:
                            step = []
                            for m in range(0, len(covariate_columns)):
                                step.append(float(s_set[k][covariate_columns[m]]))

                            timesteps.append(np.array(step))

                            lbl = []
                            for m in range(0, len(label_columns)):
                                lbl.append(float(s_set[k + 1][label_columns[m]]))

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
        group = []
        for i in range(0, len(pdata)):
            if len(pdata[i]) == 0:
                continue
            data.append(pdata[i])
            label.append(labels[i])
            group.append(primary_group[i])

        return np.array(data), np.array(label), np.array(group)

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
        pdata, labels, grouping = RNN.build_sequences(data, primary_column, secondary_column, covariate_columns, [1, 2],
                                                      exclude_unlabeled=False)

        print '\nDataset Info:'
        print 'number of samples:', len(pdata)
        print 'sequence length of first sample:', len(pdata[0])
        print 'input nodes: ', len(pdata[0][0])

        return pdata, grouping

    @staticmethod
    def load_data(filename, primary_column, secondary_column, covariate_columns, label_columns,
                  use_next_timestep_label=False, one_hot_labels=False, load_from_file=False, limit=None):
        # load from file or rebuild dataset
        load = load_from_file

        data = None
        if not load:
            data, headers = du.loadCSVwithHeaders(filename,limit)

            for i in range(0, len(headers)):
                print '{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i])
        else:
            print 'Skipping dataset loading - using cached data instead'

            headers = du.readHeadersCSV(filename)
            for i in range(0, len(headers)):
                print '{:>2}:  {:<18}'.format(str(i), headers[i])

        print '\ntransforming data to time series...'
        if use_next_timestep_label:
            pdata, labels, grouping = RNN.build_sequences_with_next_problem_label(data, primary_column,
                                                                                  secondary_column, covariate_columns,
                                                                                  label_columns, one_hot_labels)
        else:
            pdata, labels, grouping = RNN.build_sequences(data, primary_column, secondary_column, covariate_columns,
                                                          label_columns, one_hot_labels, exclude_unlabeled=True)

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

    @staticmethod
    def build_label_mask(sequence_labels, num_output):
        mask = []
        # nseq = np.array(sequence_labels)
        for i in sequence_labels:
            if i is None or len(i) == 0 or np.nansum(i) == 0:
                mask.append([0 for j in range(0,num_output)])
            else:
                mask.append([1 for j in range(0,num_output)])
        return np.array(mask)

    @staticmethod
    def weighted_crossentropy(predictions, targets, weights_per_label):
        # implementation derived from the following source:
        # http://stackoverflow.com/questions/39412051/how-to-implement-weighted-binary-crossentropy-on-theano

        # Copy the tensor
        tgt = targets.copy("tgt")
        newshape = (T.shape(tgt)[0],)
        tgt = T.reshape(tgt, newshape)

        # Make it an integer.
        tgt = T.cast(tgt, 'int32')

        # weights_per_label = theano.shared(lasagne.utils.floatX([0.2, 0.4]))

        weights = weights_per_label[tgt]  # returns a targets-shaped weight matrix
        loss = lasagne.objectives.aggregate(T.nnet.categorical_crossentropy(predictions, tgt), weights=weights)

        return loss

    @staticmethod
    def train_on_batch(RNN_model,training_batch,training_batch_labels,covariates=None):
        training_cpy = list(training_batch)
        if covariates is None and RNN_model.covariates is None:
            RNN_model.num_input = du.len_deepest(training_cpy)
            RNN_model.covariates = range(0,RNN_model.num_input)
        elif covariates is not None:
            assert type(covariates) is list
            assert max(covariates) < du.len_deepest(training_cpy)
            assert min(covariates) >= 0
            RNN_model.covariates = du.unique(covariates)
            RNN_model.num_input = len(RNN_model.covariates)
            for a in range(0, len(training_cpy)):
                if type(training_cpy[a]) is not list:
                    training_cpy[a] = training_cpy[a].tolist()
                for e in range(0, len(training_cpy[a])):
                    c = []
                    for i in range(0, len(RNN_model.covariates)):
                        c.append(training_batch[a][e][RNN_model.covariates[i]])
                    training_cpy[a][e] = c

                RNN_model.num_output = du.len_deepest(training_batch_labels)
        else:
            assert max(RNN_model.covariates) < du.len_deepest(training_cpy)
            assert min(RNN_model.covariates) >= 0
            RNN_model.num_input = len(RNN_model.covariates)
            for a in range(0, len(training_cpy)):
                if type(training_cpy[a]) is not list:
                    training_cpy[a] = training_cpy[a].tolist()
                for e in range(0, len(training_cpy[a])):
                    c = []
                    for i in range(0, len(RNN_model.covariates)):
                        c.append(training_batch[a][e][RNN_model.covariates[i]])
                    training_cpy[a][e] = c

                RNN_model.num_output = du.len_deepest(training_batch_labels)

        if not RNN_model.isBuilt:
            RNN_model.build_network()

        RNN.isTrained = True

        t_tr = du.transpose(RNN.flatten_sequence(training_cpy))

        if len(RNN_model.cov_mean) == 0:
            RNN_model.cov_mean = []
            RNN_model.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                RNN_model.cov_mean.append(mn)
                RNN_model.cov_stdev.append(sd)

        training_samples = []

        import math
        for a in range(0, len(training_cpy)):
            sample = []
            for e in range(0, len(training_cpy[a])):
                covar = []
                for i in range(0, len(training_cpy[a][e])):
                    cov = 0
                    if RNN_model.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (training_cpy[a][e][i] - RNN_model.cov_mean[i]) / RNN_model.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covar.append(cov)
                sample.append(covar)
            training_samples.append(sample)

        label_train = training_batch_labels

        label_distribution = RNN.get_label_distribution(label_train)
        RNN_model.majorityclass = label_distribution.index(np.nanmax(label_distribution))

        rep_training = np.array(training_samples)
        rep_label_train = np.array(label_train)

        epoch = 0
        n_train = 0


        for i in range(0, len(rep_training), len(training_batch)):
            batch_cost = []
            # get the cost of each sequence in the batch
            for j in range(i, min(len(rep_training) - 1, i + len(training_batch) - 1)):
                batch_cost.append(RNN_model.train_RNN_no_update([rep_training[j]], rep_label_train[j]))

            j = min(len(rep_training) - 1, i + len(training_batch) - 1)

            epoch += RNN_model.train_RNN_update([rep_training[j]], rep_label_train[j],
                                                batch_cost, len(training_batch))

            n_train += 1

        RNN_model.RNN_train_err.append(epoch / n_train)

    @staticmethod
    def train_on_epoch(RNN_model, training_batch, training_batch_labels,batch_size = 10, covariates=None):
        training_cpy = list(training_batch)
        if covariates is None and RNN_model.covariates is None:
            RNN_model.num_input = du.len_deepest(training_cpy)
            RNN_model.covariates = range(0, RNN_model.num_input)
        elif covariates is not None:
            assert type(covariates) is list
            assert max(covariates) < du.len_deepest(training_cpy)
            assert min(covariates) >= 0
            RNN_model.covariates = du.unique(covariates)
            RNN_model.num_input = len(RNN_model.covariates)
            for a in range(0, len(training_cpy)):
                if type(training_cpy[a]) is not list:
                    training_cpy[a] = training_cpy[a].tolist()
                for e in range(0, len(training_cpy[a])):
                    c = []
                    for i in range(0, len(RNN_model.covariates)):
                        c.append(training_batch[a][e][RNN_model.covariates[i]])
                    training_cpy[a][e] = c

                RNN_model.num_output = du.len_deepest(training_batch_labels)
        else:
            assert max(RNN_model.covariates) < du.len_deepest(training_cpy)
            assert min(RNN_model.covariates) >= 0
            RNN_model.num_input = len(RNN_model.covariates)
            for a in range(0, len(training_cpy)):
                if type(training_cpy[a]) is not list:
                    training_cpy[a] = training_cpy[a].tolist()
                for e in range(0, len(training_cpy[a])):
                    c = []
                    for i in range(0, len(RNN_model.covariates)):
                        c.append(training_batch[a][e][RNN_model.covariates[i]])
                    training_cpy[a][e] = c

                RNN_model.num_output = du.len_deepest(training_batch_labels)

        if not RNN_model.isBuilt:
            RNN_model.build_network()

        RNN.isTrained = True

        t_tr = du.transpose(RNN.flatten_sequence(training_cpy))

        if len(RNN_model.cov_mean) == 0:
            RNN_model.cov_mean = []
            RNN_model.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                RNN_model.cov_mean.append(mn)
                RNN_model.cov_stdev.append(sd)

        training_samples = []

        import math
        for a in range(0, len(training_cpy)):
            sample = []
            for e in range(0, len(training_cpy[a])):
                covar = []
                for i in range(0, len(training_cpy[a][e])):
                    cov = 0
                    if RNN_model.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (training_cpy[a][e][i] - RNN_model.cov_mean[i]) / RNN_model.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covar.append(cov)
                sample.append(covar)
            training_samples.append(sample)

        label_train = training_batch_labels

        label_distribution = RNN.get_label_distribution(label_train)
        RNN_model.majorityclass = label_distribution.index(np.nanmax(label_distribution))

        rep_training = np.array(training_samples)
        rep_label_train = np.array(label_train)

        epoch = 0
        n_train = 0

        for i in range(0, len(rep_training), batch_size):
            batch_cost = []
            # get the cost of each sequence in the batch
            for j in range(i, min(len(rep_training) - 1, i + batch_size - 1)):
                batch_cost.append(RNN_model.train_RNN_no_update([rep_training[j]], rep_label_train[j]))

            j = min(len(rep_training) - 1, i + batch_size - 1)

            epoch += RNN_model.train_RNN_update([rep_training[j]], rep_label_train[j],
                                                batch_cost, batch_size)

            n_train += 1

        RNN_model.RNN_train_err.append(epoch / n_train)
        return RNN_model

    @staticmethod
    def test_model(RNN_model,test,test_labels):

        if test_labels is None:
            return RNN_model.predict(test)
        test_cpy = list(test)
        if not du.len_deepest(test_cpy) == RNN_model.num_input:
            if RNN_model.covariates is not None:
                for a in range(0, len(test_cpy)):
                    if type(test_cpy[a]) is not list:
                        test_cpy[a] = test_cpy[a].tolist()
                    for e in range(0, len(test_cpy[a])):
                        c = []
                        for i in range(0, len(RNN_model.covariates)):
                            c.append(test_cpy[a][e][RNN_model.covariates[i]])
                        test_cpy[a][e] = c

        if len(RNN_model.cov_mean) == 0 or len(RNN_model.cov_stdev) == 0:
            t_tr = du.transpose(RNN.flatten_sequence(test_cpy))
            RNN_model.cov_mean = []
            RNN_model.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                RNN_model.cov_mean.append(mn)
                RNN_model.cov_stdev.append(sd)

        test_samples = []

        import math
        for a in range(0, len(test_cpy)):
            sample = []
            for e in range(0, len(test_cpy[a])):
                covariates = []
                for i in range(0, len(test_cpy[a][e])):
                    cov = 0
                    if RNN_model.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (test_cpy[a][e][i] - RNN_model.cov_mean[i]) / RNN_model.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covariates.append(cov)
                sample.append(covariates)
            test_samples.append(sample)

        label_test = test_labels

        classes = []
        p_count = 0

        avg_class_err = []
        avg_err_RNN = []


        predictions_RNN = []
        for i in range(0, len(test_samples)):
            # get the prediction and calculate cost
            prediction_RNN = RNN_model.pred_RNN([test_samples[i]])
            # prediction_RNN += .5-self.avg_preds
            if RNN_model.scale_output:
                prediction_RNN -= RNN_model.min_preds
                prediction_RNN /= (RNN_model.max_preds - RNN_model.min_preds)
                prediction_RNN = np.clip(prediction_RNN, 0, 1)
                prediction_RNN = [(x * [1 if c == RNN_model.majorityclass else 0.9999 for c in range(0, RNN_model.num_output)])
                                  if np.sum(x) == 4 else x for x in prediction_RNN]
            avg_err_RNN.append(RNN_model.compute_cost_RNN([test_samples[i]], label_test[i]))

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

        from sklearn.metrics import roc_auc_score, f1_score
        from skll.metrics import kappa

        kpa = []
        auc = []
        f1s = []
        apr = []
        t_pred = du.transpose(predictions_RNN)
        t_lab = du.transpose(flattened_label)

        for i in range(0, len(t_lab)):
            # if i == 0 or i == 3:
            #    t_pred[i] = du.normalize(t_pred[i],method='max')
            temp_p = [round(j) for j in t_pred[i]]

            try:
                kpa.append(kappa(t_lab[i], t_pred[i]))
                apr.append(du.Aprime(t_lab[i], t_pred[i]))
                auc.append(roc_auc_score(t_lab[i], t_pred[i]))
            except ValueError as e:
                RNN_model.eval_metrics = [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')]
                return predictions_RNN

            if np.nanmax(temp_p) == 0:
                f1s.append(0)
            else:
                f1s.append(f1_score(t_lab[i], temp_p))

                RNN_model.eval_metrics = [np.nanmean(avg_err_RNN), np.nanmean(auc), np.nanmean(kpa),
                             np.nanmean(f1s), np.nanmean(cor_RNN) * 100]

        return RNN_model

    @staticmethod
    def save_model(model, filename):
        print "Saving Model to: " + filename
        pu.saveInstance(model, filename)

    @staticmethod
    def load_model(filename):
        print "Loading Model from: " + filename
        return pu.loadInstance(filename)

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
        self.dropout1 = 0.3
        self.dropout2 = 0.3

        self.network_output = T.matrix('network_output')
        self.target_values = T.matrix('target_output')
        self.label_mask = T.matrix('target_mask')
        self.cost_vector = T.dvector('cost_list')
        self.num_elements = T.dscalar('batch_size')
        self.entropy_weights = None

        self.covariates = None

        self.min_preds = [1,1,1,1]
        self.min_preds = [0,0,0,0]
        self.avg_preds = [0.5,0.5,0.5,0.5]

        self.eval_metrics = ['NA', 'NA', 'NA', 'NA', 'NA']
        self.kappa_threshold = []

        self.l_in = None
        self.l_AE_drop = None
        self.l_AE_hidden = None
        self.l_AE_inverse = None
        self.l_norm = None
        self.l_drop1 = None
        self.l_Recurrent = None
        self.l_reshape_RNN = None
        self.l_relu = None
        self.l_drop2 = None
        self.l_output_RNN = None

        self.network_output_RNN = None
        self.network_output_RNN_test = None
        self.autoencoder_hidden = None
        self.cost_RNN = None
        self.all_params_RNN = None
        self.updates_adagrad = None
        self.updates_adam = None
        self.updates_adadelta = None
        self.updates_nesterov = None
        self.compute_cost_RNN = None
        self.pred_RNN = None
        self.train_RNN_no_update = None
        self.train_RNN_update = None
        self.train_AE_update = None
        self.train_predict_update = None

        self.RNN_train_err = []
        self.RNN_val_err = []

        self.max_preds = 1
        self.min_preds = 0

        self.train_validation_RNN = [['RNN Training Error'], ['RNN Validation Error']]

        self.isBuilt = False
        self.isInitialized = False

        self.isTrained = False
        self.trained_epoch = 0

        self.balance = False
        self.scale_output = False
        self.early_stop = False
        self.use_autoencoder = False
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

    def set_training_params(self, batch_size, num_epochs, balance=False, scale_output=False, early_stop=False,
                            num_folds=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if num_folds is not None:
            self.num_folds = num_folds
        self.balance = balance
        self.scale_output = scale_output
        self.early_stop = early_stop

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
        verbose = False
        import time
        print '\nBuilding Network...'
        if verbose:
            print '\nDefining Network Structure...'
        start = time.clock()

        if not self.isInitialized:
            # Recurrent network structure
            self.l_in = lasagne.layers.InputLayer(shape=(None, None, self.num_input))
            self.l_drop1 = lasagne.layers.DropoutLayer(self.l_in, self.dropout1)

            if self.variant == "GRU" or self.variant == "gru":
                self.l_Recurrent = lasagne.layers.GRULayer(self.l_drop1, self.num_units, precompute_input=True,
                                                           grad_clipping=100.)
            elif self.variant == "LSTM" or self.variant == "lstm":
                self.l_Recurrent = lasagne.layers.LSTMLayer(self.l_drop1, self.num_units, precompute_input=True,
                                                            grad_clipping=100.,
                                                            #forgetgate=Gate(b=lasagne.init.Constant(5)),
                                                            peepholes=False)
            else:
                self.l_Recurrent = lasagne.layers.RecurrentLayer(self.l_drop1, self.num_units, precompute_input=True,
                                                           grad_clipping=100.)

            self.l_reshape_RNN = lasagne.layers.ReshapeLayer(self.l_Recurrent, shape=(-1, self.num_units))
            self.l_relu = lasagne.layers.RandomizedRectifierLayer(self.l_reshape_RNN)
            self.l_drop2 = lasagne.layers.DropoutLayer(self.l_relu,self.dropout2)
            self.l_output_RNN = lasagne.layers.DenseLayer(self.l_drop2, num_units=self.num_output,
                                                          W=lasagne.init.Normal(),
                                                          nonlinearity=lasagne.nonlinearities.softmax)

            self.isInitialized = True
        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Compiling Output Functions...",
        start = time.clock()
        # theano variables for output
        self.network_output_RNN = lasagne.layers.get_output(self.l_output_RNN,deterministic=False)
        self.network_output_RNN_test = lasagne.layers.get_output(self.l_output_RNN, deterministic=True)

        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Defining Cost Function...",
        start = time.clock()
        # use cross-entropy for cost - average across the batch

        # self.cost_RNN = T.sum(RNN.weighted_crossentropy((self.network_output_RNN * self.label_mask) + (1-self.label_mask),
        #                                                 self.target_values + (1-self.label_mask), self.entropy_weights))

        # self.cost_RNN = (2*T.sum(
        #                     T.nnet.categorical_crossentropy((self.network_output_RNN * self.label_mask) +
        #                                                     (1-self.label_mask),
        #                                                     (self.target_values * self.entropy_weights) +
        #                                                     (1-self.label_mask)))) / \
        #                 (T.sum((self.target_values*self.entropy_weights))+(T.sum(
        #                     T.nnet.categorical_crossentropy((self.network_output_RNN * self.label_mask) +
        #                                                     (1-self.label_mask),
        #                                                     (self.target_values*self.entropy_weights) +
        #                                                     (1-self.label_mask)))))

        self.cost_RNN = lasagne.objectives.aggregate(
            T.nnet.categorical_crossentropy(self.network_output_RNN,self.target_values),
            weights=(self.label_mask*self.entropy_weights).max(axis=1)).mean()

        # lasagne.objectives.aggregate(
        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Compiling Parameter Functions...",
        start = time.clock()
        # theano variable for network parameters for updating
        self.all_params_RNN = lasagne.layers.get_all_params(self.l_output_RNN, trainable=True)

        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Compiling Update Functions...",
        start = time.clock()
        # update the network given a list of batch costs (for batches of sequences)
        self.updates_adagrad = lasagne.updates.adagrad((T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                     self.all_params_RNN,
                                                     self.step_size)

        self.updates_adam = lasagne.updates.adam((T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                 self.all_params_RNN,
                                                 self.step_size)

        self.updates_adadelta = lasagne.updates.adadelta((T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                 self.all_params_RNN,
                                                 self.step_size)

        self.updates_nesterov = lasagne.updates.nesterov_momentum((T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                 self.all_params_RNN,
                                                 self.step_size)

        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Compiling Cost Functions...",
        start = time.clock()
        # get the RNN cost given inputs and labels
        self.compute_cost_RNN = theano.function([self.l_in.input_var, self.target_values, self.label_mask],
                                                self.cost_RNN, allow_input_downcast=True)

        # get the prediction vector of the network given some inputs
        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Compiling Prediction Functions...",
        start = time.clock()
        self.pred_RNN_train = theano.function([self.l_in.input_var], self.network_output_RNN,
                                        allow_input_downcast=True)

        self.pred_RNN = theano.function([self.l_in.input_var], self.network_output_RNN_test,
                                        allow_input_downcast=True)

        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        if verbose:
            print "Compiling Training Functions...",
        start = time.clock()
        # get the cost of the network without updating parameters (for batch updating)
        self.train_RNN_no_update = theano.function([self.l_in.input_var, self.target_values, self.label_mask],
                                                   self.cost_RNN, allow_input_downcast=True)

        # get the cost of the network and update parameters based on previous costs (for batch updating)
        self.train_RNN_update = theano.function([self.l_in.input_var, self.target_values, self.label_mask,
                                                 self.cost_vector, self.num_elements],
                                                (T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                updates=self.updates_adagrad, allow_input_downcast=True)

        self.train_predict_update = theano.function([self.l_in.input_var, self.target_values, self.label_mask,
                                                     self.cost_vector, self.num_elements],
                                                    [(T.sum(self.cost_vector) + self.cost_RNN) / self.num_elements,
                                                     self.network_output_RNN], updates=self.updates_adagrad,
                                                    allow_input_downcast=True)

        if verbose:
            print '{0:.1f}s'.format(time.clock() - start)

        self.isBuilt = True

    def train_stable(self, training, training_labels, holdout_samples = None, holdout_labels = None, covariates=None, verbose=10):

        training_cpy = list(training)

        if holdout_samples is not None:
            assert holdout_labels is not None
        else:
            assert holdout_labels is None

        if covariates is None:
            self.num_input = du.len_deepest(training_cpy)
        else:
            assert type(covariates) is list
            assert max(covariates) < du.len_deepest(training_cpy)
            assert min(covariates) >= 0
            self.covariates = du.unique(covariates)
            self.num_input = len(self.covariates)
            for a in range(0,len(training_cpy)):
                if type(training_cpy[a]) is not list:
                    training_cpy[a] = training_cpy[a].tolist()
                for e in range(0,len(training_cpy[a])):
                    c = []
                    for i in range(0,len(self.covariates)):
                        c.append(training[a][e][self.covariates[i]])
                    training_cpy[a][e] = c

        self.num_output = du.len_deepest(training_labels)

        dist = 1-np.array(RNN.get_label_distribution(training_labels))
        wt = 1 + (dist - dist.min())
        # wt /= wt
        # wt = 1-np.array(du.normalize(du.softmax(np.array(RNN.get_label_distribution(training_labels))))) + 1
        # wt = [15,1,6,15]
        # wt = np.array(du.normalize(wt))
        # wt += 1-wt

        self.entropy_weights = theano.shared(lasagne.utils.floatX(wt))

        if not self.isBuilt:
            self.build_network()

        self.kappa_threshold = []
        for i in range(0,self.num_output):
            self.kappa_threshold.append(0.5)

        print "Cross Entropy Class Weights:\n\t",wt

        print "Network Params:", count_params(self.l_output_RNN)

        t_tr = du.transpose(RNN.flatten_sequence(training_cpy))

        if len(self.cov_mean) == 0:
            self.cov_mean = []
            self.cov_stdev = []

            for a in range(0,len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                self.cov_mean.append(mn)
                self.cov_stdev.append(sd)

        training_samples = []
        # training_samples = training_cpy

        import math
        for a in range(0,len(training_cpy)):
            sample = []
            for e in range(0,len(training_cpy[a])):
                covar = []
                for i in range(0,len(training_cpy[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (training_cpy[a][e][i]-self.cov_mean[i])/self.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covar.append(cov)
                sample.append(covar)
            training_samples.append(sample)

        label_train = training_labels

        label_distribution = RNN.get_label_distribution(label_train)
        self.majorityclass = label_distribution.index(np.nanmax(label_distribution))


        # introduce cross-validation
        from sklearn.cross_validation import KFold
        skf = KFold(len(training_samples), n_folds=self.num_folds)

        if not self.isTrained and verbose > 2:
            print"Number of Folds:", len(skf)

            print "Training Samples (Sequences):", len(training_samples)

            if self.balance:
                print "\nTraining " + self.variant + " with Balanced Labels..."
            else:
                print "\nTraining " + self.variant + "..."


            print "{:^9}".format("Epoch"), \
                "{:^9}".format("Train"), \
                "{:^9}".format("Val(ACE)"), \
                "{:^9}".format("Val(AUC)"), \
                "{:^9}".format("Val(Kpa)"), \
                "{:^9}".format("Time"), \
                "\n{:=^68}".format('')

        self.isTrained = True
        start_time = time.clock()
        self.RNN_train_err = []
        self.RNN_val_err = []
        previous = 0

        hold_moving = 0

        # for each epoch...
        pred = []
        for e in range(0, self.num_epochs):
            pred = []
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

            # train and validate



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
                        mask = RNN.build_label_mask(rep_label_train[j], self.num_output)
                        batch_cost.append(self.train_RNN_no_update([rep_training[j]], rep_label_train[j], mask))

                    j = min(len(rep_training) - 1, i + self.batch_size - 1)

                    mask = RNN.build_label_mask(rep_label_train[j],self.num_output)
                    # print np.nansum(rep_label_train[j]), ':', np.nansum(mask), '--',

                    res = self.train_predict_update([rep_training[j]], rep_label_train[j], mask,
                                                    batch_cost, self.batch_size)

                    # print res[0]

                    epoch += res[0]

                    for k in res[1]:
                        pred.append(k)
                    n_train += int(res[0] > 0)

                self.max_preds = np.max(pred, axis=0)
                self.min_preds = np.min(pred, axis=0)
                for i in range(0, len(ktest)):
                    # get the validation error
                    mask = RNN.build_label_mask(label_train[ktest[i]], self.num_output)
                    c = self.compute_cost_RNN([training_samples[ktest[i]]], label_train[ktest[i]], mask)
                    eval += c
                    n_test += int(c > 0)

            self.test(holdout_samples, holdout_labels, verbose=0, training=True)
            eval = self.eval_metrics
            # print r.eval_metrics
            n_test = 1
            # for i in range(0,len(holdout_set)):
            #     eval = self.compute_cost_RNN([holdout_set[i]], holdout_labels[i])
            #     n_test = 1

            self.RNN_train_err.append(epoch / n_train)

            try:
                self.RNN_val_err.append(float(eval[0]) / n_test)
            except TypeError:
                eval = [0, 0, 0, 0, 0]
                self.RNN_val_err.append(eval[0])



            if verbose > 1:
                print "{:^9}".format("Epoch " + str(self.trained_epoch + 1) + ":"), \
                    "{0:^9.4f}".format(epoch / n_train), \
                    "{0:^9.4f}".format(float(eval[0]) / n_test), \
                    "{0:^9.4f}".format(float(eval[1]) / n_test), \
                    "{0:^9.4f}".format(float(eval[2]) / n_test), \
                    "{:^9}".format("{0:.1f}s".format(time.clock() - epoch_time))
                # "{0:^9.4f}".format(eval[0] / n_test), \
                # eval = eval[0]

            past_epochs = 10

            if self.early_stop and len(self.RNN_train_err) > past_epochs + 1:
                avg_train = 0
                avg_val = 0

                for i in range(0, past_epochs):
                    avg_train += self.RNN_val_err[len(self.RNN_val_err) - 2 - i]
                    avg_val += self.RNN_val_err[len(self.RNN_val_err) - 1 - i]

                avg_train /= float(past_epochs)
                avg_val /= float(past_epochs)

                if avg_val - avg_train > 0:
                    self.load_from_file('RNN_tmp_params')
                    break

            self.save_parameters('RNN_tmp_params')
            self.trained_epoch += 1

            if math.isnan(epoch / n_train):
                if verbose > 2:
                    print "NaN Value found: Rebuilding Network..."
                self.isBuilt = False
                self.isInitialized = False

        if self.num_epochs > 1 and verbose > 2:
            print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

        self.train_validation_RNN = [['RNN Training Error'], ['RNN Validation Error']]
        for i in range(0, len(self.RNN_train_err)):
            self.train_validation_RNN[0].append(str(self.RNN_train_err[i]))
        for i in range(0, len(self.RNN_val_err)):
            self.train_validation_RNN[1].append(str(self.RNN_val_err[i]))

    def train(self, training, training_labels, holdout_size=.2, covariates=None, verbose=10):

        training, label_train = du.shuffle(training, training_labels)
        training, holdout_samples, training_labels, holdout_labels = du.split_training_test(training,label_train,
                                                                                            1-holdout_size)
        training_cpy = np.array(training)

        if covariates is None:
            covariates = range(du.len_deepest(training_cpy))

        self.covariates = np.array(du.unique(covariates), dtype=int)
        assert max(covariates) < du.len_deepest(training_cpy) and min(covariates) >= 0
        self.num_input = len(self.covariates)
        for a in range(0,len(training_cpy)):
            training_cpy[a] = np.array(training_cpy[a])
            for b in range(0,len(training_cpy[a])):
                training_cpy[a][b] = np.array(np.array(training_cpy[a][b])[self.covariates],dtype=float)

        self.num_output = du.len_deepest(training_labels)

        dist = 1-np.array(RNN.get_label_distribution(training_labels))
        wt = 1 + (dist - dist.min())

        import math

        t_label_train = du.transpose(RNN.flatten_sequence(label_train))
        rep = []
        for r in range(0, self.num_output):
            rep.append(int(math.floor((len(t_label_train[r]) / np.nansum(t_label_train[r])) + 1)))

        rep = np.array(rep,dtype=int)
        rep /= rep.min()
        wt = rep
        # wt /= wt
        # wt = 1-np.array(du.normalize(du.softmax(np.array(RNN.get_label_distribution(training_labels))))) + 1
        # wt = [15,1,6,15]
        # wt = np.array(du.normalize(wt))
        # wt += 1-wt

        self.entropy_weights = theano.shared(lasagne.utils.floatX(wt))

        if not self.isBuilt:
            self.build_network()

        self.kappa_threshold = []
        for i in range(0,self.num_output):
            self.kappa_threshold.append(0.5)

        print "Cross Entropy Class Weights:\n\t",wt

        print "Network Params:", count_params(self.l_output_RNN)

        t_tr = du.transpose(RNN.flatten_sequence(training_cpy))

        if len(self.cov_mean) == 0:
            self.cov_mean = []
            self.cov_stdev = []

            for a in range(0,len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                self.cov_mean.append(mn)
                self.cov_stdev.append(sd)

        training_samples = []
        # training_samples = training_cpy


        for a in range(0,len(training_cpy)):
            sample = []
            for e in range(0,len(training_cpy[a])):
                covar = []
                for i in range(0,len(training_cpy[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (training_cpy[a][e][i]-self.cov_mean[i])/self.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covar.append(cov)
                sample.append(covar)
            training_samples.append(sample)

        label_train = training_labels

        label_distribution = RNN.get_label_distribution(label_train)
        self.majorityclass = label_distribution.index(np.nanmax(label_distribution))

        if not self.isTrained and verbose > 2:

            print "Training Samples (Sequences):", len(training_samples)

            if self.balance:
                print "\nTraining " + self.variant + " with Balanced Labels..."
            else:
                print "\nTraining " + self.variant + "..."


            print "{:^9}".format("Epoch"), \
                "{:^9}".format("Train"), \
                "{:^9}".format("Moving E"), \
                "{:^9}".format("Val(ACE)"), \
                "{:^9}".format("Val(AUC)"), \
                "{:^9}".format("Val(Kpa)"), \
                "{:^9}".format("Time"), \
                "\n{:=^77}".format('')

        self.isTrained = True
        start_time = time.clock()
        self.RNN_train_err = []
        self.RNN_val_err = []
        previous = 0

        hold_moving = 0

        # for each epoch...
        pred = []
        for e in range(0, self.num_epochs):
            pred = []
            epoch_time = time.clock()
            epoch = 0
            eval = 0
            n_train = 0
            n_test = 0

            # train and validate
            # training, label_train = du.shuffle(training_samples, label_train)
            # fold_training, holdout_samples, \
            # fold_train_labels, holdout_labels = du.split_training_test(np.array(training_samples),
            #                                                            np.array(label_train),
            #                                                            1 - holdout_size)

            fold_training = np.array(training_samples)[:]
            fold_train_labels = np.array(label_train)[:]
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
                    mask = RNN.build_label_mask(rep_label_train[j], self.num_output)
                    batch_cost.append(self.train_RNN_no_update([rep_training[j]], rep_label_train[j], mask))

                j = min(len(rep_training) - 1, i + self.batch_size - 1)

                mask = RNN.build_label_mask(rep_label_train[j],self.num_output)
                # print np.nansum(rep_label_train[j]), ':', np.nansum(mask), '--',

                res = self.train_predict_update([rep_training[j]], rep_label_train[j], mask,
                                                batch_cost, self.batch_size)

                # print res[0]

                epoch += res[0]

                for k in res[1]:
                    pred.append(k)
                n_train += int(res[0] > 0)

                self.max_preds = np.max(pred, axis=0)
                self.min_preds = np.min(pred, axis=0)

            self.test(holdout_samples, holdout_labels, verbose=0, training=True)
            eval = self.eval_metrics
            # print r.eval_metrics
            n_test = 1
            # for i in range(0,len(holdout_set)):
            #     eval = self.compute_cost_RNN([holdout_set[i]], holdout_labels[i])
            #     n_test = 1

            self.RNN_train_err.append(epoch / n_train)

            try:
                self.RNN_val_err.append(float(1-eval[1]) / n_test) ##changed from eval[0]
            except TypeError:
                eval = [0, 0, 0, 0, 0]
                self.RNN_val_err.append(1-eval[1]) ##changed from eval[0]

            past_epochs = 3
            avg_train = 0
            avg_val = 0

            for i in range(0, past_epochs * int(self.trained_epoch >= past_epochs - 1)):
                avg_train += self.RNN_val_err[len(self.RNN_val_err) - 2 - i]
                avg_val += self.RNN_val_err[len(self.RNN_val_err) - 1 - i]

            avg_train /= float(past_epochs)
            avg_val /= float(past_epochs)

            hold_moving = avg_val

            if verbose > 1:
                print "{:^9}".format("Epoch " + str(self.trained_epoch + 1) + ":"), \
                    "{0:^9.4f}".format(epoch / n_train), \
                    "{0:^9.4f}".format(hold_moving) if self.trained_epoch >= past_epochs - 1 \
                        else "{0:^9}".format("---"), \
                    "{0:^9.4f}".format(float(eval[0]) / n_test), \
                    "{0:^9.4f}".format(float(eval[1]) / n_test), \
                    "{0:^9.4f}".format(float(eval[2]) / n_test), \
                    "{:^9}".format("{0:.1f}s".format(time.clock() - epoch_time))
                # "{0:^9.4f}".format(eval[0] / n_test), \
                # eval = eval[0]


            if self.early_stop and len(self.RNN_train_err) > past_epochs:
                if avg_val - avg_train > 0:
                    self.load_from_file('RNN_tmp_params')
                    break

            self.save_parameters('RNN_tmp_params')
            self.trained_epoch += 1

            if math.isnan(epoch / n_train):
                if verbose > 2:
                    print "NaN Value found: Rebuilding Network..."
                self.isBuilt = False
                self.isInitialized = False

        if self.num_epochs > 1 and verbose > 2:
            print "Total Training Time:", "{0:.1f}s".format(time.clock() - start_time)

        self.train_validation_RNN = [['RNN Training Error'], ['RNN Validation Error']]
        for i in range(0, len(self.RNN_train_err)):
            self.train_validation_RNN[0].append(str(self.RNN_train_err[i]))
        for i in range(0, len(self.RNN_val_err)):
            self.train_validation_RNN[1].append(str(self.RNN_val_err[i]))

    def predict(self, test):
        test_cpy = list(test)
        if not du.len_deepest(test_cpy) == self.num_input:
            if self.covariates is not None:
                for a in range(0, len(test_cpy)):
                    if type(test_cpy[a]) is not list:
                        test_cpy[a] = test_cpy[a].tolist()
                    for e in range(0, len(test[a])):
                        c = []
                        for i in range(0, len(self.covariates)):
                            c.append(test_cpy[a][e][self.covariates[i]])
                        test_cpy[a][e] = c

        if len(self.cov_mean) == 0 or len(self.cov_stdev) == 0:
            print "Scaling factors have not been generated: calculating using test sample"
            t_tr = du.transpose(RNN.flatten_sequence(test_cpy))
            self.cov_mean = []
            self.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                self.cov_mean.append(mn)
                self.cov_stdev.append(sd)

        test_samples = []
        # test_samples = test_cpy

        import math
        for a in range(0, len(test_cpy)):
            sample = []
            for e in range(0, len(test_cpy[a])):
                covariates = []
                for i in range(0, len(test_cpy[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (test_cpy[a][e][i] - self.cov_mean[i]) / self.cov_stdev[i]

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

    def test(self, test, test_labels=None, label_names=None, verbose=10, training=False):
        if test_labels is None:
            return self.predict(test)
        test_cpy = list(test)
        if not du.len_deepest(test_cpy) == self.num_input:
            if self.covariates is not None:
                for a in range(0, len(test_cpy)):
                    if type(test_cpy[a]) is not list:
                        test_cpy[a] = test_cpy[a].tolist()
                    for e in range(0, len(test_cpy[a])):
                        c = []
                        for i in range(0, len(self.covariates)):
                            c.append(test_cpy[a][e][self.covariates[i]])
                        test_cpy[a][e] = c

        if len(self.cov_mean) == 0 or len(self.cov_stdev) == 0:
            if verbose > 3:
                print "Scaling factors have not been generated: calculating using test sample"
            t_tr = du.transpose(RNN.flatten_sequence(test_cpy))
            self.cov_mean = []
            self.cov_stdev = []

            for a in range(0, len(t_tr)):
                mn = np.nanmean(t_tr[a])
                sd = np.nanstd(t_tr[a])
                self.cov_mean.append(mn)
                self.cov_stdev.append(sd)

        test_samples = []
        # test_samples = test_cpy

        import math
        for a in range(0, len(test_cpy)):
            sample = []
            for e in range(0, len(test_cpy[a])):
                covariates = []
                for i in range(0, len(test_cpy[a][e])):
                    cov = 0
                    if self.cov_stdev[i] == 0:
                        cov = 0
                    else:
                        cov = (test_cpy[a][e][i] - self.cov_mean[i]) / self.cov_stdev[i]

                    if math.isnan(cov) or math.isinf(cov):
                        cov = 0

                    covariates.append(cov)
                sample.append(covariates)
            test_samples.append(sample)

        label_test = test_labels
        if verbose > 3:
            print("\nTesting...")
        if verbose > 2:
            print "Test Samples:", len(test_samples)

        classes = []
        p_count = 0

        avg_class_err = []
        avg_err_RNN = []

        if self.scale_output and verbose > 3:
            print "Scaling output..."

        predictions_RNN = []
        for i in range(0, len(test_samples)):
            # get the prediction and calculate cost
            prediction_RNN = self.pred_RNN([test_samples[i]])

            if self.scale_output:
                prediction_RNN -= self.min_preds
                prediction_RNN /= (self.max_preds - self.min_preds)
                prediction_RNN = np.clip(prediction_RNN,0,1)
                prediction_RNN = [(x * [1 if c == self.majorityclass else 0.9999 for c in range(0, self.num_output)])
                                  if np.sum(x) == 4 else x for x in prediction_RNN]

            mask = RNN.build_label_mask(label_test[i], self.num_output)
            avg_err_RNN.append(self.compute_cost_RNN([test_samples[i]], label_test[i], mask))

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

        if verbose > 3:
            print "Predicting:", avg_class_pred, "for baseline*"
        for i in range(0, len(flattened_label)):
            res = RNN.AverageCrossEntropy(np.array(avg_class_pred), np.array(classes[i]))
            avg_class_err.append(res)
            # res = RNN.AverageCrossEntropy(np.array(predictions_RNN[i]), np.array(classes[i]))
            # avg_err_RNN.append(res)
        if verbose > 3:
            print "*This is calculated from the TEST labels"

        from sklearn.metrics import roc_auc_score,f1_score

        actual = []
        predicted = []
        flattened_label = flattened_label.tolist()
        for i in range(0, len(predictions_RNN)):
            if np.sum(flattened_label[i]) == 0:
                continue
            actual.append(flattened_label[i].index(max(flattened_label[i])))
            predicted.append(predictions_RNN[i].index(max(predictions_RNN[i])))

        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(actual, predicted)

        kpa = []
        opk = []
        auc = []
        f1s = []
        apr = []
        fkp = []

        fleiss = du.fleiss_kappa(conf_mat)

        t_pred = du.transpose(predictions_RNN)
        t_lab = du.transpose(flattened_label)

        for i in range(0,len(t_lab)):

            L = []
            P = []
            for j in range(0,len(t_lab[i])):
                if np.sum(flattened_label[j]) == 0:
                    continue
                L.append(t_lab[i][j])
                P.append(t_pred[i][j])

            #if i == 0 or i == 3:
            #    t_pred[i] = du.normalize(t_pred[i],method='max')
            temp_p = [round(j) for j in P]

            if training:
                self.kappa_threshold[i] = du.kappa_optimized_threshold(L, P)
            opk.append(du.kappa(L, P, split=self.kappa_threshold[i]))
            kpa.append(du.kappa(L, P))
            # fkp.append(du.fleiss_kappa([t_lab[i], P[i]]))
            apr.append(du.Aprime(L,P))
            auc.append(roc_auc_score(L,P))

            if np.nanmax(temp_p)==0:
                f1s.append(0)
            else:
                try:
                    f1s.append(f1_score(L,temp_p))
                except:
                    f1s.append(0)

        if label_names is None or len(label_names) != len(t_lab):
            label_names = []
            for i in range(0, len(t_lab)):
                label_names.append("Label " + str(i + 1))

        if verbose > 2:
            RNN.print_label_distribution(label_test, label_names)

        self.eval_metrics = [np.nanmean(avg_err_RNN),np.nanmean(auc),np.nanmean(kpa), np.nanmean(opk),
                             np.nanmean(f1s),fleiss,np.nanmean(cor_RNN) * 100]

        if verbose > 2:
            print "\nBaseline Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_class_err))
        if verbose > 1:
            print "\nNetwork Performance:"
            print "Average Cross-Entropy:", "{0:.4f}".format(np.nanmean(avg_err_RNN))
            print "A':", "{0:.4f}".format(np.nanmean(apr))
            print "Kappa:", "{0:.4f}".format(np.nanmean(kpa))
            print "Optimal Kappa:", "{0:.4f}".format(np.nanmean(opk))
            print "Fleiss Kappa:", "{0:.4f}".format(fleiss)
            print "F1 Score:", "{0:.4f}".format(np.nanmean(f1s))
            print "Percent Correct:", "{0:.2f}%".format(np.nanmean(cor_RNN) * 100)

            print "\n{:<15}".format("  Label"), \
                "{:<9}".format("  A'"), \
                "{:<9}".format("  Kappa"), \
                "{:<9}".format("  Op Kappa"), \
                "{:<9}".format("  F Stat"), \
                "\n=============================================="

            for i in range(0,len(t_lab)):
                print "{:<15}".format(label_names[i]), \
                    "{:<9}".format("  {0:.4f}".format(apr[i])), \
                    "{:<9}".format("  {0:.4f}".format(kpa[i])), \
                    "{:<9}".format("  {0:.4f}".format(opk[i])), \
                    "{:<9}".format("  {0:.4f}".format(f1s[i]))
            print "\n=============================================="

        if verbose > 2:
            print "Confusion Matrix:"

            for cm in conf_mat:
                cm_row = "\t"
                for element in cm:
                    cm_row += "{:<6}".format(element)
                print cm_row
            print "\n=============================================="

        return predictions_RNN

    def get_name(self):
        return self.variant + '_' + du.array_as_string(self.covariates)

    def get_performance(self):
        return self.eval_metrics

