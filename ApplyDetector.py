from RNN import RNN
import DataUtility as du
import numpy as np


def train_and_save_model(model_file, recurrent_nodes, batches, epochs, dropout1=0.0, dropout2=0.3, step_size=0.001,
                         balance_model=False, scale_output=True, early_stop=True, variant="LSTM"):

    filename = "Dataset/affect_ground_truth_reduced_features.csv"

    model_cov = range(3, 95)
    model_lab = range(95, 99)

    import random as rand
    np.random.seed(0)
    rand.seed(0)

    data, labels, student = RNN.load_data(filename, 0, 1, model_cov, model_lab, one_hot_labels=False)

    data, labels = du.shuffle(data, labels)
    data, holdout, labels, hold_label = du.split_training_test(data, labels, 0.8)

    #####################

    cor = np.corrcoef(du.transpose(RNN.flatten_sequence(data)))
    du.writetoCSV(cor, "correlationMat")

    covariates = []
    n_cov = du.len_deepest(data)
    print "Covariates before filtering:", n_cov

    for i in range(0, len(cor)):
        valid = True
        for j in range(i + 1, len(cor[i])):
            if (np.floor(10 * cor[i][j])) * .1 > 1:
                valid = False
                break
        if valid:
            covariates.append(i)

    print "Covariates after filtering:", len(covariates)

    #####################

    RNN.print_label_distribution(labels, ["Confused", "Concentrating", "Bored", "Frustrated"])

    net = RNN(variant)
    net.set_hyperparams(recurrent_nodes, batch_size=batches, num_folds=2, num_epochs=epochs, step_size=step_size,
                         dropout1=dropout1, dropout2=dropout2)

    RNN.print_label_distribution(labels, ["Confused", "Concentrating", "Bored", "Frustrated"])
    net.set_training_params(batches, epochs, balance=balance_model, scale_output=scale_output,
                             early_stop=early_stop)

    net.train_stable(data, labels, covariates=covariates, holdout_samples=holdout, holdout_labels=hold_label)

    ###
    # pickle and save RNN model as model_file
    ###
    RNN.save_model(net, model_file)
    return net


def apply_model(model_file, file, output):
    NET = None
    try:
        ###
        # attempt to load pickled model from model_file
        ###
        NET = RNN.load_model(model_file)
    except:
        NET = train_and_save_model(model_file, 200, 1, 100, dropout1=0.0, dropout2=0.3, step_size=0.001, balance_model=False,
                             scale_output=True, early_stop=True, variant="LSTM")

    confidence_table = []
    model_cov = range(4, 96)
    test, student, unformatted_data = RNN.load_unlabeled_data(file, 3, 1, model_cov)
    pred = NET.predict(test)
    ft = unformatted_data

    for k in range(0, len(pred)):
        confidence_table.append([ft[k][0], ft[k][2], ft[k][3], pred[k][0], pred[k][1], pred[k][2], pred[k][3]])

    # du.writetoCSV(confidence_table, output, ['clip', 'problem_log_id', 'user_id', 'Confused', 'Concentrating',
    #                                          'Bored', 'Frustrated'])
    return confidence_table


def aggregate_estimates(confidence_table, outfile):
    # assumes data in the format of: clip, problem log id, user id, confusion, concentration, boredom, frustrated
    estimates = np.array(confidence_table)
    students = np.array(np.unique(estimates[:, 2]))

    agg_table = []

    for s in students:
        subset = estimates[np.where(estimates[:, 2] == s)]
        m = np.mean(np.array(subset[:, 3:], dtype=float), axis=0)
        agg_table.append([s, len(subset), m[0], m[1], m[2], m[3]])

    du.writetoCSV(agg_table, outfile, ['user_id', 'n_clips', 'Confusion', 'Concentration', 'Boredom', 'Frustration'])

