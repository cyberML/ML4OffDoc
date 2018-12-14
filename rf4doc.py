import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve

# source: git@github.com:cyberML/ML4OffDoc.git

# number of arguments
NUM_ARGS = 2

NoOfTrees = 100
RSEED = 50

# if you don't have anti-virus detection information in your feed, you can comment this out.
avs = [
        'AV1',
        'AV2',
        'AV3',
        'AV4',
        'AV5',
        'AV6',
        'AV7'
]

def get_features_from_df(df):
    df_columns = df.columns
    dtypes = df.dtypes

    candidate_features = [c for c, dtype in zip(df_columns, df.dtypes.values) if dtype == np.int64]
    for av in avs:
        if av in candidate_features:
            candidate_features.remove(av)
    if 'LABEL' in candidate_features:
        candidate_features.remove('LABEL')

    return candidate_features

def usage(scriptFName):
    print('usage: ' + scriptFName + ' <feed_path>')

def main(args=[]):
    argLen = len(args)
    if argLen<NUM_ARGS :
        usage(os.path.basename(args[0]))
        return
    elif argLen>NUM_ARGS:
        for arg in args[NUM_ARGS:]:
            if arg == '-?' or arg == '-h' or arg == '--h':
                usage(os.path.basename(args[0]))
                return
            else:
                print('Unrecognized arg: ' + arg)
                usage(os.path.basename(args[0]))
                return

    db_path   = args[1]

    print('train feed: {}'.format(db_path))
    df = pd.read_csv(db_path)
    labels = df['LABEL']
    index = df['SHA1']
    n_obs = len(df)
    
    n_cols = df.shape[1]
    
    feature_columns = get_features_from_df(df)
    df_features = df[feature_columns]
    
    split_ix = int(n_obs * 0.9)
    train_X = df_features[df_features.index < split_ix]
    train_Y = labels[labels.index < split_ix]
    test_X = df_features[df_features.index >= split_ix]
    test_Y = labels[labels.index >= split_ix]
    test_ix = df.index[df.index >= split_ix]
    
    print('\n\nNo of trees: {}'.format(n))
    classifier = RandomForestClassifier(n_estimators=noOfTrees, 
                                        random_state = RSEED,
                                        max_features = 'sqrt',
                                        n_jobs=-1, verbose = 1)
    classifier.fit(train_X, train_Y)
    preds = classifier.predict(test_X)
    soft_preds = classifier.predict_proba(test_X)
    soft_preds = np.array([sp[1] for sp in soft_preds])
    
    classifier_accuracy = accuracy_score(test_Y, preds)
    c_fpr, c_tpr, c_thresholds = roc_curve(test_Y, soft_preds)
    classifier_auc = auc(c_fpr, c_tpr)
    
    vendor_acc = []
    for vendor in avs:
        vendor_preds = df.loc[test_ix][vendor]
        vendor_accuracy = accuracy_score(test_Y, vendor_preds.values)
    
        vendor_acc.append(vendor_accuracy)
        print('Vendor {} accuracy: {:.4f}'.format(vendor, vendor_accuracy))
    
    print('Classifier {} accuracy: {:.4f}'.format(type(classifier), classifier_accuracy))
    print('Classifier {} auc:{:.4f}'.format(type(classifier), classifier_auc))


if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print('\n\nAborted')

