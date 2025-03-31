
__all__ = ['Config']


class Config():
    seed = 2020
    MAP_CUTOFF = 14
    DIST_NORM = 15
    hidden = 256
    dropout = 0.1
    alpha = 0.7
    LAMBDA = 1.5

    learning_rate = 1E-3
    weight_decay = 0
    batch_size = 1
    num_workers = 4
    num_classes = 2  # [not bind, bind]
    epochs = 50  # 20 for PLMs

    feature_path = "./Feature/"
    graph_path = './Graph/'
    center = 'SC/'
    Test60_psepos_path = './Feature/psepos/Test60_psepos_SC.pkl'
    Test315_28_psepos_path = './Feature/psepos/Test315-28_psepos_SC.pkl'
    Btest31_psepos_path = './Feature/psepos/Test60_psepos_SC.pkl'
    UBtest31_28_psepos_path = './Feature/psepos/UBtest31-6_psepos_SC.pkl'
    Train335_psepos_path = './Feature/psepos/Train335_psepos_SC.pkl'
    dataset_path = "./Dataset/"
    virtual_node = 3
    AlphaFold3_pred = False
    is_handcrafted_feature_group = True  # if true, train handcrafted, else PLMs

    test_type = 1  # change test dataset type, 1 -> Test_60, 2 -> Test_315-28, 3 -> BTest_31-6, 4 -> UBtest_31-6