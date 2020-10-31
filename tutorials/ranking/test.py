import torch
import numpy as np
import pandas as pd
import matchzoo as mz

relation = pd.read_csv("../../../robust04/qrels.rob04.txt", sep=" ",
                      names = ["id_left", "dummy", "id_right", "label"]).drop("dummy",axis=1)
relation['label'] = relation['label'].astype(float)
relation = relation.dropna(axis=0, how='any')
relation.id_left = relation.id_left.astype("str")
relation.id_right = relation.id_right.astype("str")

left = pd.read_csv("../../../robust04/query_trec45_no.txt", sep=":",
                  names = ["id_left", "text_left"])
left = left.dropna(axis=0, how='any')
left.id_left = left.id_left.astype("str")

doc_dict = dict()
with open("../../../robust04/trec45.doc.txt") as f:
    for line in f:
        info = line.split("\t")
        if len(info) == 4:
            text = info[2] + " ".join(info[3].split(" ")[:500])
        else:
            text = " ".join(info[2].split(" ")[:500])
        doc_dict[info[1]] = text
        
right = pd.DataFrame(list(doc_dict.items()),columns = ['id_right','text_right']) 
right = right.dropna(axis=0, how='any')
right.id_right = right.id_right.astype("str")
right_id_right = right.id_right.drop_duplicates()

train_id_left = pd.read_csv('../../../cedr/data/robust/f1.train.pairs', sep = "\t",
                         names = ["id_left", "id_right"]).id_left.astype("str").drop_duplicates()
vali_id_left = pd.read_csv('../../../cedr/data/robust/f1.valid.run', sep = " ",
                         names = ["id_left", "dummy1", "id_right", "rank", "score", "dummy2"]).id_left.astype("str").drop_duplicates()
test_id_left = pd.read_csv('../../../cedr/data/robust/f1.test.run', sep = " ",
                         names = ["id_left", "dummy1", "id_right", "rank", "score", "dummy2"]).id_left.astype("str").drop_duplicates()

relation_train = pd.merge(pd.merge(relation,train_id_left, on=["id_left"], how="inner"), 
                          right_id_right, how = "inner")[["id_left", "id_right", "label"]]
relation_vali = pd.merge(pd.merge(relation,vali_id_left, on=["id_left"], how="inner"), 
                          right_id_right, how = "inner")[["id_left", "id_right", "label"]]
relation_test = pd.merge(pd.merge(relation,test_id_left, on=["id_left"], how="inner"), 
                          right_id_right, how = "inner")[["id_left", "id_right", "label"]]

left_train = pd.merge(train_id_left, left, how="inner", on="id_left")[["id_left", "text_left"]]
left_train.set_index("id_left", inplace=True)
left_vali = pd.merge(vali_id_left, left, how="inner", on="id_left")[["id_left", "text_left"]]
left_vali.set_index("id_left", inplace=True)
left_test = pd.merge(test_id_left, left, how="inner", on="id_left")[["id_left", "text_left"]]
left_test.set_index("id_left", inplace=True)

right_train = pd.merge(relation_train.id_right.drop_duplicates(), right, how="inner", on="id_right")[["id_right", "text_right"]].drop_duplicates()
right_train.set_index("id_right", inplace=True)
right_vali = pd.merge(relation_vali.id_right.drop_duplicates(), right, how="inner", on="id_right")[["id_right", "text_right"]].drop_duplicates()
right_vali.set_index("id_right", inplace=True)
right_test = pd.merge(relation_test.id_right.drop_duplicates(), right, how="inner", on="id_right")[["id_right", "text_right"]].drop_duplicates()
right_test.set_index("id_right", inplace=True)

print('data loading ...')
train_pack_raw = mz.DataPack(relation=relation_train,left=left_train,right=right_train)
dev_pack_raw = mz.DataPack(relation=relation_vali,left=left_vali,right=right_vali)
test_pack_raw = mz.DataPack(relation=relation_test,left=left_test,right=right_test)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
#ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

preprocessor = mz.preprocessors.BasicPreprocessor(
    truncated_length_left = 10,
    truncated_length_right = 500,
    #filter_low_freq = 2
)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup = 1,
    num_neg=1
)
testset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed
)

padding_callback = mz.models.KNRM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    #batch_size=20,
    stage='train',
    #resample=True,
    #sort=False,
    callback=padding_callback,
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    #batch_size=20,
    stage='dev',
    callback=padding_callback
)

model = mz.models.KNRM()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['kernel_num'] = 21
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adadelta(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=10
)

trainer.run()

train.save_model()