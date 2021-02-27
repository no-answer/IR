# IR
任务定义
给定一个文档集D={d1,d2,d3...dn}和一个查询q，输出对文档集D的full ranking结果。

数据集
目前给定文档集D，训练集与验证集，训练集和验证集中包含query与对应的document label，注意一个query可能有多个对应的document。
在期末最终测试之前请尽可能地提升模型检索的精度。期末测试的时候将会释放测试集对模型进行最终测试，期末之前测试集不可见。
文档集文档数量	训练集查询数量	验证集查询数量	测试集查询数量
50万	3万	0.3万	0.3万
文档集D，训练集，验证集和测试集分别对应以下三个文件：documents.json，trainingset.json ，validationset.json和testset.json。
以下为数据集文件的格式。
文件名称	格式
documents.json	{d1_id: d1_text, d2_id: d2_text, ...}
trainingset.json	{ queries: {q1_id: q1_text, q2_id: q2_text, ...},
labels: {q1_id: [dx_id,...], q2_id2: [dx_id,...], ...} }
validationset.json	{ queries: {q1_id: q1_text, q2_id: q2_text, ...},
labels: {q1_id: [dx_id,...], q2_id2: [dx_id,...], ...} }
testset.json	{ queries: {q1_id: q1_text, q2_id: q2_text, ...}, }

基线模型
我们使用BM25作为基线模型在数据集验证集用指标MRR@10，NDCG@10进行了测试，结果作为参考，请见下表。
 	MRR@10	NDCG@10
BM25	20.31	17.84

