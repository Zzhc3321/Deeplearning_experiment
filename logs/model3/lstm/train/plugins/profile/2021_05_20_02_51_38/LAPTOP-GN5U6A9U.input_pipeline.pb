  *?????<c@?????5?@2?
VIterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCacheImpl::ParallelMapV2q?-???!ZّXc.@)q?-???1ZّXc.@:Preprocessing2U
Iterator::Model::PaddedBatchV2RI??&???!?^??8@)?W?2ı??1m??&8(,@:Preprocessing2~
GIterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCacheImpl??????!??{{?=@)Gx$(??1ؕe??+@:Preprocessing2?
iIterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCacheImpl::ParallelMapV2::AssertCardinalityё\?C???!H???}0@)'?W???1H?)i?(@:Preprocessing2?
?Iterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap?a??4???!}??243@)?f??j+??1???*EV$@:Preprocessing2?
?Iterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord?\m?????!??1 "@)?\m?????1??1 "@:Advanced file read2h
1Iterator::Model::PaddedBatchV2::Shuffle::Prefetchx$(~??!?\C3#@)x$(~??1?\C3#@:Preprocessing2F
Iterator::Model?p=
ף??!?r ?j?>@)?_vO??1??9??@:Preprocessing2?
Iterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCacheImpl::ParallelMapV2::AssertCardinality::ParallelInterleaveV4??d?`T??!??}?@)??d?`T??1??}?@:Preprocessing2^
'Iterator::Model::PaddedBatchV2::Shuffle?:pΈ???!-?????$@)???x?&??1?Z?9?w@:Preprocessing2m
6Iterator::Model::PaddedBatchV2::Shuffle::Prefetch::Mapq???h??!%?lAG?@@)u????1???
/@:Preprocessing2z
CIterator::Model::PaddedBatchV2::Shuffle::Prefetch::Map::MemoryCache??"??~??!??z??C>@)^K?=???1?Q?_???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.