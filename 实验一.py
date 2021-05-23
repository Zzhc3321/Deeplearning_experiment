import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# 画图
from pyecharts.charts import Bar, Grid, Line, Liquid, Page, Pie, Bar3D
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

# 鸢尾花数据集
from sklearn import datasets

#超参
optimizers = ['Adagrad','Adam','SGD']
batch_sizes = [10,20,30,40]
epochs_ = [300,500,700]
validation_split = [0.1,0.2,0.3]
is_shuffle = [True,False]

iris = datasets.load_iris()
y_train = iris.target
x_train = iris.data

# 动态分配显存
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

page_accurcy = Page(layout=Page.SimplePageLayout)
page_loss = Page(layout=Page.SimplePageLayout)

total_acc = 0
total_val_acc = 0
total_loss = 100
total_val_loss = 100
best_data = [[] for i in range(4)]
best_p = ['' for i in range(4)]
names = ['acc最高','val_acc最高','loss最低','val_loss最低']
n = ['accuracy','val_accuracy','loss','val_loss']

for op in optimizers:
    for b_size in batch_sizes:
        for e in epochs_:
            for split in validation_split:
                for is_s in is_shuffle:
                    chart_name = 'optimizer:'+str(op)+'batch_sizes:'+str(b_size)+'\n'+'epochs:'+str(e)+'validation_split:'+str(split)+'is_shuffle:'+str(is_s)
                    chart_name_ = 'optimizer:'+str(op)+'batch_sizes:'+str(b_size)+'epochs:'+str(e)+'validation_split:'+str(split)+'is_shuffle:'+str(is_s)
                    print(chart_name)
                    model = keras.Sequential(
                        [
                            layers.Flatten(input_shape=[4]),
                            layers.Dense(3, activation='softmax')
                        ])
                    model.compile(optimizer=op,
                                 loss='sparse_categorical_crossentropy',
                                 metrics=['accuracy'])
                    history = model.fit(x_train, y_train, batch_size=b_size,epochs=e, validation_split=split,shuffle=is_s)
                    xaxis_1 = [i for i in range(len(history.history['accuracy']))]
                    xaxis_2 = [i for i in range(len(history.history['loss']))]

                    line_accuracy_chart = (
                        Line(init_opts=opts.InitOpts(page_title='accuracy', height='600px', width='100%'))
                            .add_xaxis(xaxis_1)
                            .add_yaxis("training_acc", history.history['accuracy'])
                            .add_yaxis("valivation_acc", history.history['val_accuracy'])
                            .set_series_opts(label_opts=opts.LabelOpts(is_show=False, font_size=5),
                                             linestyle_opts=opts.LineStyleOpts(width=1, curve=0.1), )
                            .set_global_opts(
                            title_opts=opts.TitleOpts(title=chart_name),
                            tooltip_opts=opts.TooltipOpts(trigger="axis"),
                            toolbox_opts=opts.ToolboxOpts(is_show=True),
                            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                        )
                    )
                    page_accurcy.add(line_accuracy_chart)
                    line_loss_chart = (
                        Line(init_opts=opts.InitOpts(page_title='accuracy', height='600px', width='100%'))
                            .add_xaxis(xaxis_2)
                            .add_yaxis("training_loss", history.history['loss'])
                            .add_yaxis("valivation_loss", history.history['val_loss'])
                            .set_series_opts(label_opts=opts.LabelOpts(is_show=False, font_size=5),
                                             linestyle_opts=opts.LineStyleOpts(width=1, curve=0.1), )
                            .set_global_opts(
                            title_opts=opts.TitleOpts(title=chart_name),
                            tooltip_opts=opts.TooltipOpts(trigger="axis"),
                            toolbox_opts=opts.ToolboxOpts(is_show=True),
                            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                        )
                    )
                    page_loss.add(line_loss_chart)
                    if history.history['accuracy'][-1]>total_acc:
                        total_acc=history.history['accuracy'][-1]
                        best_data[0] = history
                        best_p[0]= chart_name
                    if history.history['val_accuracy'][-1]>total_val_acc:
                        total_val_acc=history.history['val_accuracy'][-1]
                        best_data[1] = history
                        best_p[1] = chart_name
                    if history.history['loss'][-1]<total_loss:
                        total_loss=history.history['loss'][-1]
                        best_data[2] = history
                        best_p[2] = chart_name
                    if history.history['val_loss'][-1]<total_val_loss:
                        total_val_loss=history.history['val_loss'][-1]
                        best_data[3] = history
                        best_p[3] = chart_name

for i in range(4):
    xaxis_ = [i for i in range(len(best_data[i].history[n[i]]))]
    page = Page(layout=Page.SimplePageLayout)
    line1 = (
        Line(init_opts=opts.InitOpts(page_title=best_p[i], height='600px', width='100%'))
            .add_xaxis(xaxis_)
            .add_yaxis("training_acc", best_data[i].history['accuracy'])
            .add_yaxis("valivation_acc", best_data[i].history['val_accuracy'])
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False, font_size=5),
                             linestyle_opts=opts.LineStyleOpts(width=1, curve=0.1), )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=best_p[i]),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )
    line2 = (
        Line(init_opts=opts.InitOpts(page_title=best_p[i], height='600px', width='100%'))
            .add_xaxis(xaxis_)
            .add_yaxis("training_loss", best_data[i].history['loss'])
            .add_yaxis("valivation_loss", best_data[i].history['val_loss'])
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False, font_size=5),
                             linestyle_opts=opts.LineStyleOpts(width=1, curve=0.1), )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=best_p[i]),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )
    page.add(line1,line2)
    page.render('res_charts/'+names[i]+'.html')

page_accurcy.render('res_charts/page_accuracy.html')
page_loss.render('res_charts/page_loss.html')