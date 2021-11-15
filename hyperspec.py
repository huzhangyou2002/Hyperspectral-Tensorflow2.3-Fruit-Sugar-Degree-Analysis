import csv
import tensorflow as tf

#经过实验分析
#可见光和红外的算法 有明显差异

#还有待测试水果品种的参数
#还有 测试的糖度数据的自身准确性

#红外参数

filename = "./hyperspec.csv"
all_data_len = 563
train_data_len = 500
test_data_len = 60

#可见光参数
#filename = "./fs_hw_i.csv"
#all_data_len = 1016
#train_data_len = 900
#test_data_len = 100


#可以优化的参数
#Sequential 里面的层数量，层参数
#fit里面的epochs

def load_hyperspec_data(filename):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    pred_x = []
    pred_y = []
    i = 0
    with open(filename) as f:
        line_csv = csv.reader(f)
        for rows in line_csv:
            if(i < train_data_len):
                # 转化为float数值
                value = map(float,rows[:-1])
                train_x.append(list(value))
                # 转化为float数值
                float_y = float(rows[-1:][0])
                train_y.append(float_y)
            elif(i > train_data_len + test_data_len -1):

                value = map(float, rows[:-1])
                pred_x.append(list(value))
                # 转化为float数值
                float_y = float(rows[-1:][0])
                pred_y.append(float_y)
            else:
                # 转化为float数值
                value = map(float, rows[:-1])
                test_x.append(list(value))
                #转化为float数值
                float_y = float(rows[-1:][0])
                test_y.append(float_y)
            i = i + 1
    return train_x,train_y,test_x,test_y,pred_x,pred_y

def construct_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1)
    ])
    return model

tr_x,tr_y,t_x,t_y,p_x,p_y = load_hyperspec_data(filename)
model = construct_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())
history = model.fit(tr_x, tr_y, batch_size=50, epochs=500,validation_data=(t_x, t_y), validation_freq=1)
model.summary()
print(model.predict(p_x))
print(p_y)