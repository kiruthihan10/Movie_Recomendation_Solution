import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.random.set_seed(10)
np.random.seed(10)

train_data= tfds.load("movie_lens/100k-ratings",split=["train"],batch_size=-1,shuffle_files=True)[0]

print(train_data.keys())

movie_genres = np.array(train_data["movie_genres"])

numerical = np.array(train_data["raw_user_age"],dtype=np.float32)
numerical = numerical.reshape(numerical.shape[0],1)
#numerical=np.append(numerical,np.array(train_data["raw_user_age"],dtype=np.float32).reshape(numerical.shape[0],1),axis=1)
numerical=np.append(numerical,np.array(train_data["timestamp"],dtype=np.float32).reshape(numerical.shape[0],1),axis=1)
numerical=np.append(numerical,np.array(train_data["user_gender"],dtype=np.float32).reshape(numerical.shape[0],1),axis=1)
print(numerical.shape)

occupation = np.array(train_data["user_occupation_label"])

title = np.array(train_data["movie_title"])

zip_code = np.array(train_data["user_zip_code"])

rating = np.array(train_data["user_rating"])

print("Highest Rating is : {}".format(np.amax(rating)))
print("Lowest Rating is : {}".format(np.amin(rating)))

c = list(zip(movie_genres,numerical,occupation,title,zip_code,rating))

np.random.shuffle(c)
movie_genres,numerical,occupation,title,zip_code,rating = zip(*c)

movie_genres = np.array(movie_genres)
numerical = np.array(numerical)
occupation = np.array(occupation)
title = np.array(title)
zip_code = np.array(zip_code)
rating = np.array(rating)

print(zip_code[:10])
print(movie_genres[:10])
print(title[:10])
print(numerical[:10])
print(occupation[:10])
print(str("*"*100))

val_split = 0.2

def splitter(arr,value=0.2):
    value = 1-value
    return arr[:int(arr.shape[0]*value)],arr[-int(arr.shape[0]*value):]

movie_genres , test_movie_genres = splitter(movie_genres)

numerical, test_numerical = splitter(numerical)

occupation, test_occupation = splitter(occupation)

title, test_title = splitter(title)

zip_code, test_zip_code = splitter(zip_code)

rating, test_rating = splitter(rating)

def converter(arr):
  new_arr = []
  for i in range(arr.shape[0]):
      new_line = arr[i].decode(encoding="UTF-8",errors='ignore')
      new_arr.append(new_line)
  return np.array(new_arr)
title = converter(title)
test_title = converter(test_title)
zip_code = converter(zip_code)
test_zip_code = converter(test_zip_code)
def text_preprocessing(t1,t2,char_level=False):
    title_enc = keras.preprocessing.text.Tokenizer(char_level=char_level)
    title_enc.fit_on_texts(t1)
    t1 = title_enc.texts_to_sequences(t1)
    t1 = keras.preprocessing.sequence.pad_sequences(t1)
    max_length = t1.shape[1]
    t2 = title_enc.texts_to_sequences(t2)
    t2 = keras.preprocessing.sequence.pad_sequences(t2,max_length)
    if char_level:
        max_class = np.amax([t1,t2])+1
        t1 = keras.utils.to_categorical(t1,max_class)[:,1:]
        t2 = keras.utils.to_categorical(t2,max_class)[:,1:]
    return t1,t2

title,test_title = text_preprocessing(title,test_title)
zip_code,test_zip_code = text_preprocessing(zip_code,test_zip_code,True)

emb_dim = np.amax([title,test_title])

print("Embedding Dim is {}".format(emb_dim))

for col in range(numerical.shape[1]):
    mean = np.mean(numerical[:,col])
    std = np.std(numerical[:,col])
    numerical[:,col] = (numerical[:,col]-mean)/std
    test_numerical[:,col] = (test_numerical[:,col]-mean)/std

occupation_max_class = np.amax([occupation,test_occupation])+1

occupation = keras.utils.to_categorical(occupation,occupation_max_class)
test_occupation = keras.utils.to_categorical(test_occupation,occupation_max_class)

max_number = np.amax([movie_genres,test_movie_genres])

def numbers_to_vector(arr):
  out = np.zeros((int(arr.shape[0]),int(max_number)))
  for i in range(arr.shape[0]):
    for n in arr[i]:
      if n > 0:
        out[i][n-1]=1
  return out

movie_genres = numbers_to_vector(movie_genres)
test_movie_genres = numbers_to_vector(test_movie_genres)

print(movie_genres.shape)
print(occupation.shape)
print(zip_code.shape)
print(title.shape)
print(numerical.shape)

max_rating = np.amax(rating)

rating = rating/max_rating

test_rating = rating/max_rating

rating_mean =np.mean(rating)

#rating_std = np.std(rating)

#rating = (rating-rating_mean)/rating_std

#test_rating = (test_rating-rating_mean)/rating_std


def base_bias(shape,dtype=None):
    return np.array([np.mean(rating)])


batch_size = 256

ds = tf.data.Dataset.from_tensor_slices(({
    'numerical':numerical,
    'zip':zip_code,
    'title':title,
    'genre':movie_genres,
    'occupation':occupation},np.array(rating))).shuffle(1024).cache().batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(({
    'numerical':test_numerical,
    'zip':test_zip_code,
    'title':test_title,
    'genre':test_movie_genres,
    'occupation':test_occupation},test_rating)).shuffle(1024).cache().batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

mirrored_strategy = tf.distribute.MirroredStrategy()

neurons_in_connecter = 256

def real_loss(y_true,y_pred):
    return keras.losses.mean_absolute_error(y_true,y_pred)*max_rating
    #return (keras.losses.mean_absolute_error(y_true,y_pred)*rating_std)

def make_model():

    drop = keras.layers.GaussianDropout
    l2 = keras.regularizers.l2
    l1_l2 = keras.regularizers.l1_l2
    he_init = keras.initializers.he_normal
    elu_weight_init = keras.initializers.VarianceScaling(scale=2,mode="fan_avg")
    relu = tf.nn.relu
    add = tf.math.add

    ##########################################################################################
    numerical_input = keras.Input(numerical.shape[1],name='numerical',batch_size=batch_size)#1D

    zip_input = keras.Input(zip_code.shape[1:],name='zip',batch_size=batch_size)#2D (5, 27)

    title_input = keras.Input(title.shape[1],name='title',batch_size=batch_size)#1D

    movie_genres_input = keras.Input(movie_genres.shape[1],name='genre',batch_size=batch_size)#2D (6, 20)

    occupation_input = keras.Input(occupation.shape[1:],name='occupation',batch_size=batch_size)#1D

    ##########################################################################################
    movie_genres_model = keras.Sequential([
        keras.layers.Dense(128,activation="swish"),
        drop(0.3),
        keras.layers.Dense(movie_genres.shape[1],activation="elu",kernel_initializer=he_init)
    ])
    movie_genres_adder = add(movie_genres_model(movie_genres_input),movie_genres_input)
    movie_genres_connecter = keras.layers.BatchNormalization()(keras.layers.Dense(neurons_in_connecter,activation="elu",kernel_initializer=he_init)(relu(movie_genres_adder)))
    
    ##########################################################################################
    zip_model = keras.Sequential([
        keras.layers.Conv1D(64,3,1),
        drop(0.3),
        keras.layers.Conv1D(128,2),
        keras.layers.Flatten(),
        keras.layers.Lambda(keras.activations.swish),
        drop(0.3),
        keras.layers.Dense(neurons_in_connecter,"elu",kernel_initializer=he_init)
    ])
    zip_connecter = keras.layers.BatchNormalization()(zip_model(zip_input))

    ##########################################################################################
    title_model = keras.Sequential([
        keras.layers.Embedding(emb_dim+1,128,mask_zero=True,input_length=title.shape[1]),
        keras.layers.LSTM(128,recurrent_dropout=0.1,unroll=True,),
        keras.layers.Lambda(keras.activations.swish),
        drop(0.3),
        keras.layers.Dense(neurons_in_connecter,"elu",kernel_initializer=he_init)
    ])
    
    title_connecter = keras.layers.BatchNormalization()(title_model(title_input))
    ##########################################################################################

    numerical_model = keras.Sequential([
        keras.layers.Dense(128,activation="swish"),
        drop(0.3),
        keras.layers.Dense(numerical_input.shape[1],activation="elu",kernel_initializer=he_init)
    ])
    numerical_adder = add(numerical_model(numerical_input),numerical_input)
    numerical_connecter = keras.layers.BatchNormalization()(keras.layers.Dense(neurons_in_connecter,activation="elu",kernel_initializer=he_init)(relu(numerical_adder)))
    ##########################################################################################
    occupation_model = keras.Sequential([
        keras.layers.Dense(128,activation="swish"),
        drop(0.3),
        keras.layers.Dense(occupation_input.shape[1],activation="elu",kernel_initializer=he_init)
    ])
    occupation_adder = add(occupation_model(occupation_input),occupation_input)
    occupation_connecter = keras.layers.BatchNormalization()(keras.layers.Dense(neurons_in_connecter,activation="elu",kernel_initializer=he_init)(relu(occupation_adder)))
    ##########################################################################################
    connected = keras.layers.concatenate([
        occupation_connecter,
        title_connecter,
        zip_connecter,
        movie_genres_connecter,
        numerical_connecter])

    main_model_1 = keras.Sequential([
        keras.layers.BatchNormalization(),
        drop(0.25),
        keras.layers.Dense(1024,activation="elu",kernel_initializer=he_init,kernel_regularizer=l1_l2(1e-4,1e-4)),
        keras.layers.BatchNormalization(),
        drop(0.225),
    ])

    out_1 = main_model_1(connected)

    main_model_2 = keras.Sequential([
        keras.layers.Dense(1024,activation="elu",kernel_initializer=he_init,kernel_regularizer=l2(1e-4)),
        keras.layers.BatchNormalization(),
        drop(0.2),
    ])

    out_2 = main_model_2(out_1)

    out_add = tf.nn.tanh(sum(out_1,out_2))

    main_model_out = keras.Sequential([
        keras.layers.BatchNormalization(),
        drop(0.2),
        keras.layers.Dense(1,activation="sigmoid",bias_initializer=base_bias)
    ])

    out = main_model_out(out_add)

    model = keras.Model([numerical_input,zip_input,title_input,movie_genres_input,occupation_input],out)

    return model

with mirrored_strategy.scope():
    model = make_model()

lr = 1e-3

opt = keras.optimizers.Nadam(lr,0.99)

model.compile(optimizer=opt,loss="mse",metrics=[real_loss])

epochs = 200

def exponential_decay_fn(epoch, lr):
    print(lr)
    return lr*1e-2**(1/epochs)

exponential_decay_fn= tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=rating.shape[0]//batch_size*5,
    decay_rate=0.01**(1/40),
    staircase=True)

cosine = tf.keras.experimental.LinearCosineDecay(lr,epochs*rating.shape[0]//batch_size)

ploynomial_decay = keras.optimizers.schedules.PolynomialDecay(lr,epochs*rating.shape[0]//batch_size,power=0.5,end_learning_rate=1e-6)

lr_schedule = keras.callbacks.LearningRateScheduler(exponential_decay_fn,verbose=1)

lr_callback = keras.callbacks.ReduceLROnPlateau(patience=epochs//20,verbose=1)

early_stop = keras.callbacks.EarlyStopping(monitor="val_real_loss",patience=epochs//10,restore_best_weights=True)

history = model.fit(ds,validation_data=test_ds,epochs=epochs,workers=4,callbacks=[lr_callback, early_stop,lr_schedule],use_multiprocessing=True)

model.evaluate(test_ds)

print(model.predict(test_ds))

loss = history.history['real_loss']
val_loss = history.history['val_real_loss']

epochs_range = list(range(int(len(loss))))

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

