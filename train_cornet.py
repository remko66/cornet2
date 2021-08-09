import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as plt
from base_cornet import create_res_net
from base_cornet import expand_res_net


# needed to avoid a wierd dnn error in tensorflow on my hardware
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)




def train_batch(model,epochs=10,batchsize=64,stepsperupdate=500):
    """
    Stanar training loop, makes graphs of loss en accuracy

    :param model: the model (dah)
    :param epochs: number of epoch
    :param batchsize: batchsize
    :param stepsperupdate: number of steps before putting in a new block (linear here, but in real live better solutions are available)
    :return: absolutely nothing. Just make to print out training data. But can give back to model to work with or inference offcourse
    """
    steps = 1
    block=1
    plt.ion()
    loss=[]
    acc=[]
    graphx=[]
    count = 0
    model.summary()
    batchnr=int(len(x_train)/batchsize)+1
    start = time.time()
    for epoch in range(epochs):
        for batch in range(batchnr):
            xt=x_train[batch*batchsize:(batch+1)*batchsize]
            yt=y_train[batch*batchsize:(batch+1)*batchsize]

            res=model.train_on_batch(
                xt,
                yt,
                sample_weight=None,
                class_weight=None,
                reset_metrics=False,
                return_dict=False,
            )
            graphx.append(count)
            loss.append(res[0])
            acc.append(res[1])
            if count %100==0:
                fig = plt.figure(1)
                plt.plot(graphx, loss)
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.canvas.set_window_title('loss')

                fig = plt.figure(2)
                plt.plot(graphx, acc)
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.canvas.set_window_title('acc')
            steps+=1
            count+=1
            if steps>stepsperupdate:
                if block < 13:
                    model=expand_res_net(model,block,trainold=False)
                    block+=1
                    print("expanded!")
                steps=0
            print('\r',epoch+1, '/', epochs, '-', batch+1, '/', batchnr,'--',res[0],res[1],round(time.time()-start),block)
    plt.figure(1)
    plt.savefig('graphs/pronet_loss')
    plt.figure(2)
    plt.savefig('graphs/pronet_acc')
    model.evaluate(x_test,y_test,verbose=1)
    print("needed", time.time() - start)


#get cifar data (nice and preprocessed)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#create our model
model = create_res_net()
#trainit
train_batch(model,epochs=10)